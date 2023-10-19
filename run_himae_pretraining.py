import logging
import os
from pathlib import Path
import random
import numpy as np
import pickle
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

import torch
from torch import Tensor
from torch.nn import functional as F
from torch import distributed as dist

import transformers
from transformers import DataCollatorWithPadding
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM
from transformers.file_utils import ModelOutput
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from transformers import DataCollatorForWholeWordMask
from transformers.trainer_utils import is_main_process

from data import MINDDataset, MINDEvaluationDataset

from model import DualEncoder, NewsEncoder, UserEncoder, TransformerEncoder

from arguments import (
    ModelArguments,
    DataArguments,
    MINDTrainingArguments,
)

from trainer import MINDTrainer

from utils import tensorize_batch, AverageMeter, dist_gather_tensor

logger = logging.getLogger(__name__)

@dataclass
class MINDPretrainingArguments(MINDTrainingArguments):
    encoder_mlm_probability: float = field(default=0.3)
    decoder_mlm_probability: float = field(default=0.5)


class MINDPretrainingDataset(MINDDataset):
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return data["history_news"], data["history_mask"]


@dataclass
class HiMAECollator(DataCollatorForWholeWordMask):
    encoder_mlm_probability: float = 0.15
    decoder_mlm_probability: float = 0.15
    news_max_len: int = 128

    def __call__(self, examples):
        # flatten
        history_news = [e[0] for e in examples]
        history_masks = [e[1] for e in examples]
        
        examples = sum(history_news, []) # List[{"input_ids": [...]}]
        input_ids_batch = []
        attention_mask_batch = []
        encoder_mlm_mask_batch = []
        decoder_mlm_mask_batch = []

        for e in examples:
            tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e["input_ids"]]

            self.mlm_probability = self.encoder_mlm_probability
            text_encoder_mlm_mask = self._whole_word_mask(tokens)

            self.mlm_probability = self.decoder_mlm_probability
            text_decoder_mlm_mask = self._whole_word_mask(tokens)

            input_ids_batch.append(torch.tensor(e["input_ids"]))
            attention_mask_batch.append(torch.tensor([1] * len(e["input_ids"])))

            encoder_mlm_mask_batch.append(torch.tensor(text_encoder_mlm_mask))
            decoder_mlm_mask_batch.append(torch.tensor(text_decoder_mlm_mask))

        input_ids_batch = tensorize_batch(input_ids_batch, self.tokenizer.pad_token_id)
        attention_mask_batch = tensorize_batch(attention_mask_batch, 0)
        
        encoder_mlm_mask_batch = tensorize_batch(encoder_mlm_mask_batch, 0)
        decoder_mlm_mask_batch = tensorize_batch(decoder_mlm_mask_batch, 0)

        encoder_input_ids_batch, encoder_labels_batch = self.torch_mask_tokens(input_ids_batch.clone(), encoder_mlm_mask_batch)
        decoder_input_ids_batch, decoder_labels_batch = self.torch_mask_tokens(input_ids_batch.clone(), decoder_mlm_mask_batch)
        
        global_mask = torch.tensor(history_masks, dtype=torch.float) # (batch_size, num_history)

        batch = {
            "input_ids": encoder_input_ids_batch, # (batch_size * num_history, seq_len)
            "attention_mask": attention_mask_batch,
            "labels": encoder_labels_batch,
            "decoder_input_ids": decoder_input_ids_batch,
            "decoder_labels": decoder_labels_batch,
        }

        return {
            "history": batch,
            "history_mask": global_mask,
        }


@dataclass
class HiMAEOutput(ModelOutput):
    loss: Optional[Tensor] = None
    encoder_mlm_loss: Optional[float] = None
    decoder_mlm_loss: Optional[float] = None
    local_mlm_loss: Optional[float] = None
    global_mbm_loss: Optional[float] = None


class HiMAE(DualEncoder):
    def __init__(self, args):
        super(HiMAE, self).__init__(args)
        self.lm = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
        self.news_encoder.lm = self.lm
        
        # additional news decoder
        self.news_decoder = TransformerEncoder(self.lm.config, args.n_head_layers)
        self.news_decoder.apply(self.lm._init_weights)
    
    def gradient_checkpointing_enable(self):
        self.lm.gradient_checkpointing_enable()
    
    def forward(self, history: Dict[str, Tensor], history_mask: Tensor):
        """Overwrite forward method since only involves unsupervised user behaviors
        If history_mask is None, only local mlm loss
        Args:
            history (Dict[str, Tensor]): (batch_size * num_history, max_seq_len)
            history_mask (Tensor): (batch_size, num_history)
        Returns:
            loss: (shape) ()
        """

        # local masked language modeling
        decoder_input_ids = history.pop("decoder_input_ids")
        decoder_labels = history.pop("decoder_labels")
        history_news_outputs = self.lm(**history, output_hidden_states=True, return_dict=True)
        encoder_mlm_loss = history_news_outputs.loss # encoder loss
        last_hidden_states = history_news_outputs.hidden_states[-1] # (batch_size * num_history, seq_len, news_dim)
        history_news_embeddings = self.news_encoder.pool_news_embedding(
            last_hidden_states,
            history["attention_mask"]
        ) # (batch_size * num_history, news_dim)
        # decoder loss
        decoder_embedding_output = self.lm.bert.embeddings(input_ids=decoder_input_ids)
        hiddens = torch.cat([history_news_embeddings.unsqueeze(1), decoder_embedding_output[:, 1:]], dim=1)
        # (batch_size * num_history, max_seq_len, news_dim)
        hiddens = self.news_decoder(hiddens, history["attention_mask"].type_as(hiddens))
        pred_scores, decoder_mlm_loss = self.mlm_loss(hiddens, decoder_labels)
        local_mlm_loss = encoder_mlm_loss + decoder_mlm_loss

        global_mbm_loss = torch.tensor(.0, dtype=local_mlm_loss.dtype, device=local_mlm_loss.device)
        # global contrastive masked behavior modeling
        if self.args.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            history_mask = dist_gather_tensor(history_mask)
            history_news_embeddings = dist_gather_tensor(history_news_embeddings)
        
        
        batch_size, num_history = history_mask.shape
        imputed_history_mask = history_mask.clone()
        imputed_history_mask[..., -2:] = 1 # avoid overflow in user encoder mean pooling
        # generate self news mask for each user in the batch
        user_history_news_embeddings = history_news_embeddings.reshape(batch_size, num_history, -1)
        behavior_mask = torch.eye(num_history).to(device=history_mask.device) # (num_history, num_history)
        behavior_mask = ~(behavior_mask.expand(size=(batch_size, num_history, num_history)).bool()) # (batch_size, num_history, num_history)
        
        # user to self viewed news
        user_self_behavior_mask = torch.eye(batch_size).to(device=history_mask.device).repeat_interleave(num_history, dim=1).bool() # (batch_size, batch_size * num_history)
        """ this tensor is like (batch_size=2, num_history=3)
        1 1 1 0 0 0
        0 0 0 1 1 1
        """
        # use repeat rather than expand here since original column size is not 1
        user_self_history_mask = F.one_hot(torch.arange(batch_size) * num_history, num_classes=batch_size*num_history).repeat(1, 2) # (batch_size, 2 * (batch_size * num_history))
        """this one looks like, repeat twice for narrowing
        1 0 0 0 0 0  1 0 0 0 0 0  
        0 0 0 1 0 0  0 0 0 1 0 0 
        """

        for i in range(num_history):
            new_mask = (behavior_mask[:, i].bool() & imputed_history_mask.bool()).type_as(history_mask) # (batch_size, num_history)
            masked_user_embeddings = self.user_encoder(user_history_news_embeddings, new_mask) # (batch_size, user_emb_dim)

            scores = masked_user_embeddings @ history_news_embeddings.T # (batch_size, batch_size * num_history)
            
            scores /= self.args.temperature
            labels = torch.arange(batch_size).to(scores.device) * num_history + i # (batch_size, )

            # mask out scores of news in the same behavior sequence for the user
            # target_news_mask = user_self_history_mask.narrow(dim=1, start=batch_size * num_history- i, end=2 * batch_size * num_history - i) # (batch_size, batch_size * num_history)
            # easy implementation
            target_news_mask = F.one_hot(labels, num_classes=batch_size*num_history)
            user_self_news_mask = target_news_mask.bool() ^ user_self_behavior_mask.bool()
            # import pdb; pdb.set_trace();
            scores = scores.masked_fill(user_self_news_mask, torch.finfo(scores.dtype).min)
            
            # mask padded news labels
            padded_news = history_mask[:, i] # (batch_size, ), 0 is padded news
            labels[~padded_news.bool()] = -100
            # print(scores)
            cl_loss = self.cross_entropy(scores, labels)
            # print(cl_loss.item())
            # import pdb; pdb.set_trace();
            global_mbm_loss = global_mbm_loss + cl_loss
        global_mbm_loss = global_mbm_loss / num_history

        # global_mbm_loss = torch.tensor(.0, dtype=local_mlm_loss.dtype, device=local_mlm_loss.device)
        loss = local_mlm_loss + global_mbm_loss

        return HiMAEOutput(
            loss=loss,
            encoder_mlm_loss=encoder_mlm_loss.item(),
            decoder_mlm_loss=decoder_mlm_loss.item(),
            local_mlm_loss=local_mlm_loss.item(),
            global_mbm_loss=global_mbm_loss.item(),
        )
    
    def mlm_loss(self, hiddens, labels):
        pred_scores = self.lm.cls(hiddens)
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.lm.config.vocab_size),
            labels.view(-1)
        )
        return pred_scores, masked_lm_loss

    def save_pretrained(self, output_path):
        super(HiMAE, self).save_pretrained(output_path)
        torch.save(self.news_decoder.state_dict(), os.path.join(output_path, 'news_decoder.pt'))

    def load_pretrained(self, output_path):
        super(HiMAE, self).load_pretrained(output_path)
        try:
            news_decoder_state_dict = torch.load(os.path.join(output_path, "news_decoder.pt"))
            self.news_decoder.load_state_dict(news_decoder_state_dict)
        except FileNotFoundError:
            logger.info("Can't find pretrained decoder weights. Using random initialization.")


class HiMAETrainer(MINDTrainer):
    def __init__(self, *args, **kwargs):
        super(HiMAETrainer, self).__init__(*args, **kwargs)
        self.enc_mlm_loss = AverageMeter('enc_mlm_loss', round_digits=3)
        self.dec_mlm_loss = AverageMeter('dec_mlm_loss', round_digits=3)
        self.local_mlm_loss = AverageMeter('local_mlm_loss', round_digits=3)
        self.global_mbm_loss = AverageMeter('global_mbm_loss', round_digits=3)
        self.last_epoch = 0
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        logs["step"] = self.state.global_step
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        
        # Add current enc and dec mlm loss to logs
        logs["enc_mlm_loss"] = self.enc_mlm_loss.value()
        logs["dec_mlm_loss"] = self.dec_mlm_loss.value()
        logs["local_mlm_loss"] = self.local_mlm_loss.value()
        logs["global_mbm_loss"] = self.global_mbm_loss.value()
        self._reset_meters_if_needed()

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        loss, outputs = super(HiMAETrainer, self).compute_loss(model, inputs, return_outputs=True)

        if self.model.training:
            self.enc_mlm_loss.update(outputs.encoder_mlm_loss)
            self.dec_mlm_loss.update(outputs.decoder_mlm_loss)
            self.local_mlm_loss.update(outputs.local_mlm_loss)
            self.global_mbm_loss.update(outputs.global_mbm_loss)

        return (loss, outputs) if return_outputs else loss

    def _reset_meters_if_needed(self):
        if int(self.state.epoch) != self.last_epoch:
            self.last_epoch = int(self.state.epoch)
            self.enc_mlm_loss.reset()
            self.dec_mlm_loss.reset()
            self.local_mlm_loss.reset()
            self.global_mbm_loss.reset()


class TrainerCallbackForSaving(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        control.should_save = True


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, MINDPretrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )


    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    if training_args.local_rank in (0, -1):
        logger.info("Training/evaluation parameters %s", training_args)
        logger.info("Model parameters %s", model_args)
        logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )

    train_dataset = MINDPretrainingDataset(data_args, tokenizer) if training_args.do_train else None

    eval_dataset = MINDEvaluationDataset(data_args, tokenizer) if training_args.do_eval else None

    data_collator=HiMAECollator(
        tokenizer,
        encoder_mlm_probability=training_args.encoder_mlm_probability,
        decoder_mlm_probability=training_args.decoder_mlm_probability,
        news_max_len=data_args.news_max_len,
    )
    
    model_args.temperature = training_args.temperature
    model_args.negatives_x_device = training_args.negatives_x_device
    model = HiMAE(model_args)
    
    model.load_pretrained(model_args.model_name_or_path)

    # import pdb; pdb.set_trace();
    
    trainer = HiMAETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
    
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset, metric_key_prefix="eval")
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
