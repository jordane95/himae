import logging
import os
import sys
from typing import Dict, Optional
from dataclasses import dataclass, field

import torch
from torch import nn, Tensor

from datasets import load_dataset
from transformers import DataCollatorForWholeWordMask

import transformers
from transformers import BertForMaskedLM, AutoModelForMaskedLM
from transformers.modeling_outputs import MaskedLMOutput, ModelOutput
from transformers.models.bert.modeling_bert import BertLayer
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    AutoConfig,
    HfArgumentParser,
    set_seed,
)
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import is_main_process
from model import TransformerEncoder

from utils import tensorize_batch
from utils import AverageMeter

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrain data"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated. Default to the max input length of the model."
        },
    )
    encoder_mlm_probability: float = field(default=0.3)
    decoder_mlm_probability: float = field(default=0.5)

    def __post_init__(self):
        if self.data_dir is not None:
            files = os.listdir(self.data_dir)
            self.train_path = [
                os.path.join(self.data_dir, f)
                for f in files
                if f.endswith('json')
            ]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default='bert-base-uncased',
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )

    n_head_layers: int = field(default=2)


@dataclass
class MAECollator(DataCollatorForWholeWordMask):
    max_seq_length: int = 512
    encoder_mlm_probability: float = 0.15
    decoder_mlm_probability: float = 0.15

    def __call__(self, examples):
        input_ids_batch = []
        attention_mask_batch = []
        encoder_mlm_mask_batch = []
        decoder_mlm_mask_batch = []

        tgt_len = self.max_seq_length - self.tokenizer.num_special_tokens_to_add(False)

        for e in examples:
            e_trunc = self.tokenizer.build_inputs_with_special_tokens(e['token_ids'][:tgt_len])
            tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e_trunc]

            self.mlm_probability = self.encoder_mlm_probability
            text_encoder_mlm_mask = self._whole_word_mask(tokens)

            self.mlm_probability = self.decoder_mlm_probability
            text_decoder_mlm_mask = self._whole_word_mask(tokens)

            input_ids_batch.append(torch.tensor(e_trunc))
            attention_mask_batch.append(torch.tensor([1] * len(e_trunc)))

            encoder_mlm_mask_batch.append(torch.tensor(text_encoder_mlm_mask))
            decoder_mlm_mask_batch.append(torch.tensor(text_decoder_mlm_mask))

        input_ids_batch = tensorize_batch(input_ids_batch, self.tokenizer.pad_token_id)
        attention_mask_batch = tensorize_batch(attention_mask_batch, 0)
        
        encoder_mlm_mask_batch = tensorize_batch(encoder_mlm_mask_batch, 0)
        decoder_mlm_mask_batch = tensorize_batch(decoder_mlm_mask_batch, 0)

        encoder_input_ids_batch, encoder_labels_batch = self.torch_mask_tokens(input_ids_batch.clone(), encoder_mlm_mask_batch)
        decoder_input_ids_batch, decoder_labels_batch = self.torch_mask_tokens(input_ids_batch.clone(), decoder_mlm_mask_batch)
        

        batch = {
            "encoder_input_ids": encoder_input_ids_batch,
            "encoder_attention_mask": attention_mask_batch,
            "encoder_labels": encoder_labels_batch,
            "decoder_input_ids": decoder_input_ids_batch,
            "decoder_attention_mask": attention_mask_batch,
            "decoder_labels": decoder_labels_batch,
        }

        return batch



@dataclass
class MAEOutput(ModelOutput):
    loss: Optional[Tensor] = None
    encoder_mlm_loss: Optional[float] = None
    decoder_mlm_loss: Optional[float] = None


class MAEForPretraining(nn.Module):
    def __init__(
            self,
            bert: BertForMaskedLM,
            model_args: ModelArguments,
    ):
        super(MAEForPretraining, self).__init__()
        self.lm = bert

        self.decoder_embeddings = self.lm.bert.embeddings
        self.c_head = TransformerEncoder(self.lm.config, model_args.n_head_layers)
        self.c_head.apply(self.lm._init_weights)

        self.cross_entropy = nn.CrossEntropyLoss()

        self.model_args = model_args

    def forward(self,
                encoder_input_ids, encoder_attention_mask, encoder_labels,
                decoder_input_ids, decoder_attention_mask, decoder_labels):
        # return (torch.sum(self.lm.bert.embeddings.position_ids[:, :decoder_input_ids.size(1)]), )
        lm_out: MaskedLMOutput = self.lm(
            encoder_input_ids, encoder_attention_mask,
            labels=encoder_labels,
            output_hidden_states=True,
            return_dict=True
        )
        cls_hiddens = lm_out.hidden_states[-1][:, :1]  # B 1 D

        decoder_embedding_output = self.decoder_embeddings(input_ids=decoder_input_ids)
        hiddens = torch.cat([cls_hiddens, decoder_embedding_output[:, 1:]], dim=1)
        
        hiddens = self.c_head(hiddens, decoder_attention_mask.type_as(hiddens)) # for compatibility
        
        pred_scores, loss = self.mlm_loss(hiddens, decoder_labels)

        total_loss = loss + lm_out.loss

        return MAEOutput(
            loss=total_loss,
            encoder_mlm_loss=lm_out.loss.item(),
            decoder_mlm_loss=loss.item(),
        )

    def mlm_loss(self, hiddens, labels):
        pred_scores = self.lm.cls(hiddens)
        masked_lm_loss = self.cross_entropy(
            pred_scores.view(-1, self.lm.config.vocab_size),
            labels.view(-1)
        )
        return pred_scores, masked_lm_loss

    def save_pretrained(self, output_dir: str):
        self.lm.save_pretrained(output_dir)
        torch.save(self.c_head.state_dict(), os.path.join(output_dir, "news_decoder.pt"))
    
    def load_pretrained(self, output_dir: str):
        self.lm = self.lm.from_pretrained(output_dir)
        decoder_state_dict = torch.load(os.path.join(output_dir, "news_decoder.pt"))
        self.c_head.load_state_dict(decoder_state_dict)

    @classmethod
    def from_pretrained(
        cls, model_args: ModelArguments,
        *args, **kwargs
    ):
        hf_model = AutoModelForMaskedLM.from_pretrained(*args, **kwargs)
        model = cls(hf_model, model_args)
        return model


class PreTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(PreTrainer, self).__init__(*args, **kwargs)
        self.enc_mlm_loss = AverageMeter('enc_mlm_loss', round_digits=3)
        self.dec_mlm_loss = AverageMeter('dec_mlm_loss', round_digits=3)
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
        self._reset_meters_if_needed()

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save_pretrained'):
            logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            state_dict = self.model.state_dict()
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        else:
            self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else: # here
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.model.training:
            self.enc_mlm_loss.update(outputs.encoder_mlm_loss)
            self.dec_mlm_loss.update(outputs.decoder_mlm_loss)

        return (loss, outputs) if return_outputs else loss

    def _reset_meters_if_needed(self):
        if int(self.state.epoch) != self.last_epoch:
            self.last_epoch = int(self.state.epoch)
            self.enc_mlm_loss.reset()
            self.dec_mlm_loss.reset()


class TrainerCallbackForSaving(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        control.should_save = True


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    training_args.remove_unused_columns = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
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

    set_seed(training_args.seed)

    if model_args.model_name_or_path:
        model = MAEForPretraining.from_pretrained(model_args, model_args.model_name_or_path)
        logger.info(f"------Load model from {model_args.model_name_or_path}------")
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    elif model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name)
        bert = BertForMaskedLM(config)
        model = MAEForPretraining(bert, model_args)
        logger.info("------Init the model------")
        tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_name)
    else:
        raise ValueError("You must provide the model_name_or_path or config_name")

    dataset = load_dataset("json", data_files=data_args.train_path, split="train")
    data_collator = MAECollator(
        tokenizer,
        encoder_mlm_probability=data_args.encoder_mlm_probability,
        decoder_mlm_probability=data_args.decoder_mlm_probability,
        max_seq_length=data_args.max_seq_length
    )

    # Initialize our Trainer
    trainer = PreTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    trainer.add_callback(TrainerCallbackForSaving())

    # Training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()


if __name__ == "__main__":
    main()
