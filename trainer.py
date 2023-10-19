import os
from tqdm import tqdm
from itertools import repeat
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
from transformers.trainer import logger, Trainer
from transformers import DataCollatorWithPadding

import torch
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist

from utils import cal_metric

TRAINING_ARGS_NAME = "training_args.bin"


class MINDTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(MINDTrainer, self).__init__(*args, **kwargs)
        # self._dist_loss_scale_factor = dist.get_world_size() if self.args.negatives_x_device else 1
        self._dist_loss_scale_factor = 1

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        self.model.save_pretrained(output_dir)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=True,
            num_workers=self.args.dataloader_num_workers,
        )

    def training_step(self, *args):
        return super(MINDTrainer, self).training_step(*args) / self._dist_loss_scale_factor

    @torch.no_grad()
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        metrics: List[str] = ['group_auc', 'mean_mrr', 'ndcg@5;10'],
    ) -> Dict[str, float]:
        eval_dataset = eval_dataset or self.eval_dataset
        model = self.model
        batch_size = self.args.per_device_eval_batch_size
        eval_data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            max_length=self.data_collator.news_max_len,
            padding='max_length'
        )
        model.eval()
        news_ids = []
        news_embeddings = []
        news_batch = []
        logger.info("Runing evaluation")
        logger.info(f"Encoding news...")
        for news_id, news_tokens in tqdm(eval_dataset.tokenized_news.items()):
            news_ids.append(news_id)
            news_batch.append(news_tokens)
            if len(news_batch) == batch_size:
                news_inputs = eval_data_collator(news_batch)
                news_inputs = self._prepare_inputs(news_inputs) # move to model device
                news_vecs = model.news_encoder(news_inputs)
                news_embeddings.extend(news_vecs)
                news_batch = []
        news_inputs = eval_data_collator(news_batch)
        news_inputs = self._prepare_inputs(news_inputs) 
        news_vecs = model.news_encoder(news_inputs)
        news_embeddings.extend(news_vecs)
        assert len(news_ids) == len(news_embeddings), f"News embeddings length and news ids don't match"

        news_id_to_embeddings = dict(zip(news_ids, news_embeddings))

        all_preds = []
        all_labels = []
        logger.info("Encoding users...")
        for i in tqdm(range(len(eval_dataset))):
            behavior = eval_dataset.behaviors[i] # Dict[str, Any]
            history_news_ids = behavior["history"] # List[str]
            if len(history_news_ids) < 1:
                history_news_ids = eval_dataset.impute_history(behavior["uid"], behavior["time"])
            candidate_news_ids = behavior["candidates"]
            labels = behavior["labels"]
            history_news_embeddings = torch.stack([news_id_to_embeddings[news_id] for news_id in history_news_ids]).unsqueeze(0)
            # (1, num_history, news_embedding_dim)
            history_mask = torch.ones(len(history_news_embeddings)).unsqueeze(0) # (1, num_history)
            history_news_embeddings, history_mask = self._prepare_inputs([history_news_embeddings, history_mask])
            user_embeddings = model.user_encoder(history_news_embeddings, history_mask) # (1, user_dim)

            candidate_news_embeddings = torch.stack([news_id_to_embeddings[news_id] for news_id in candidate_news_ids]).unsqueeze(0)
            # (1, num_candidates, news_embedding_dim)
            candidate_scores = user_embeddings @ candidate_news_embeddings.squeeze(0).T # (1, num_candidates)
            all_labels.append(np.array(labels))
            all_preds.append(candidate_scores.squeeze(0).cpu().numpy())
        # import pdb; pdb.set_trace();
        res = cal_metric(all_labels, all_preds, metrics)
        
        self.log(res)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, res)
        self._memory_tracker.stop_and_update_metrics(res)

        model.train()
        return res
