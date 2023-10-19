import random
import numpy as np
import pickle
from dataclasses import dataclass
from collections import defaultdict

from typing import List, Dict, Any, Tuple

import torch
from transformers import DataCollatorWithPadding

from utils import get_time_stamp

def pad_or_truncate(history_news_ids: List[str], max_history_len: int = 10):
    # left padding
    left_padded_history_news_ids = ["PAD"] * max_history_len + history_news_ids
    left_padded_history_mask = [0] * max_history_len + [1] * len(history_news_ids)
    return left_padded_history_news_ids[-max_history_len:], left_padded_history_mask[-max_history_len:]

def convert_news_to_text(news: Dict[str, str], fields: List[str] = ['category', 'subcategory', 'title']) -> str:
    infos = []
    for attribute in fields:
        if attribute in news:
            infos.append(news[attribute])
    return " | ".join(infos)


class MINDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        tokenizer,
    ):
        self.tokenizer = tokenizer

        self.args = args

        self.news: Dict[str, Dict[str, Any]] = self._load_news(args.train_news_file)

        self.tokenized_news = {}
        for nid, news in self.news.items():
            self.tokenized_news[nid] = self.tokenizer(
                convert_news_to_text(news, fields=args.news_fields),
                truncation='only_first',
                max_length=self.args.news_max_len,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False
            )
        
        self.behaviors: List[Dict[str, Any]] = self._load_behaviors(args.train_behavior_file)

    def _load_news(self, news_file):
        """init news information from news file

        Args:
            news_file (str): path of news file
        Returns:
            news (Dict[str, Dict[str, str]]): news data
        """
        news = { "PAD": {} }

        with open(news_file, "r") as f:
            for line in f:
                nid, category, subcategory, title, abstract, url, _, _ = line.strip("\n").split("\t")
                if nid in news:
                    continue
                news[nid] = {
                    "nid": nid,
                    "title": title,
                    "category": category,
                    "subcategory": subcategory,
                    "abstract": abstract,
                }
        
        return news

    def _load_behaviors(self, behaviors_file):
        """init behavior logs given behaviors file.

        Args:
            behaviors_file(str): path of behaviors file
        """
        behaviors: List[Dict[str, Any]] = []

        self._user_behavior: Dict[str, Dict[float, Dict[str, Any]]] = {}
        # record user history

        with open(behaviors_file, "r") as f:
            for line in f:
                idx, uid, t, history, impr = line.strip("\n").split("\t")

                history_news_ids = history.split() # List[str], historical news ids

                # filter out empty behaviors, avoid loss nan
                if self.args.filter_null_behavior and len(history_news_ids) < 1:
                    continue
                
                candidate_news_ids = [] # List[str], candidate news ids
                labels = [] # List[int], candidate news labels
                pos_news_ids = []
                neg_news_ids = []
                for i in impr.split():
                    item = i.split("-")
                    candidate_news_ids.append(item[0])
                    label = int(item[1]) if len(item) == 2 else 0
                    labels.append(label)
                    if label == 1:
                        pos_news_ids.append(item[0])
                    else:
                        neg_news_ids.append(item[0])
                
                t = get_time_stamp(t)
                behavior = {
                    "uid": uid,
                    "time": t,
                    "history": history_news_ids,
                    "candidates": candidate_news_ids,
                    "labels": labels,
                    "pos": pos_news_ids,
                    "neg": neg_news_ids,
                }
                
                # record user history for imputation
                if len(history_news_ids) > 0:
                    if uid in self._user_behavior:
                        self._user_behavior[uid][t] = behavior
                    else:
                        self._user_behavior[uid] = {t: behavior}

                if len(pos_news_ids) >= 0:
                    behaviors.append(behavior)
        return behaviors
    
    def impute_history(self, uid: str, t: float):
        # print("Imputing history...")
        # print(len(self._user_behavior))
        if uid not in self._user_behavior:
            return ["PAD"]
        user_histories = self._user_behavior[uid] # Dict[float, Dict], timestamp to behavior
        # find the nearest behavior
        distances = {abs(t - past_t): past_t for past_t in user_histories}
        nearest_time = distances[min(distances.keys())]
        nearest_behavior = user_histories[nearest_time] # Dict[str, Any]
        # print("sucessfully imputed!")
        return nearest_behavior['history']
    
    def __len__(self):
        return len(self.behaviors)
    
    def __getitem__(self, idx):
        behavior = self.behaviors[idx]
        history_news_ids = behavior["history"] # List[str]
        if len(history_news_ids) < 1:
            history_news_ids = self.impute_history(behavior["uid"], behavior["time"])
        history, history_mask = pad_or_truncate(history_news_ids, max_history_len=self.args.max_history_len)
        history_news = [ self.tokenized_news[i] for i in history ]

        pos_news_id = behavior["pos"][0] # always use first positive 
        pos_news = self.tokenized_news[pos_news_id]

        neg_news_ids = behavior["neg"] # List[str]
        # sampling procedure
        if len(neg_news_ids) < self.args.npratio:
            negs = random.sample(self.news.keys(), k=self.args.npratio - len(neg_news_ids))
            negs.extend(neg_news_ids)
        else:
            negs = random.sample(neg_news_ids, k=self.args.npratio)
        
        neg_news = [self.tokenized_news[i] for i in negs]

        candidate_news = [pos_news] + neg_news
        # first positive, rest negatives

        data = {
            "history_news": history_news, # List[{"input_ids": [...]}]
            "candidate_news": candidate_news,
            "history_mask": history_mask,
            "pos_news": pos_news,
            "neg_news": neg_news,
        }

        return data


@dataclass
class MINDCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    news_max_len: int = 32

    def __call__(self, features):
        query = [f["history_news"] for f in features] # List[List[Encoding]] shape (batch_size, num_history)
        doc = [f["candidate_news"] for f in features] # List[List[Encoding]] shape (batch_size, npratio+1)
        qmask = [f["history_mask"] for f in features] # List[List[int]] shape (batch_size, num_history)


        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(doc[0], list):
            doc = sum(doc, [])

        q_collated = self.tokenizer.pad(
            query,
            padding='max_length',
            max_length=self.news_max_len,
            return_tensors="pt",
        ) # (batch_size * num_history, max_seq_len)
        d_collated = self.tokenizer.pad(
            doc,
            padding='max_length',
            max_length=self.news_max_len,
            return_tensors="pt",
        ) # (batch_size * (npratio+1), max_seq_len)

        qmask = torch.tensor(qmask, dtype=torch.float)

        return {"history": q_collated, "candidate": d_collated, "history_mask": qmask}


class MINDEvaluationDataset(MINDDataset):
    def __init__(
        self,
        args,
        tokenizer,
    ):
        self.tokenizer = tokenizer

        self.args = args

        self.news: Dict[str, Dict[str, Any]] = self._load_news(args.eval_news_file)

        self.tokenized_news = {}
        for nid, news in self.news.items():
            self.tokenized_news[nid] = self.tokenizer(
                convert_news_to_text(news, fields=args.news_fields),
                truncation='only_first',
                max_length=self.args.news_max_len,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False
            )
        
        self.behaviors: List[Dict[str, Any]] = self._load_behaviors(args.eval_behavior_file)

