import os
from dataclasses import dataclass, field
from typing import Optional, Union, List

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default='bert-base-uncased', metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # news encoder
    news_pooling: str = field(default='cls')
    news_query_vector_dim: int = field(default=None)
    news_dim: int = field(default=768)
    add_news_pooler: bool = field(default=False)
    normalize_news: bool = field(default=False)
    n_head_layers: int = field(default=2)
    
    # user encoder
    user_model: str = field(default='transformer') # "NRMS" / "transformer" / ...
    n_user_layers: int = field(default=2)
    num_attention_heads: int = field(default=12) # specific for NRMS model
    user_pooling: str = field(default='mean')
    user_query_vector_dim: int = field(default=768)
    user_dim: int = field(default=768) # normally should eqal to news_dim
    add_user_pooler: bool = field(default=False)
    normalize_user: bool = field(default=False)


@dataclass
class DataArguments:
    sample_neg_from_topk: int = field(
        default=200, metadata={"help": "sample negatives from top-k"}
    )
    teacher_score_files: str = field(
        default=None, metadata={"help": "Path to score_file for distillation"}
    )

    prediction_save_path: Union[str] = field(
        default=None, metadata={"help": "Path to save prediction"}
    )

    news_fields: List[str] = field(default_factory=list)

    news_max_len: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    train_data_dir: str = field(
        default=None, metadata={"help": "Path to training data"}
    )

    train_news_file: str = field(
        default=None, metadata={"help": "Path to training news corpus."}
    )

    train_behavior_file: str = field(
        default=None, metadata={"help": "Path to training behavior file."}
    )

    npratio: int = field(
        default=4, metadata={"help": "negative to positive ratio"}
    )

    max_history_len: int = field(
        default=10, metadata={"help": "maximum number of history news for user modeling"}
    )

    eval_data_dir: str = field(
        default=None, metadata={"help": "Path to training data"}
    )

    eval_news_file: str = field(
        default=None, metadata={"help": "Path to training news corpus."}
    )

    eval_behavior_file: str = field(
        default=None, metadata={"help": "Path to training behavior file."}
    )

    filter_null_behavior: bool = field(default=False)

    def __post_init__(self):

        if self.train_data_dir and not self.train_news_file:
            if not os.path.exists(os.path.join(self.train_data_dir, 'news.tsv')):
                raise FileNotFoundError(f'There is no mews.tsv in {self.train_data_dir}')
            self.train_news_file = os.path.join(self.train_data_dir, 'news.tsv')
        
        if self.train_data_dir and not self.train_behavior_file:
            if not os.path.exists(os.path.join(self.train_data_dir, 'behaviors.tsv')):
                raise FileNotFoundError(f'There is no behaviors.tsv in {self.train_data_dir}')
            self.train_behavior_file = os.path.join(self.train_data_dir, 'behaviors.tsv')
        
        if self.eval_data_dir and not self.eval_news_file:
            if not os.path.exists(os.path.join(self.eval_data_dir, 'news.tsv')):
                raise FileNotFoundError(f'There is no mews.tsv in {self.eval_data_dir}')
            self.eval_news_file = os.path.join(self.eval_data_dir, 'news.tsv')
        
        if self.eval_data_dir and not self.eval_behavior_file:
            if not os.path.exists(os.path.join(self.eval_data_dir, 'behaviors.tsv')):
                raise FileNotFoundError(f'There is no behaviors.tsv in {self.eval_data_dir}')
            self.eval_behavior_file = os.path.join(self.eval_data_dir, 'behaviors.tsv')

        if len(self.news_fields) == 0: # keep basic attributes
            self.news_fields = ['category', 'subcategory', 'title']

@dataclass
class MINDTrainingArguments(TrainingArguments):
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    temperature: Optional[float] = field(default=1.0)
    contrastive_loss_weight: Optional[float] = field(default=0.0)

