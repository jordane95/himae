import os
import math
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

from transformers.models.bert.modeling_bert import BertLayer
from transformers.file_utils import ModelOutput

from utils import get_extended_attention_mask


logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None

class AttentionPooling(nn.Module):

    def __init__(self, emb_size, hidden_size):
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(emb_size, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: batch_size, seq_len, emb_dim
            attn_mask: batch_size, seq_len
        Returns:
            (shape) batch_size, emb_dim
        """
        bz = x.shape[0]
        e = self.att_fc1(x) # (batch_size, seq_len, hidden_size)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e) # (batch_size, seq_len, 1)
        alpha = torch.exp(alpha)

        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)

        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha).squeeze(dim=-1)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, position_embedding_type=None):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.attention_probs_dropout_prob = 0.1
        self.dropout = nn.Dropout(self.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = self.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * self.max_position_embeddings - 1, self.attention_head_size)


    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            input_shape = hidden_states.size()[:-1] # (batch_size, seq_len)
            attention_mask = get_extended_attention_mask(attention_mask, input_shape)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = context_layer

        return outputs


class TransformerEncoder(nn.Module):
    def __init__(self, config, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([BertLayer(config=config) for _ in range(n_layers)])

    def forward(
        self,
        inputs: Tensor, # (batch_size, seq_len, hidden_size)
        input_mask: Tensor, # (batch_size, seq_len)
    ) -> Tensor: # (batch_size, seq_len, hidden_size)
        attention_mask = get_extended_attention_mask(
            input_mask,
            input_mask.shape,
            input_mask.device,
        )
        hiddens = inputs
        for layer in self.layers:
            hiddens = layer(hiddens, attention_mask)[0]
        outputs = hiddens
        return outputs


class NewsEncoder(nn.Module):
    """Wrapper of BERT-based news encoder"""
    def __init__(self, args, lm):
        super(NewsEncoder, self).__init__()
        self.args = args
        self.pooling = args.news_pooling
        self.add_pooler = args.add_news_pooler
        self.normalize = args.normalize_news
        self.lm = lm
        if self.pooling == 'att':
            self.attn = AttentionPooling(self.lm.config.hidden_size, args.news_query_vector_dim)
        self.pooler = nn.Linear(self.lm.config.hidden_size, args.news_dim or self.lm.config.hidden_size) if self.add_pooler else nn.Identity()

    def forward(self, news: Dict[str, Tensor]):
        """Batch of news in, news embeddings out
        Args:
            news (Dict[str, Tensor]): tensor shape (batch_size, max_seq_len)
        Returns:
            news_embeddings (Tensor): shape (batch_size, news_emb_dim)
        """
        outputs = self.lm(**news, output_hidden_states=True, return_dict=True) # compatible with MLM pretraiing
        word_vecs = outputs.hidden_states[-1] # (batch_size, max_seq_len, hidden_size)
        news_embeddings = self.pool_news_embedding(word_vecs, news['attention_mask'])
        return news_embeddings

    def pool_news_embedding(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        word_vecs = last_hidden_states
        if self.pooling == 'cls':
            news_vec = word_vecs[:, 0] # (batch_size, hidden_size)
        elif self.pooling == 'mean':
            news_vec = self.mean_pooling(word_vecs, attention_mask)
        elif self.pooling == 'att':
            news_vec = self.attn(word_vecs, attention_mask)
        news_embeddings = self.pooler(news_vec)
        if self.normalize:
            news_embeddings = F.normalize(news_embeddings, p=2, dim=-1)
        return news_embeddings
    
    def mean_pooling(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def save_pretrained(self, output_dir):
        if self.add_pooler:
            torch.save(self.pooler.state_dict(), os.path.join(output_dir, 'news_pooler.pt'))
    
    def load_pretrained(self, output_dir):
        if self.add_pooler:
            pooler_state_dict = torch.load(os.path.join(output_dir, 'news_pooler.pt'))
            self.pooler.load_state_dict(pooler_state_dict)


class UserEncoder(nn.Module):
    """Get user embeddings from sequence of news embeddings"""
    def __init__(self, args, model = None):
        super(UserEncoder, self).__init__()
        self.args = args
        self.pooling = args.user_pooling
        self.add_pooler = args.add_user_pooler
        self.normalize = args.normalize_user

        self.model = model
        
        if self.pooling == 'att':
            self.attn = AttentionPooling(args.news_dim, args.user_query_vector_dim)
        
        self.pooler = nn.Linear(args.news_dim, args.user_dim) if self.add_pooler else nn.Identity()

    def forward(self, news_vecs, log_mask):
        """
        Args:
            news_embeddings (Tensor): (batch_size, user_log_length, news_embedding_dim)
            log_mask (Tensor): (batch_size, user_log_length)
        Returns:
            user_embeddings (Tensor): (shape) (batch_size, user_dim)
        """
        bz = news_vecs.shape[0]

        # self.model: Tensor in, Tensor out, shape not change
        if self.model is not None:
            news_vecs = self.model(news_vecs, log_mask) # user model
        
        # pooling: reduce the seq_len dimension to 1
        if self.pooling == 'mean':
            user_vec = self.mean_pooling(news_vecs, log_mask)
        elif self.pooling == 'weightedmean':
            user_vec = self.weighted_mean_pooling(news_vecs, log_mask)
        elif self.pooling == 'att':
            user_vec = self.attn(news_vecs, log_mask)
        
        # optinal linear projection
        user_embeddings = self.pooler(user_vec)

        # optinal l2 normalization
        if self.normalize:
            user_embeddings = F.normalize(user_embeddings, p=2, dim=-1)
        return user_embeddings
    
    def mean_pooling(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / torch.clamp(attention_mask.sum(dim=1), min=1e-9)[..., None]
    
    def weighted_mean_pooling(self, token_embeddings: Tensor, attention_mask: Tensor):
        sequence_lengths = attention_mask.sum(dim=1) # (batch_size,)
        max_seq_len = attention_mask.shape[1]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # token_embeddings shape: bs, seq, hidden_dim
        # weights = (
        #         torch.arange(start=1, end=token_embeddings.shape[1] + 1)
        #         .unsqueeze(0)
        #         .unsqueeze(-1)
        #         .expand(token_embeddings.size())
        #         .float().to(token_embeddings.device)
        # )
        weights = (
            torch.stack([
                torch.arange(start=seq_len+1-max_seq_len, end=seq_len+1) for seq_len in sequence_lengths
            ])
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .type_as(input_mask_expanded)
            .to(token_embeddings.device)
        )
        # import pdb; pdb.set_trace();
        assert weights.shape == token_embeddings.shape == input_mask_expanded.shape
        input_mask_expanded = input_mask_expanded * weights
        
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

        sum_mask = input_mask_expanded.sum(1)

        sum_mask = torch.clamp(sum_mask, min=1e-9)
        sequence_embeddings = sum_embeddings / sum_mask
        return sequence_embeddings

    def save_pretrained(self, output_dir):
        if self.add_pooler:
            torch.save(self.pooler.state_dict(), os.path.join(output_dir, 'user_pooler.pt'))

    def load_pretrained(self, output_dir):
        if self.add_pooler:
            pooler_state_dict = torch.load(os.path.join(output_dir, 'user_pooler.pt'))
            self.pooler.load_state_dict(pooler_state_dict)


class DualEncoder(torch.nn.Module):

    def __init__(self, args):
        super(DualEncoder, self).__init__()
        self.args = args

        self.lm = AutoModel.from_pretrained(args.model_name_or_path)

        if self.args.user_model == 'NRMS':
            self.um = MultiHeadSelfAttention(args.news_dim, args.num_attention_heads)
        elif self.args.user_model == 'transformer':
            self.um = TransformerEncoder(self.lm.config, args.n_user_layers)
        else:
            self.um = None
        
        self.news_encoder = NewsEncoder(args, self.lm)
        self.user_encoder = UserEncoder(args, self.um)

        self.cross_entropy = nn.CrossEntropyLoss()
    
    def gradient_checkpointing_enable(self):
        self.lm.gradient_checkpointing_enable()

    def forward(self, history, candidate, history_mask):
        """
        Args:
            history (Dict[str, Tensor]): (batch_size * num_history, max_seq_len)
            history_mask (Tensor): (batch_size, num_history)
            candidate (Dict[str, Tensor]): (batch_size * (npratio+1), max_seq_len)
        Returns:
            loss: (shape) ()
            score: (shape) (batch_size)
        """
        batch_size, num_history = history_mask.shape
        num_candidates = candidate['input_ids'].shape[0] // batch_size

        history_news_embeddings = self.news_encoder(history) # (batch_size * num_history, news_dim)
        candidate_news_embeddings = self.news_encoder(candidate) # (batch_size * (npratio+1), news_dim)

        history_news_embeddings = history_news_embeddings.reshape(batch_size, num_history, -1)

        user_embeddings = self.user_encoder(history_news_embeddings, history_mask) # (batch_size, user_dim)

        # in batch negatives
        scores = user_embeddings @ candidate_news_embeddings.t() # (batch_size, batch_size * (npratio+1))
        # print("temperature", self.args.temperature)
        scores /= self.args.temperature

        labels = torch.arange(batch_size).to(scores.device) * num_candidates # (batch_size)
        
        # import pdb; pdb.set_trace();
        # ordinary dot product
        # score = torch.bmm(candidate_news_embeddings, user_embeddings.unsqueeze(dim=-1)).squeeze(dim=-1)
        # (batch_size, npratio+1, news_dim) * (batch_size, news_dim, 1) -> (batch_size, npratio+1, 1) -> (batch_size, npratio+1)
        # from pprint import pprint
        loss = self.cross_entropy(scores, labels)
        return EncoderOutput(
            loss=loss,
            scores=scores,
        )

    def save_pretrained(self, output_path):
        self.lm.save_pretrained(output_path)
        if self.um is not None:
            torch.save(self.um.state_dict(), os.path.join(output_path, 'user_model.pt'))
        self.news_encoder.save_pretrained(output_path)
        self.user_encoder.save_pretrained(output_path)
    
    def load_pretrained(self, output_path):
        self.lm = self.lm.from_pretrained(output_path)
        self.news_encoder.lm = self.lm
        if self.um is not None:
            user_model_state_dict = torch.load(os.path.join(output_dir, 'user_model.pt'))
            self.um.load_state_dict(user_model_state_dict)
            self.user_encoder.model = um
        self.news_encoder.load_pretrained(output_path)
        self.user_encoder.load_pretrained(output_path)
