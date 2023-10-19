from typing import Dict, List, Any
import json

from transformers import AutoTokenizer

from datasets import Dataset, load_dataset


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_mind(data_name_or_path: str = "mind", save_path: str = "mind.json"):
    news: Dict[str, List[Any]] = {
        "nid": [],
        "title": [],
        "category": [],
        "subcategory": [],
        "abstract": [],
    }
    with open(data_name_or_path, "r") as f:
        for line in f:
            nid, category, subcategory, title, abstract, url, _, _ = line.strip("\n").split("\t")
            news["nid"].append(nid)
            news["title"].append(title)
            news["category"].append(category)
            news["subcategory"].append(subcategory)
            news["abstract"].append(abstract)
    
    mind_news = Dataset.from_dict(news)

    def mind_news_tokenize_function(examples):
        examples["text"] = [title + " " + abstract for title, abstract in zip(examples["title"], examples["abstract"])]
        examples["token_ids"] = tokenizer(examples["text"], add_special_tokens=False, truncation=False, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        return examples

    mind = mind_news.map(mind_news_tokenize_function, batched=True, batch_size=1000, remove_columns=mind_news.column_names)
    
    mind.to_json(save_path)


def preprocess_cc_news(data_name_or_path: str = "cc_news", save_path: str = "cc_news.json"):
    cc_news = load_dataset(data_name_or_path, split="train")
    cc_news = cc_news.rename_column("text", "article")

    def cc_news_tokenize_function(examples):
        examples["text"] = [title + " " + desc for title, desc in zip(examples["title"], examples["description"])]
        examples["token_ids"] = tokenizer(examples["text"], add_special_tokens=False, truncation=False, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        return examples

    cc_news = cc_news.map(cc_news_tokenize_function, batched=True, batch_size=1000, remove_columns=cc_news.column_names)

    cc_news.to_json(save_path)


def preprocess_cnn_dailymail(data_name_or_path: str = "cnn_dailymail", save_path: str = "cnn_dailymail.json"):
    cnn_daily_mail = load_dataset(data_name_or_path, '1.0.0', split="train")

    def cnn_daily_mail_tokenize_function(examples):
        examples["text"] = examples["highlights"]
        examples["token_ids"] = tokenizer(examples["text"], add_special_tokens=False, truncation=False, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        return examples

    cnn_daily_mail = cnn_daily_mail.map(cnn_daily_mail_tokenize_function, batched=True, batch_size=1000, remove_columns=cnn_daily_mail.column_names)

    cnn_daily_mail.to_json(save_path)


def preprocess_npr_or_agnews(data_name_or_path: str = "data/npr.jsonl.gz", save_path: str = "npr.json"):
    npr = load_dataset('text', data_files=data_name_or_path, split="train")
    npr = npr.rename_column("text", "texts")
    
    def npr_tokenize_function(examples):
        examples["text"] = list(map(lambda x: json.loads(x), examples["texts"]))
        examples["text"] = [" ".join(text_list) for text_list in examples["text"]]
        examples["token_ids"] = tokenizer(examples["text"], add_special_tokens=False, truncation=False, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
        return examples

    npr = npr.map(npr_tokenize_function, batched=True, batch_size=1000, remove_columns=npr.column_names)

    npr.to_json(save_path)


if __name__ == "__main__":
    # preprocess_mind(data_name_or_path="data/mind/news.tsv", save_path="pretrain_data/mind.json")
    preprocess_cc_news(data_name_or_path="cc_news", save_path="pretrain_data/cc_news.json")
    # preprocess_cnn_dailymail(data_name_or_path="cnn_dailymail", save_path="pretrain_data/cnn_dailymail.json")
    # preprocess_npr_or_agnews(data_name_or_path="data/npr.jsonl.gz", save_path="pretrain_data/npr.json")
    # preprocess_npr_or_agnews(data_name_or_path="data/agnews.jsonl.gz", save_path="pretrain_data/agnews.json")
