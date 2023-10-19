
# mkdir -p data
# wget -P data https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/npr.jsonl.gz
# wget -P data https://huggingface.co/datasets/sentence-transformers/embedding-training-data/resolve/main/agnews.jsonl.gz

mkdir -p pretrain_data

python preprocess.py


