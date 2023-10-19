

from model import TransformerEncoder
import torch
import os

filepath = 'ckpt/news-mae-base-100k1024bs3e-4-ds'


for ckpt in os.listdir(filepath) + [""]:
    ckpt_path = os.path.join(filepath, ckpt)
    if not os.path.isdir(ckpt_path):
        continue
    print(ckpt_path)
    decoder_path = os.path.join(ckpt_path, "decoder.pt")
    news_decoder_path = os.path.join(ckpt_path, "news_decoder.pt")
    decoder = torch.load(decoder_path)
    torch.save(decoder.state_dict(), news_decoder_path)

