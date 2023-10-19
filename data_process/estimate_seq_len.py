

import sys
from datasets import load_dataset


filepath = sys.argv[1]

news = load_dataset('json', data_files=filepath, split='train')

token_lens = []
for data in news:
    num_tokens = len(data["token_ids"])
    token_lens.append(num_tokens)

print(sum(token_lens) / len(token_lens))

