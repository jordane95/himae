from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Args:
    news_fields: Optional[List[str]] = field(default_factory=list)
    steps: int = field(default=100)
    def __post_init__(self):
        if len(self.news_fields) == 0:
            self.news_fields = ['category', 'subcategory', 'title']


from transformers import HfArgumentParser



parser = HfArgumentParser((Args,))

args, = parser.parse_args_into_dataclasses()

print(args)