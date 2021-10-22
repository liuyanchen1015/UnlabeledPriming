from dataclasses import dataclass
from typing import Optional


@dataclass
class InputExample:
    text_a: str
    text_b: Optional[str] = ""
    label: Optional[str] = None
