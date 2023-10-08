from typing import List, NamedTuple

import torch
import re
from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: list[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: list[int]) -> str:
        text = ''.join([self.ind2char[ind] for ind in inds])
        text_with_tokens = re.sub(fr'[^{re.escape(self.EMPTY_TOK)}]+',
                                  lambda x: re.sub(r'(.)\1+', r'\1', x.group(0)),
                                  text)
        return text_with_tokens.replace(self.EMPTY_TOK, "")

    def ctc_beam_search(self,
                        probs: torch.tensor,
                        probs_length,
                        beam_size: int = 100) -> list[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: list[Hypothesis] = []
        # TODO: your code here
        raise NotImplementedError
        return sorted(hypos, key=lambda x: x.prob, reverse=True)
