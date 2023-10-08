import multiprocessing
from multiprocessing import cpu_count
from typing import List, NamedTuple

import torch
import re

from torchaudio.models.decoder._ctc_decoder import download_pretrained_files

from .char_text_encoder import CharTextEncoder
from pyctcdecode import build_ctcdecoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: list[str] = None, vocab_path=None):
        super().__init__(alphabet)
        if vocab_path is not None:
            lm_files = download_pretrained_files('librispeech-3-gram')
            unigrams = [i.split('\t')[0].lower().strip() for i in open(lm_files.lexicon).read().splitlines()]
            self.decoder = build_ctcdecoder(
                labels=[''] + list(self.alphabet),
                kenlm_model_path=lm_files.lm,
                unigrams=unigrams,
                alpha=0.6,
                beta=0.15,
            )
        vocab = [''] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: list[int]) -> str:
        text = ''.join([self.ind2char[ind] for ind in inds])
        text_with_tokens = re.sub(fr'[^{re.escape(self.EMPTY_TOK)}]+',
                                  lambda x: re.sub(r'(.)\1+', r'\1', x.group(0)),
                                  text)
        return text_with_tokens.replace(self.EMPTY_TOK, "")

    def lm_ctc_beam_search(self, probs, probs_length, beam_size=10):
        text = self.decoder.decode(probs[:probs_length], beam_width=beam_size)
        return [Hypothesis(text, 1.0)]

    def ctc_beam_search(self,
                        probs: torch.tensor,
                        probs_length,
                        beam_size: int = 50) -> list[Hypothesis]:
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
