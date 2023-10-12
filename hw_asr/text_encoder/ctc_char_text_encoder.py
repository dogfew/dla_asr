from collections import defaultdict
from typing import List, NamedTuple
import numpy as np
import torch
import re

from torchaudio.models.decoder._ctc_decoder import download_pretrained_files

from .char_text_encoder import CharTextEncoder
from hw_asr.utils.download_lm import get_language_model
from pyctcdecode import build_ctcdecoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: list[str] = None, vocab_path=1):
        super().__init__(alphabet)
        if vocab_path is not None:
            # model_path, unigrams = get_language_model()
            # unigrams = open(unigrams).read().splitlines()
            lm_files = download_pretrained_files("librispeech-3-gram")
            unigrams = [
                i.split("\t")[0].lower().strip()
                for i in open(lm_files.lexicon).read().splitlines()
            ]
            model_path = lm_files.lm
            self.decoder = build_ctcdecoder(
                labels=[""] + list(self.alphabet),
                kenlm_model_path=model_path,
                unigrams=unigrams,
                alpha=0.6,
                beta=0.15,
            )
            # vocab = [''] + list(self.alphabet)
        self.ind2char = dict(enumerate(["^"] + list(self.alphabet)))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: list[int]) -> str:
        text = "".join([self.ind2char[ind] for ind in inds])
        text_with_tokens = re.sub(
            rf"[^{re.escape(self.EMPTY_TOK)}]+",
            lambda x: re.sub(r"(.)\1+", r"\1", x.group(0)),
            text,
        )
        return text_with_tokens.replace(self.EMPTY_TOK, "")

    def lm_ctc_beam_search(self, probs, probs_length, beam_size=25):
        text = self.decoder.decode(probs[:probs_length], beam_width=beam_size)
        return [Hypothesis(text, 1.0)]

    def ctc_beam_search(
        self, probs: torch.Tensor | np.ndarray, probs_length, beam_size: int = 5
    ) -> list[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        state = {("", self.EMPTY_TOK): 1}
        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()
        for frame in probs[:probs_length]:
            state = self.__extend_and_merge(frame, state)
            state = self.__truncate(state, beam_size)
        return sorted(
            [Hypothesis(k[0], v) for k, v in state.items()],
            key=lambda x: x.prob,
            reverse=True,
        )

    def __extend_and_merge(self, frame, state):
        new_state = defaultdict(float)
        for next_char_index, next_char_proba in enumerate(frame):
            for (pref, last_char), pref_proba in state.items():
                next_char = self.ind2char[next_char_index]
                if next_char == last_char:
                    new_prefix = pref
                else:
                    if next_char != self.EMPTY_TOK:
                        new_prefix = pref + next_char
                    else:
                        new_prefix = pref
                    last_char = next_char
                new_state[(new_prefix, last_char)] += pref_proba * next_char_proba
        return new_state

    def __truncate(self, state, beam_size):
        state_list = sorted(state.items(), key=lambda x: -x[1])
        return dict(state_list[:beam_size])
