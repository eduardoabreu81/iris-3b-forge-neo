# Iris 3B support for Forge Neo
# The Iris text encoder: Qwen3-VL 4B raw hidden states of layers [2, 5, ... 35] under the Iris chat template.
#
# The whole exchange (system instruction, user text, assistant suffix) is encoded, padded to system + 300 tokens,
# and only the last 300 states go to the DiT (upstream's text_len), pads zeroed. Each prompt's conditioning is the
# raw (300, layers, dim) states with no batch dimension, as Forge's native engines return theirs (Forge stacks the
# batch); the DiT runs its layerwise adapter inside forward, on the GPU, as Forge's Krea 2 does with its text fusion.
# The zeroed pads carry the mask: the DiT reads it back from them.

import torch
from backend.args import dynamic_args
from backend.text_processing import emphasis
from backend.text_processing._comfy import INF, SDClipModel, SDTokenizer

PAD = 151643
HIDDEN_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35]
PREFIX = ("<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
          "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
TEXT_LENGTH = 300


class IrisTextEngine:
    def __init__(self, text_encoder, tokenizer):
        self.text_encoder = SDClipModel(text_encoder, layer=HIDDEN_LAYERS, layer_idx=None,
                                        special_tokens={"pad": PAD}, layer_norm_hidden_state=False,
                                        enable_attention_masks=True, return_attention_masks=True)
        self.tokenizer = SDTokenizer(tokenizer, pad_with_end=False, has_start_token=False, has_end_token=False,
                                     pad_to_max_length=False, max_length=INF, min_length=1, pad_token=PAD)
        self.prefix_ids = self.tokenizer.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.tokenizer.encode(SUFFIX, add_special_tokens=False)

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.EmphasisNone()

    def caption_ids(self, text: str) -> list[int]:
        # surrounding whitespace is dropped (as Forge's Krea 2 engine does): a lone space is a token of its own (220)
        # in Qwen's byte-level BPE, and "", " " and a prompt left with only LoRA tags must all be the empty prompt
        # Iris was trained on, its CFG null and the fine-tunes' conditioning
        return self.tokenizer.tokenizer.encode(text.strip(), add_special_tokens=False)

    def tokenize(self, texts: str | list[str]) -> int | list[int]:
        if isinstance(texts, str):
            return len(self.caption_ids(texts))
        return [len(self.caption_ids(t)) for t in texts]

    def token_ids(self, text: str) -> list[int]:
        budget = TEXT_LENGTH - len(self.suffix_ids)
        ids = self.prefix_ids + self.caption_ids(text)[:budget] + self.suffix_ids
        return ids + [PAD] * (len(self.prefix_ids) + TEXT_LENGTH - len(ids))

    def __call__(self, texts: list[str]) -> list[torch.Tensor]:
        if any(emphasis.uses_emphasis(text) for text in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = "None"

        zs = []
        cache = {}
        for line in texts:
            if line not in cache:
                cache[line] = self.encode(line)
            zs.append(cache[line])
        return zs

    def encode(self, line: str) -> torch.Tensor:
        pairs = [[(token, 1.0) for token in self.token_ids(line)]]
        states, _, extra = self.text_encoder.encode_token_weights(pairs)
        # the DiT consumes the last TEXT_LENGTH tokens: the padded system exchange stays behind
        states = states.transpose(1, 2)
        start = states.shape[1] - TEXT_LENGTH
        mask = extra["attention_mask"][:, start:]
        states = states[:, start:] * mask[:, :, None, None].to(states.dtype)
        return states[0]
