# Iris 3B support for Forge Neo
# The Iris text encoder: Qwen3-VL 4B raw hidden states of layers [2, 5, ... 35] under the Iris chat template.
#
# The whole exchange (system instruction, user text, assistant suffix) is encoded, padded to system + 300 tokens,
# and only the last 300 states go to the DiT (upstream's text_len), pads zeroed and masked. The DiT's
# preprocess_text runs the layerwise adapter, so the conditioning cache holds the final tensor.

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
    def __init__(self, text_encoder, tokenizer, dit):
        self.text_encoder = SDClipModel(text_encoder, layer=HIDDEN_LAYERS, layer_idx=None,
                                        special_tokens={"pad": PAD}, layer_norm_hidden_state=False,
                                        enable_attention_masks=True, return_attention_masks=True)
        self.tokenizer = SDTokenizer(tokenizer, pad_with_end=False, has_start_token=False, has_end_token=False,
                                     pad_to_max_length=False, max_length=INF, min_length=1, pad_token=PAD)
        self.dit = dit
        self.prefix_ids = self.tokenizer.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.tokenizer.encode(SUFFIX, add_special_tokens=False)

    @property
    def emphasis(self) -> "emphasis.Emphasis":
        return emphasis.EmphasisNone()

    def tokenize(self, texts: str | list[str]) -> int | list[int]:
        if isinstance(texts, str):
            return len(self.tokenizer.tokenizer.encode(texts, add_special_tokens=False))
        return [len(self.tokenizer.tokenizer.encode(t, add_special_tokens=False)) for t in texts]

    def token_ids(self, text: str) -> list[int]:
        budget = TEXT_LENGTH - len(self.suffix_ids)
        ids = self.prefix_ids + self.tokenizer.tokenizer.encode(text, add_special_tokens=False)[:budget] + self.suffix_ids
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
        states = states[:, start:]
        mask = extra["attention_mask"][:, start:]
        states = states * mask[:, :, None, None].to(states.dtype)
        # the adapter rides on the DiT, so the compute runs wherever its weights are
        device = self.dit.y_pos_embedding.device
        return self.dit.preprocess_text(states.to(device), mask.to(device))
