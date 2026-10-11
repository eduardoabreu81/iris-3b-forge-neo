# Iris 3B support for Forge Neo
# Recognize an Iris DiT and read its configuration from tensor shapes.
#
# Works on a loaded state dict or on a safetensors header: shapes are read through shape_of, so no tensor has to be
# materialized. Follows the MiniMax H3 extension's detect (forge_h3/native/detect.py).

import re

IDENTITY_KEYS = ("s_embedder.proj.weight", "pixel_embedder.proj.weight", "y_pos_embedding")


def shape_of(value):
    # a torch tensor, or a safetensors header entry {"dtype", "shape", "data_offsets"}
    if isinstance(value, dict):
        return tuple(value["shape"])
    return tuple(value.shape)


def count_blocks(keys, prefix):
    pattern = re.compile(re.escape(prefix) + r"(\d+)\.")
    found = {int(m.group(1)) for k in keys if (m := pattern.match(k))}
    return max(found) + 1 if found else 0


def is_iris(state_dict, key_prefix=""):
    return all(f"{key_prefix}{k}" in state_dict for k in IDENTITY_KEYS)


def detect(state_dict, key_prefix=""):
    """The IrisDiT keyword arguments for this checkpoint, or None when it is not an Iris DiT."""
    if not is_iris(state_dict, key_prefix):
        return None

    def shape(key):
        return shape_of(state_dict[f"{key_prefix}{key}"])

    keys = list(state_dict.keys())
    hidden = shape("s_embedder.proj.weight")[0]
    config = {"image_model": "iris"}
    config["hidden_size"] = hidden
    config["patch_size"] = round((shape("s_embedder.proj.weight")[1] / 3) ** 0.5)  # in = patch² * 3 channels
    config["depth"] = count_blocks(keys, f"{key_prefix}blocks.")
    # the first single-stream block swaps the dual-stream joint attention for plain q/k/v projections
    config["dual_depth"] = next((i for i in range(config["depth"]) if f"{key_prefix}blocks.{i}.q_proj.weight" in state_dict),
                                config["depth"])
    head_dim = shape("blocks.0.attn.q_norm_x.weight")[0]
    config["num_heads"] = shape("blocks.0.attn.q_proj_x.weight")[0] // head_dim
    config["num_kv_heads"] = shape("blocks.0.attn.k_proj_x.weight")[0] // head_dim
    # absolute widths: the INT8 checkpoint zero-pads the SwiGLU FFN (6826 -> 6912), and Forge must build with
    # exactly the width the checkpoint carries, so detect reads it instead of deriving it from a ratio
    config["ffn_hidden_size"] = shape("blocks.0.mlp_x.w1.weight")[0]
    config["text_len"] = shape("y_pos_embedding")[1]
    # the layerwise adapter runs on the raw text dim, not the model's hidden size; its head count is not in the
    # shapes (the fused qkv reshapes freely), so it stays at the IrisConfig default
    config["text_dim"] = shape("y_embedder.layer_blocks.0.attn.qkv.weight")[1]
    config["text_lap_num_layers"] = shape("y_embedder.layer_pool.weight")[1]  # Linear(layers, 1)
    config["text_lap_ffn_hidden_size"] = shape("y_embedder.layer_blocks.0.mlp.0.weight")[0]
    config["pixel_hidden_size"] = shape("pixel_embedder.proj.weight")[0]
    config["pixel_depth"] = count_blocks(keys, f"{key_prefix}pixel_blocks.")
    config["pixel_attn_hidden_size"] = shape("pixel_blocks.0.attn.proj.weight")[1]
    config["pixel_num_heads"] = config["pixel_attn_hidden_size"] // shape("pixel_blocks.0.attn.q_norm.weight")[0]
    config["pixel_ffn_hidden_size"] = shape("pixel_blocks.0.mlp.fc1.weight")[0]
    return config
