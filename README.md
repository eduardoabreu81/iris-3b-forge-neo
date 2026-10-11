# 🖼️ Iris 3B for Forge Neo

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![Version](https://img.shields.io/badge/Version-0.1.0-orange)](https://github.com/eduardoabreu81/iris-3b-forge-neo)
![Status](https://img.shields.io/badge/Status-Work_in_progress-orange)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green)](LICENSE)

> **Extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**

</div>

Generate images with **[Iris 3B](https://huggingface.co/speridlabs/iris-3b)** inside Forge Neo, using the checkpoint, preset and Generate button you already know. Iris is a 3B rectified-flow transformer that works **in pixel space**: it has **no VAE** — the transformer denoises the RGB image directly — and reads the **Qwen3-VL 4B** text encoder the Forge Neo already loads natively for Krea 2.

> [!Important]
> **Work in progress.** The native backend is built and covered by a CPU test suite (28 tests); real generations still need a GPU session. Text-to-image and image-to-image are the first targets, followed by the depth fine-tune and the upscaler LoRA.

> [!Note]
> This extension requires an up-to-date **Forge Neo** (`neo` branch). On an older version it stays disabled and tells you so in the console - Forge Neo itself keeps working as usual.

## How it works

- **No VAE** - the loader's VAE slot is answered by an empty stand-in; `decode_first_stage` / `encode_first_stage` are the identity (the same pattern Forge Neo's PiD engine uses), and the sampler's "latent" is the image itself.
- **Text encoder for free** - Iris reads the same Qwen3-VL 4B as Krea 2 (the Forge Neo loads it natively; `qwen3vl_4b_bf16.safetensors` from Comfy-Org/Qwen3-VL). The extension's text engine applies the Iris chat template, keeps the last 300 tokens, and runs the checkpoint's layerwise text adapter (12 raw Qwen layers) on the DiT.
- **INT8 ConvRot builds load as they are** - Forge's Krea 2 branch is the generic DiT loader; the extension swaps its transformer class for one call, inheriting dtype, quantization and device handling. The INT8 checkpoint zero-pads the SwiGLU FFN (6826 → 6912), and the detector reads the width from the shapes, so both builds round-trip exactly.
- **"Iris" schedule** - the upstream rectified-flow sigma grid with shift 4 (first sigma ≈ 0.999, last 0), registered in Forge's scheduler list.
- **"iris" UI preset** - DPM++ 2M, Iris, 50 steps, CFG 3.
- **Pixel-space preview** - the latent is RGB, so Forge's RGB live preview is exact.

## Installation

1. Place the extension in `extensions/iris-3b-forge-neo`
2. Place the model files:

| Part | File | Folder |
| :--- | :--- | :--- |
| Iris checkpoint | [iris-3b_bf16.safetensors](https://huggingface.co/nynxz/iris-3b) (6 GB) or `iris-3b_int8_convrot` (3.4 GB) | `models/Stable-diffusion` |
| Text encoder | [qwen3vl_4b_bf16.safetensors](https://huggingface.co/Comfy-Org/Qwen3-VL) (the same file Krea 2 uses) | `models/text_encoder` |

3. Pick the **iris** preset, the checkpoint, and the text encoder under **VAE / Text Encoder**

## Development

```bash
python -m unittest discover -s tests   # CPU suite, no Forge and no model weights needed
python -m ruff check .
```

The detection tests run against the real published checkpoints through their safetensors headers only (no weights downloaded).

## Credits

- **[Iris 3B](https://huggingface.co/speridlabs/iris-3b)** by Speridlabs (Apache-2.0)
- **[ComfyUI-Iris](https://github.com/envy-ai/ComfyUI-Iris)** by envy-ai - the reference port this backend follows (Apache-2.0, LICENSE and NOTICE included)
- **[nynxz/iris-3b](https://huggingface.co/nynxz/iris-3b)** - the bf16 and INT8 ConvRot builds
- **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by Haoming02
- Built on the patterns of the [MiniMax H3](https://github.com/eduardoabreu81/minimax-h3-forge-neo) and [Qwen-Image 2.1](https://github.com/eduardoabreu81/qwen2.1-forge-neo) extensions

## License

AGPL-3.0 - see [LICENSE](LICENSE)
