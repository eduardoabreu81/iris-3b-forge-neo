# 🖼️ Iris 3B for Forge Neo

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![Version](https://img.shields.io/badge/Version-0.2.0-brightgreen)](https://github.com/eduardoabreu81/iris-3b-forge-neo)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](LICENSE)

> **Extension for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)**

</div>

Generate with **[Iris 3B](https://huggingface.co/speridlabs/iris-3b)** inside Forge Neo, using the checkpoint, preset and Generate button you already know. Iris is a 3B image model that paints **every pixel directly** - there is no VAE to pick, and the live preview shows the real image as it forms. The same extension runs its **4× restoration and upscaling** and **depth** fine-tunes, and the **4-step Turbo LoRA**, even on 8 GB GPUs.

> [!Important]
> This extension requires an up-to-date **Forge Neo** (`neo` branch). On an older version it stays disabled and tells you so in the console - Forge Neo itself keeps working as usual.

---

## 📋 Table of Contents

- [What's New](#-whats-new)
- [Features](#-features)
- [Installation](#-installation)
- [Settings](#%EF%B8%8F-settings)
- [Turbo LoRA](#-turbo-lora)
- [Restore x4 and Depth](#-restore-x4-and-depth)
- [Other Iris Files](#-other-iris-files)
- [Credits](#-credits)

---

## 🆕 What's New

### v0.2.0 - Restoration, Depth and Turbo

- **Restore x4** and **Depth** in img2img, as the official recipes run them: one pass, tiled for large images, with the official colour fix
- The **Turbo LoRA** (4 steps), with an **Iris Turbo** schedule for it
- The **Iris DPM++ 2M** sampler - the exact sampler Iris was made with
- Works on GPUs without bfloat16 (RTX 20 series): the parts of the model that outgrow float16 run in float32
- Tested on an 8 GB RTX 2070 with the bf16, INT8 ConvRot, depth and official upscaler checkpoints

### v0.1.0 - First Release

- Text-to-image and image-to-image with Iris 3B, in pixel space, with the Qwen3-VL 4B text encoder Forge Neo already loads for Krea 2

---

## 🎯 Features

> [!Tip]
> Every example below, in full resolution and with its prompt and settings, is in the **[Wiki](https://github.com/eduardoabreu81/iris-3b-forge-neo/wiki)**.

### 🖼️ Text-to-Image

Iris paints the image directly, pixel by pixel. The bf16 and INT8 checkpoints give practically the same picture.

<p align="center"><img src=".github/images/text-to-image.jpg" alt="Text-to-image with the INT8 and bf16 checkpoints"></p>

### ⚡ Turbo LoRA

A 1024×1024 image in 4 steps - about 4 seconds on an RTX 2070. Without the LoRA, 4 steps give only a blur.

<p align="center"><img src=".github/images/turbo.jpg" alt="Base model, Turbo LoRA, and 4 steps without the LoRA"></p>

### 🔍 Restore x4

A small or damaged image, restored and enlarged four times - with the 0.5 GB upscaler LoRA on your Iris checkpoint, or with the official upscaler.

<p align="center"><img src=".github/images/restore.jpg" alt="Bicubic, Restore x4 with the LoRA, Restore x4 with the official upscaler"></p>

### 🌊 Depth

A depth map of any picture, near is bright - ready for depth ControlNets.

<p align="center"><img src=".github/images/depth.jpg" alt="Pictures and their depth maps"></p>

---

## 📦 Installation

1. Place the extension in `extensions/iris-3b-forge-neo`
2. Download one Iris checkpoint and the text encoder:

| Part | File | Folder |
| :--- | :--- | :--- |
| Iris checkpoint | [iris-3b_bf16.safetensors](https://huggingface.co/nynxz/iris-3b) (6 GB) or `iris-3b_int8_convrot.safetensors` (3.4 GB, faster and lighter; RTX 20 series or newer) | `models/Stable-diffusion` |
| Text encoder | [qwen3vl_4b_bf16.safetensors](https://huggingface.co/Comfy-Org/Qwen3-VL/tree/main/text_encoders) or `qwen3vl_4b_fp8_scaled.safetensors` (the same files Krea 2 uses) | `models/text_encoder` |

3. Pick the **iris** preset, the checkpoint, and the text encoder under **VAE / Text Encoder**

On an 8 GB RTX 2070, a 1024×1024 image takes about 95 seconds at the preset's 50 steps with the INT8 checkpoint.

---

## ⚙️ Settings

The **iris** preset sets them for you:

- **Sampler:** Iris DPM++ 2M · **Schedule type:** Iris
- **Steps:** 50 (100 matches the official samples) · **CFG:** 3
- **Negative prompt:** leave it empty - Iris learned its guidance from the empty prompt
- **Size:** about 1 megapixel, sides a multiple of 16: 1024×1024, 1152×896, 1280×832, 1344×768 (or portrait). Small sizes such as 512×512 show 16-pixel blocks.
- **Prompts:** up to about 295 tokens; longer prompts are cut

Good to know:

- **Hires. fix** is not supported yet: generate at the final size instead.
- **Settings > VAE:** keep *VAE for Decoding* and *VAE for Encoding* on **Full**. Iris has no VAE, so the fast approximations do not apply; the extension stops with a message if they are changed.
- **img2img** (variations of an image) works with *Just resize*, *Crop and resize* and *Resize and fill*. Iris reworks fine detail strongly even at low denoising - 0.3 already starts from mostly noise - so the result tends to look sharper than the input. To enlarge or clean up an image, use **Restore x4** instead.

---

## ⚡ Turbo LoRA

Download `r2-1000/student.safetensors` from the [Iris-3B Turbo LoRA](https://huggingface.co/pentacoxian-dev/iris-3b-turbo-lora), rename it (for example `iris-3b-turbo_r2-1000.safetensors`) and place it in `models/Lora`. Then:

- **Prompt:** your prompt plus `<lora:iris-3b-turbo_r2-1000:1>`
- **Sampler:** LCM · **Schedule type:** Iris Turbo · **Steps:** 4 · **CFG:** 1

It works with both checkpoints, with *Patch LoRAs on-the-fly* on or off. It trades some variety for speed: backgrounds and styles follow the prompt less closely than the base model.

---

## 🔍 Restore x4 and Depth

Iris comes with two fine-tunes that run in **one pass** instead of a sampling loop. In **img2img**, open the **Iris 3B** panel and pick the task:

| Task | Checkpoint | Prompt |
| :--- | :--- | :--- |
| **Restore x4** | your Iris checkpoint, with the [upscaler LoRA](https://huggingface.co/nynxz/iris-3b/tree/main/loras) (0.5 GB, in `models/Lora`) | `<lora:iris-3b-upscaler_r128:1>` |
| **Restore x4** | or the official [`upscaler/model.safetensors`](https://huggingface.co/speridlabs/iris-3b/tree/main/upscaler) (12 GB) as the checkpoint | leave it empty |
| **Depth** | [`iris-3b-depth_bf16`](https://huggingface.co/nynxz/iris-3b) or `iris-3b-depth_int8_convrot` as the checkpoint | leave it empty |

- Both tasks ignore the prompt text, steps, CFG and denoising; LoRA tags in the prompt still apply.
- **Restore x4** gives an image four times the input's size, and works best on inputs of 256 to 512 px. Large outputs are processed in overlapping 1024×1024 tiles, so a 512 px input (2048 px output) takes 9 passes.
- **Depth** runs at up to 1024 px on the long side and returns the map at the input's size. Inputs under about 600 px come out blocky, so enlarge them first. The map is relative depth, not distance.
- The official upscaler looks like a regular Iris checkpoint: use it only with **Restore x4**, as it cannot make new images.

---

## 📁 Other Iris Files

| File | Works? |
| :--- | :--- |
| The original `model.safetensors` ([speridlabs/iris-3b](https://huggingface.co/speridlabs/iris-3b), 12 GB) or `iris-3b.safetensors` ([nynxz](https://huggingface.co/nynxz/iris-3b)) | Yes, the same model at full precision (needs more RAM to load) |
| `Iris-3B-W4A8.safetensors` and the GGUF files ([realrebelai/iris-3b_GGUFs](https://huggingface.co/realrebelai/iris-3b_GGUFs)) | Should load (smaller, lower quality); not tested yet |
| The official `depth/model.safetensors` | No: use nynxz's `iris-3b-depth_*`, the same model in the standard layout |
| [Aikimi/iris-3b-int8](https://huggingface.co/Aikimi/iris-3b-int8) and [Aikimi/iris-3b-w4a8](https://huggingface.co/Aikimi/iris-3b-w4a8) | No: they are made for Aikimi Forge Neo only, and Forge Neo says so |

---

## 🙏 Credits

- **[Iris 3B](https://huggingface.co/speridlabs/iris-3b)** by Speridlabs ([code](https://github.com/speridlabs/iris-3b), Apache-2.0) - the model, its sampler, and the restoration and depth recipes
- **[ComfyUI-Iris](https://github.com/envy-ai/ComfyUI-Iris)** by envy-ai - the reference port this backend follows (Apache-2.0; see [NOTICE](NOTICE) and [LICENSE-APACHE](LICENSE-APACHE))
- **[nynxz/iris-3b](https://huggingface.co/nynxz/iris-3b)** and [Nynxz/ComfyUI-Iris](https://github.com/Nynxz/ComfyUI-Iris) - the bf16, INT8 ConvRot and depth builds, the upscaler LoRA, and the sampling recommendations
- **[Iris-3B Turbo LoRA](https://huggingface.co/pentacoxian-dev/iris-3b-turbo-lora)** by pentacoxian-dev
- **[realrebelai/iris-3b_GGUFs](https://huggingface.co/realrebelai/iris-3b_GGUFs)** - the GGUF and W4A8 builds
- **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by Haoming02
- Built on the patterns of the [MiniMax H3](https://github.com/eduardoabreu81/minimax-h3-forge-neo) and [Qwen-Image 2.1](https://github.com/eduardoabreu81/qwen2.1-forge-neo) extensions

## 📄 License

AGPL-3.0 - see [LICENSE](LICENSE). Code ported from Apache-2.0 projects keeps its notices in [NOTICE](NOTICE).
