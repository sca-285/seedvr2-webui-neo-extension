# seedvr2-webui-neo-extension

SeedVR2 upscaling for the Stable Diffusion WebUI Forge family:

| WebUI | Status |
| --- | --- |
| Forge Classic / **Neo** (Haoming02) | supported |
| **reForge** (Panchovix) | supported |
| **Forge** (lllyasviel) | supported |
| A1111 1.7+ | should work (same API as reForge) |

## Install

Install from URL in the Extensions tab, or clone into `extensions/`.
`install.py` installs only the packages that are missing (`rotary-embedding-torch`, `einops`,
`omegaconf`, `diffusers`, `gguf`, `psutil`, `opencv-python`); it never upgrades what the WebUI
already pins. If it fails, install them by hand inside the WebUI's venv:

```
pip install rotary-embedding-torch
```

## Models

Put the files in any of these folders (the folder name is case-insensitive):

- `<webui>/models/SeedVR2/`  (recommended)
- `<extension>/models/SeedVR2/`

You need one VAE (`ema_vae_fp16.safetensors`) and one DiT, e.g.
`seedvr2_ema_3b_fp16.safetensors` or `seedvr2_ema_7b_sharp-Q4_K_M.gguf`.
Files with `vae` in their name are listed as VAE, everything else as DiT. 7B models are
detected by `7b` in the file name. Use the 🔄 button to rescan without restarting.

## Usage

Open the **SeedVR2 Native Upscaler** section in txt2img or img2img and tick it. After each
image is generated (after hires fix), SeedVR2 upscales it so its shortest edge is
**Upscale Resolution**. In img2img the normal img2img pass still runs first.

- **Seed** `-1` follows each image's own seed.
- **Unload SD Checkpoint** frees the checkpoint's VRAM while SeedVR2 runs. Use it when you are
  short on VRAM, especially with 7B models.
- **Keep SeedVR2 models in RAM** keeps the DiT/VAE in system RAM between images, so they don't
  have to be read from disk every time. Turn it off if you are short on RAM (a 7B fp16 DiT
  needs ~16 GB).
- **Force Reload** drops the cached models and loads them again from disk.
- **Enable VAE Tiling** lowers VRAM use for large outputs.

Interrupt and Skip work while SeedVR2 runs. The settings are written to the image's infotext
under `SeedVR2`.

---

## Tiếng Việt

Đặt model vào `<webui>/models/SeedVR2/` (hoặc `<extension>/models/SeedVR2/`), gồm
`ema_vae_fp16.safetensors` và một model DiT như `seedvr2_ema_7b_sharp-Q4_K_M.gguf`.
Mở mục **SeedVR2 Native Upscaler** trong txt2img/img2img, tích chọn và chỉnh
**Upscale Resolution** (cạnh ngắn của ảnh đầu ra). Thiếu VRAM thì bật **Unload SD Checkpoint**
và **Enable VAE Tiling**; thiếu RAM thì tắt **Keep SeedVR2 models in RAM**.

<img width="1660" height="694" alt="image" src="https://github.com/user-attachments/assets/777c34e7-aca6-4e51-9994-f02f817311ea" />
