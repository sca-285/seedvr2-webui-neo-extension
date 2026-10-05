import os
import sys
import random
import importlib
import importlib.util
import traceback
from contextlib import contextmanager
from types import SimpleNamespace

import gradio as gr
import numpy as np
import torch
from PIL import Image

from modules import scripts, shared, devices

try:
    from modules.ui_components import InputAccordion, ToolButton
except ImportError:  # very old A1111 builds
    InputAccordion = None
    from modules.ui_components import ToolButton

EXTENSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cache slots used for the global SeedVR2 model cache
DIT_CACHE_ID = "webui_dit"
VAE_CACHE_ID = "webui_vae"

MODEL_EXTENSIONS = (".gguf", ".safetensors")


class SeedVR2Interrupted(Exception):
    pass


# ============================================================
# LOADING THE SeedVR2 PACKAGE
# ============================================================
# The bundled code lives in ./src, which used to be imported as the top-level
# package `src` after pushing the extension root onto sys.path[0]. `src` is
# about the most common package name there is: if another extension (or the
# WebUI itself) got there first, `import src.core...` resolves to their code
# and fails, and in the other direction our sys.path entry shadows theirs.
# The package only uses relative imports internally, so load it under a name
# nobody else will pick, without touching sys.path at all.

_PACKAGE_NAME = "sd_webui_seedvr2_src"
_seedvr2 = None


def load_seedvr2():
    global _seedvr2
    if _seedvr2 is not None:
        return _seedvr2

    if _PACKAGE_NAME not in sys.modules:
        src_dir = os.path.join(EXTENSION_ROOT, "src")
        spec = importlib.util.spec_from_file_location(
            _PACKAGE_NAME,
            os.path.join(src_dir, "__init__.py"),
            submodule_search_locations=[src_dir],
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[_PACKAGE_NAME] = package
        try:
            spec.loader.exec_module(package)
        except Exception:
            del sys.modules[_PACKAGE_NAME]
            raise

    def sub(name):
        return importlib.import_module(f"{_PACKAGE_NAME}.{name}")

    _seedvr2 = SimpleNamespace(
        debug=sub("utils.debug"),
        utils=sub("core.generation_utils"),
        phases=sub("core.generation_phases"),
        cache=sub("core.model_cache"),
    )
    return _seedvr2


# ============================================================
# MODEL FILES
# ============================================================
# The README has said ./model/seedvr2 while the code looked in ./models/SeedVR2;
# on Linux those are different folders. Accept any capitalisation of
# "seedvr2" under the extension's model(s) folder and the WebUI models folder.

def model_dirs():
    bases = [
        os.path.join(EXTENSION_ROOT, "models"),
        os.path.join(EXTENSION_ROOT, "model"),
        getattr(shared, "models_path", None),
    ]
    found, seen = [], set()
    for base in bases:
        if not base or not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            path = os.path.join(base, name)
            if name.lower() != "seedvr2" or not os.path.isdir(path):
                continue
            key = os.path.normcase(os.path.realpath(path))
            if key not in seen:
                seen.add(key)
                found.append(path)
    return found


def scan_models():
    """filename -> folder it lives in. The first folder wins on duplicates."""
    files = {}
    for folder in model_dirs():
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(MODEL_EXTENSIONS) and name not in files:
                files[name] = folder
    return files


def model_choices():
    files = list(scan_models())
    dit = [f for f in files if "vae" not in f.lower()] or files
    vae = [f for f in files if "vae" in f.lower()] or files
    default_dit = dit[0] if dit else None
    default_vae = next((f for f in vae if "ema_vae_fp16" in f), vae[0] if vae else None)
    return dit, vae, default_dit, default_vae


# ============================================================
# GETTING THE CHECKPOINT OUT OF VRAM, AND BACK
# ============================================================
# Every fork spells this differently, and one of them spells it in a way that
# cannot be undone with the same call:
#
#   A1111 / reForge   unload_model_weights()  moves the model to CPU
#                     reload_model_weights()  brings it back
#   Forge             unload_model_weights()  = memory_management.unload_all_models()
#                     reload_model_weights()  is literally `pass` - nothing to undo
#   Forge Classic     unload_model_weights()  REPLACES model_data.sd_model with a
#   (Neo)             FakeInitialModel and clears forge_hash. There is no
#                     reload_model_weights at all, and because shared.sd_model
#                     is then a FakeInitialModel, create_infotext blows up on
#                     use_distilled_cfg_scale while saving the image.
#
# So on Neo we do what Forge's own unload does instead: go straight to the
# memory manager, which frees the VRAM and leaves shared.sd_model alone. Then
# there is nothing to put back, and nothing left holding a fake model.

def _memory_manager():
    """The Forge-family loader, under whichever name this build uses."""
    for path in ("backend.memory_management",              # Forge, Forge Classic / Neo
                 "ldm_patched.modules.model_management"):  # reForge
        try:
            return importlib.import_module(path)
        except Exception:
            continue
    return None


def _model_is_real():
    """False when shared.sd_model is missing or Neo's placeholder."""
    model = getattr(shared, "sd_model", None)
    return model is not None and type(model).__name__ != "FakeInitialModel"


def offload_sd_model():
    """Free the checkpoint's VRAM. Returns a callable that restores it."""
    import modules.sd_models as sd_models

    if getattr(shared, "sd_model", None) is None:
        return lambda: None

    # A1111, Forge and reForge: the pair they already document.
    if callable(getattr(sd_models, "reload_model_weights", None)):
        sd_models.unload_model_weights()
        return sd_models.reload_model_weights

    # Forge Classic (Neo). Its unload_model_weights() is a one-way door.
    manager = _memory_manager()
    if manager is not None and callable(getattr(manager, "unload_all_models", None)):
        manager.unload_all_models()
        if callable(getattr(manager, "soft_empty_cache", None)):
            manager.soft_empty_cache()
        return restore_sd_model

    # Nothing better left: take the destructive route, then rebuild.
    sd_models.unload_model_weights()
    return restore_sd_model


def restore_sd_model():
    """Put a real checkpoint back in shared.sd_model if one is missing.

    Cheap when nothing was destroyed - it only reloads when the model really is
    gone, and Neo's forge_model_reload() is itself a no-op when the hash still
    matches."""
    if _model_is_real():
        return

    import modules.sd_models as sd_models

    for name in ("forge_model_reload", "reload_model_weights"):
        fn = getattr(sd_models, name, None)
        if not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:
            print(f"[SeedVR2] {name}() failed: {exc}")
            continue
        if _model_is_real():
            return

    print(
        "[SeedVR2] the checkpoint could not be put back in memory. Saving this "
        "image may fail; switch checkpoint once to recover."
    )


# ============================================================
# HELPERS
# ============================================================

@contextmanager
def preserved_rng_state():
    """SeedVR2 reseeds python/numpy/torch globally on every run. Without this,
    the WebUI's own "random" seeds (seed -1) become a pure function of the
    SeedVR2 seed, so the next generation keeps rolling the same "random" seed."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = None
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            cuda_states = torch.cuda.get_rng_state_all()
    except Exception:
        cuda_states = None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            try:
                torch.cuda.set_rng_state_all(cuda_states)
            except Exception:
                pass


def resolve_seed(seed, p):
    seed = -1 if seed is None else int(seed)
    if seed >= 0:
        return seed
    # Follow the seed of *this* image, not just the first one of the batch.
    batch_index = getattr(p, "batch_index", 0) or 0
    for seeds in (getattr(p, "seeds", None), getattr(p, "all_seeds", None)):
        try:
            value = int(seeds[batch_index])
            if value >= 0:
                return value
        except Exception:
            continue
    base = getattr(p, "seed", None)
    try:
        if base is not None and int(base) >= 0:
            return int(base)
    except Exception:
        pass
    return random.SystemRandom().randint(0, 2**32 - 1)


def inference_device():
    device = getattr(devices, "device", None)
    if device is None and callable(getattr(devices, "get_optimal_device", None)):
        device = devices.get_optimal_device()
    return str(device) if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")


def image_to_tensor(img):
    """PIL -> [1, H, W, C] float16 in 0..1. Keeps alpha; SeedVR2 handles RGBA."""
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)
    img = img.convert("RGBA" if has_alpha else "RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).to(dtype=torch.float16)


def tensor_to_image(frame):
    """[H, W, C] in 0..1 -> PIL."""
    arr = frame.detach().to(device="cpu", dtype=torch.float32).clamp_(0, 1).mul_(255).round_().to(torch.uint8).numpy()
    return Image.fromarray(arr, "RGBA" if arr.shape[-1] == 4 else "RGB")


def interrupted():
    state = shared.state
    return bool(getattr(state, "interrupted", False) or getattr(state, "skipped", False))


def drop_cached_models(sv, debug=None):
    cache = sv.cache.get_global_cache()
    cache.remove_dit({"node_id": DIT_CACHE_ID}, debug)
    cache.remove_vae({"node_id": VAE_CACHE_ID}, debug)


# ============================================================
# SCRIPT
# ============================================================

class Script(scripts.Script):

    def title(self):
        return "SeedVR2 Native Upscaler"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        dit_choices, vae_choices, default_dit, default_vae = model_choices()
        tab = "img2img" if is_img2img else "txt2img"

        def section():
            if InputAccordion is not None:
                return InputAccordion(False, label="SeedVR2 Native Upscaler", elem_id=f"{tab}_seedvr2")
            return gr.Accordion("SeedVR2 Native Upscaler", open=False)

        with section() as accordion:
            if InputAccordion is not None:
                enabled = accordion
            else:
                enabled = gr.Checkbox(label="Enable", value=False, elem_id=f"{tab}_seedvr2_enabled")

            with gr.Row():
                dit_model = gr.Dropdown(label="DiT Model", choices=dit_choices, value=default_dit, elem_id=f"{tab}_seedvr2_dit")
                vae_model = gr.Dropdown(label="VAE Model", choices=vae_choices, value=default_vae, elem_id=f"{tab}_seedvr2_vae")
                refresh = ToolButton(value="\U0001f504", elem_id=f"{tab}_seedvr2_refresh")

            with gr.Row():
                seed = gr.Number(label="Seed (-1 = follow image seed)", value=-1, precision=0, elem_id=f"{tab}_seedvr2_seed")
                resolution = gr.Slider(label="Upscale Resolution (Shortest Edge)", minimum=512, maximum=3840, step=64, value=1080, elem_id=f"{tab}_seedvr2_resolution")

            with gr.Row():
                unload_sd = gr.Checkbox(label="Unload SD Checkpoint", value=False, elem_id=f"{tab}_seedvr2_unload_sd")
                keep_cached = gr.Checkbox(label="Keep SeedVR2 models in RAM", value=True, elem_id=f"{tab}_seedvr2_keep_cached")
                force_reload = gr.Checkbox(label="Force Reload", value=False, elem_id=f"{tab}_seedvr2_force_reload")
                debug_mode = gr.Checkbox(label="Show Debug Logs", value=False, elem_id=f"{tab}_seedvr2_debug")

            with gr.Accordion("Advanced Settings (Noise & Tiling)", open=False):
                with gr.Row():
                    input_noise = gr.Slider(label="Input Noise Scale", minimum=0.0, maximum=1.0, step=0.05, value=0.0, elem_id=f"{tab}_seedvr2_input_noise")
                    latent_noise = gr.Slider(label="Latent Noise Scale", minimum=0.0, maximum=1.0, step=0.05, value=0.0, elem_id=f"{tab}_seedvr2_latent_noise")

                with gr.Row():
                    use_tile_vae = gr.Checkbox(label="Enable VAE Tiling", value=True, elem_id=f"{tab}_seedvr2_tile_vae")
                    tile_size = gr.Slider(label="Tile Size", minimum=512, maximum=2048, step=128, value=1024, elem_id=f"{tab}_seedvr2_tile_size")
                    tile_overlap = gr.Slider(label="Tile Overlap", minimum=64, maximum=512, step=32, value=128, elem_id=f"{tab}_seedvr2_tile_overlap")

        def refresh_models(current_dit, current_vae):
            dit, vae, d_dit, d_vae = model_choices()
            return (
                gr.update(choices=dit, value=current_dit if current_dit in dit else d_dit),
                gr.update(choices=vae, value=current_vae if current_vae in vae else d_vae),
            )

        refresh.click(fn=refresh_models, inputs=[dit_model, vae_model], outputs=[dit_model, vae_model])

        return [
            enabled,
            dit_model, vae_model, seed, resolution,
            input_noise, latent_noise, force_reload, unload_sd,
            use_tile_vae, tile_size, tile_overlap, debug_mode,
            keep_cached,
        ]

    def postprocess_image(self, p, pp, *args):
        (
            enabled,
            dit_model_name, vae_model_name, seed, resolution,
            input_noise, latent_noise, force_reload, unload_sd,
            use_tile_vae, tile_size, tile_overlap, debug_mode,
            *rest,
        ) = args
        keep_cached = rest[0] if rest else True

        if not enabled or getattr(pp, "image", None) is None:
            return
        if interrupted():
            return

        try:
            with preserved_rng_state():
                self.upscale(
                    p, pp,
                    dit_model_name=dit_model_name, vae_model_name=vae_model_name,
                    seed=seed, resolution=int(resolution),
                    input_noise=float(input_noise), latent_noise=float(latent_noise),
                    force_reload=bool(force_reload), unload_sd=bool(unload_sd),
                    use_tile_vae=bool(use_tile_vae), tile_size=int(tile_size), tile_overlap=int(tile_overlap),
                    debug_mode=bool(debug_mode), keep_cached=bool(keep_cached),
                )
        except Exception as e:
            traceback.print_exc()
            print(f"[SeedVR2] Error: {e}")

    def upscale(self, p, pp, *, dit_model_name, vae_model_name, seed, resolution,
                input_noise, latent_noise, force_reload, unload_sd,
                use_tile_vae, tile_size, tile_overlap, debug_mode, keep_cached):
        models = scan_models()
        missing = [n for n in (dit_model_name, vae_model_name) if not n or n not in models]
        if missing:
            dirs = model_dirs()
            where = ", ".join(dirs) if dirs else os.path.join(EXTENSION_ROOT, "models", "SeedVR2")
            print(f"[SeedVR2] Model file not found: {', '.join(str(m) for m in missing)} (looked in: {where}). Skipping.")
            return

        sv = load_seedvr2()
        debug = sv.debug.Debug(enabled=debug_mode)

        if force_reload:
            drop_cached_models(sv, debug)

        actual_seed = resolve_seed(seed, p)
        input_img = pp.image
        old_textinfo = getattr(shared.state, "textinfo", None)
        shared.state.textinfo = "SeedVR2 upscaling..."

        def check_interruption(*_args, **_kwargs):
            if interrupted():
                raise SeedVR2Interrupted("User interrupted")

        restore = None
        if unload_sd:
            try:
                restore = offload_sd_model()
            except Exception as e:
                print(f"[SeedVR2] Offload warning: {e}")
                restore = restore_sd_model
            devices.torch_gc()

        ctx = None
        try:
            device = inference_device()
            # A fresh context per image: phase state (latents, transforms,
            # alpha buffers) must never leak from one image, or one interrupted
            # run, into the next. The models are cached separately.
            ctx = sv.utils.setup_generation_context(
                dit_device=device, vae_device=device,
                dit_offload_device="cpu", vae_offload_device="cpu",
                tensor_offload_device="cpu", debug=debug,
            )
            # Honour the WebUI's Interrupt/Skip between batches as well.
            ctx["interrupt_fn"] = check_interruption

            tile = (tile_size, tile_size)
            overlap = (tile_overlap, tile_overlap)
            runner, cache_context = sv.utils.prepare_runner(
                dit_model=dit_model_name, vae_model=vae_model_name,
                model_dir=models[dit_model_name], debug=debug, ctx=ctx,
                dit_cache=keep_cached, vae_cache=keep_cached,
                dit_id=DIT_CACHE_ID, vae_id=VAE_CACHE_ID,
                block_swap_config={"blocks_to_swap": 0, "swap_io_components": False, "offload_device": None},
                encode_tiled=use_tile_vae, encode_tile_size=tile, encode_tile_overlap=overlap,
                decode_tiled=use_tile_vae, decode_tile_size=tile, decode_tile_overlap=overlap,
                tile_debug="false", attention_mode="sdpa",
                torch_compile_args_dit=None, torch_compile_args_vae=None,
            )
            # The DiT and VAE may live in different folders.
            if models[vae_model_name] != models[dit_model_name]:
                runner._vae_checkpoint = os.path.join(models[vae_model_name], vae_model_name)
            ctx["cache_context"] = cache_context

            frames, _ = sv.utils.compute_generation_info(
                ctx=ctx, images=image_to_tensor(input_img), resolution=resolution,
                max_resolution=0, batch_size=1, uniform_batch_size=False,
                seed=actual_seed, prepend_frames=0, temporal_overlap=0, debug=debug,
            )

            tiled = " (tiled)" if use_tile_vae else ""
            print(f"[SeedVR2] Phase 1/4: Encoding{tiled}...")
            ctx = sv.phases.encode_all_batches(
                runner, ctx=ctx, images=frames, debug=debug,
                batch_size=1, uniform_batch_size=False, seed=actual_seed,
                progress_callback=check_interruption, temporal_overlap=0,
                resolution=resolution, max_resolution=0,
                input_noise_scale=input_noise, color_correction="lab",
            )
            del frames

            print("[SeedVR2] Phase 2/4: Upscaling...")
            ctx = sv.phases.upscale_all_batches(
                runner, ctx=ctx, debug=debug, progress_callback=check_interruption,
                seed=actual_seed, latent_noise_scale=latent_noise, cache_model=keep_cached,
            )

            print(f"[SeedVR2] Phase 3/4: Decoding{tiled}...")
            ctx = sv.phases.decode_all_batches(
                runner, ctx=ctx, debug=debug, progress_callback=check_interruption, cache_model=keep_cached,
            )

            print("[SeedVR2] Phase 4/4: Post-processing...")
            ctx = sv.phases.postprocess_all_batches(
                ctx=ctx, debug=debug, progress_callback=check_interruption,
                color_correction="lab", prepend_frames=0, temporal_overlap=0, batch_size=1,
            )

            output = ctx.get("final_video")
            if output is None or output.numel() == 0:
                print("[SeedVR2] No output was produced; keeping the original image.")
                return

            pp.image = tensor_to_image(output[0])

            info = [
                f"DiT:{dit_model_name}",
                f"VAE:{vae_model_name}",
                f"Res:{resolution}",
                f"Seed:{actual_seed}",
            ]
            if input_noise > 0 or latent_noise > 0:
                info.append(f"Noise(In:{input_noise} Lat:{latent_noise})")
            if use_tile_vae:
                info.append(f"Tile({tile_size}x{tile_size} Overlap:{tile_overlap})")
            p.extra_generation_params["SeedVR2"] = " | ".join(info)

            print(f"[SeedVR2] Upscaled to {pp.image.width}x{pp.image.height}.")

        except SeedVR2Interrupted:
            # Each phase offloads its model in its own finally block, so the
            # cache is still valid here.
            print("[SeedVR2] Interrupted.")
        finally:
            if ctx is not None:
                ctx.clear()
            ctx = None

            if not keep_cached:
                drop_cached_models(sv, debug)

            if unload_sd:
                try:
                    print("[SeedVR2] Restoring the SD checkpoint...")
                    (restore or restore_sd_model)()
                except Exception as e:
                    print(f"[SeedVR2] Failed to restore the model: {e}")
                # Whatever happened above, the rest of this generation still
                # has to write infotext off shared.sd_model. Leaving a
                # placeholder there is what turns a warning into a crash.
                try:
                    restore_sd_model()
                except Exception as e:
                    print(f"[SeedVR2] Model check failed: {e}")

            shared.state.textinfo = old_textinfo
            try:
                devices.torch_gc()
            except Exception:
                pass
