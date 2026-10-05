import launch

# Only install what is missing; never upgrade packages the WebUI already pins.
# (import name, pip requirement)
REQUIREMENTS = [
    ("rotary_embedding_torch", "rotary-embedding-torch>=0.5.3"),
    ("einops", "einops"),
    ("omegaconf", "omegaconf>=2.3.0"),
    ("diffusers", "diffusers>=0.33.1"),
    ("gguf", "gguf"),
    ("psutil", "psutil"),
    ("cv2", "opencv-python"),
]

for module, requirement in REQUIREMENTS:
    if not launch.is_installed(module):
        launch.run_pip(f"install {requirement}", f"{requirement} for SeedVR2")
