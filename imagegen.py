"""
imagegen.py - Frame Generation with Diffusers (sd-turbo) & Themed Fallbacks
==========================================================================
Generates 6 distinct visual frames based on storyboard prompts using:
1. Primary: Local PyTorch `diffusers` with `stabilityai/sd-turbo`.
   - Fast 1-step inference on CPU (8-15s) and sub-second on CUDA GPU.
   - Seed variation: `seed + shot_number` ensures consistent yet distinct narrative frames.
2. Fallback 1: Hugging Face Inference API (`HF_TOKEN` from environment).
3. Fallback 2: Dynamic idea-themed canvas renderer (custom palette derived from prompt).
4. AI-Generated Disclosure Label: Automatically stamped onto every generated frame.
"""

import os
import io
import time
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from dotenv import load_dotenv

load_dotenv()

# Global pipeline cache to avoid reloading model weights on each call
_LOCAL_DIFFUSER_PIPE = None
_LOCAL_MODEL_ID = "stabilityai/sd-turbo"


def stamp_ai_label(
    image: Image.Image,
    label_text: str = "✨ AI-Generated | Storyboard-to-GIF",
    shot_number: Optional[int] = None
) -> Image.Image:
    """
    Stamps an official 'AI-Generated' watermark label onto the bottom of a frame.
    Satisfies Responsible AI requirement for clear synthetic content disclosure.
    """
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Load default or fallback font
    try:
        font_large = ImageFont.truetype("arial.ttf", max(14, int(height * 0.035)))
        font_small = ImageFont.truetype("arial.ttf", max(11, int(height * 0.026)))
    except Exception:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Semi-transparent dark banner bar at bottom
    banner_height = int(height * 0.08)
    banner_y0 = height - banner_height

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([(0, banner_y0), (width, height)], fill=(15, 23, 42, 195))

    # Optional shot badge in top-left
    if shot_number is not None:
        badge_text = f"SHOT {shot_number}/6"
        overlay_draw.rounded_rectangle(
            [(12, 12), (105, 38)],
            radius=6,
            fill=(30, 41, 59, 220),
            outline=(99, 102, 241, 240),
            width=2
        )
        overlay_draw.text((22, 18), badge_text, fill=(248, 250, 252, 255), font=font_small)

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    text_color = (255, 255, 255)
    bbox = draw.textbbox((0, 0), label_text, font=font_large)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    text_x = (width - text_w) // 2
    text_y = banner_y0 + (banner_height - text_h) // 2

    draw.text((text_x, text_y), label_text, fill=text_color, font=font_large)
    return img


def _create_synthetic_story_frame(
    prompt: str,
    shot_number: int,
    seed: int,
    width: int = 512,
    height: int = 512
) -> Image.Image:
    """
    Dynamic idea-grounded offline renderer that creates visual art frames
    derived directly from the user's prompt text and shot progression.
    Colors and geometry change based on keywords in the prompt.
    """
    import random
    effective_seed = seed + shot_number * 1000
    rng = random.Random(effective_seed)

    prompt_lower = prompt.lower()

    # Thematic color generation based on prompt keywords
    if any(k in prompt_lower for k in ["dragon", "fire", "flame", "bread", "baking", "oven", "warm"]):
        palettes = [
            ((45, 15, 10), (194, 65, 12), (254, 215, 170)),
            ((55, 20, 15), (217, 119, 6), (253, 230, 138)),
            ((74, 25, 20), (220, 38, 38), (254, 202, 202)),
            ((60, 10, 20), (234, 88, 12), (255, 237, 213)),
            ((88, 28, 135), (245, 158, 11), (254, 240, 138)),
            ((30, 27, 75), (217, 119, 6), (254, 243, 199)),
        ]
    elif any(k in prompt_lower for k in ["rowing", "lake", "water", "ocean", "sea", "sunrise", "boat"]):
        palettes = [
            ((15, 23, 42), (14, 165, 233), (254, 215, 170)),
            ((30, 41, 59), (56, 189, 248), (224, 242, 254)),
            ((12, 74, 110), (2, 132, 199), (240, 249, 255)),
            ((88, 28, 135), (244, 63, 94), (254, 205, 211)),
            ((24, 24, 27), (245, 158, 11), (254, 243, 199)),
            ((15, 23, 42), (99, 102, 241), (224, 231, 255)),
        ]
    elif any(k in prompt_lower for k in ["robot", "greenhouse", "plant", "flower", "sunflower"]):
        palettes = [
            ((22, 101, 52), (34, 197, 94), (220, 252, 231)),
            ((20, 83, 45), (16, 185, 129), (209, 250, 229)),
            ((30, 41, 59), (234, 179, 8), (254, 249, 195)),
            ((71, 85, 105), (239, 68, 68), (254, 226, 226)),
            ((67, 56, 202), (245, 158, 11), (254, 240, 138)),
            ((15, 23, 42), (34, 197, 94), (220, 252, 231)),
        ]
    else:
        # Hash-based unique palette
        h = int(hashlib.md5(prompt.encode()).hexdigest()[:6], 16)
        r = (h >> 16) & 255
        g = (h >> 8) & 255
        b = h & 255
        palettes = [
            ((max(10, r//4), max(10, g//4), max(10, b//4)), (r, g, b), (250, 250, 250)),
            ((20, 30, 50), (min(255, r+30), min(255, g+30), b), (230, 245, 255)),
            ((40, 20, 30), (r, min(255, g+50), min(255, b+50)), (255, 240, 245)),
            ((10, 40, 30), (min(255, r+50), g, min(255, b+20)), (240, 255, 245)),
            ((45, 35, 15), (min(255, r+70), min(255, g+70), 50), (255, 255, 220)),
            ((20, 20, 35), (min(255, r+40), min(255, g+40), min(255, b+40)), (245, 245, 255)),
        ]

    c1, c2, c3 = palettes[(shot_number - 1) % len(palettes)]

    # Generate vertical gradient base
    base = Image.new("RGB", (width, height), c1)
    draw = ImageDraw.Draw(base)
    for y in range(height):
        ratio = y / height
        r = int(c1[0] * (1 - ratio) + c2[0] * ratio)
        g = int(c1[1] * (1 - ratio) + c2[1] * ratio)
        b = int(c1[2] * (1 - ratio) + c2[2] * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Celestial/focal orb
    orb_x = width // 2 + int(rng.uniform(-60, 60))
    orb_y = int(height * 0.30) + int(rng.uniform(-20, 20))
    orb_rad = int(width * 0.15)
    draw.ellipse(
        [(orb_x - orb_rad, orb_y - orb_rad), (orb_x + orb_rad, orb_y + orb_rad)],
        fill=c3,
        outline=(255, 255, 255, 160)
    )

    # Scenery layers
    for layer_i in range(3):
        layer_y = int(height * (0.50 + layer_i * 0.12))
        points = [(0, height)]
        step = 40
        for x in range(0, width + step, step):
            jitter = rng.randint(-20, 20)
            points.append((x, layer_y + jitter))
        points.append((width, height))
        shade = int(25 + layer_i * 40)
        layer_color = (
            max(0, c1[0] - shade),
            max(0, c1[1] - shade),
            max(0, c1[2] - shade)
        )
        draw.polygon(points, fill=layer_color)

    base = base.filter(ImageFilter.SMOOTH_MORE)

    # Narrative prompt overlay card
    draw = ImageDraw.Draw(base)
    try:
        font_title = ImageFont.truetype("arial.ttf", 15)
        font_sub = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    card_w, card_h = int(width * 0.90), int(height * 0.28)
    card_x = (width - card_w) // 2
    card_y = int(height * 0.58)

    card_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    card_draw = ImageDraw.Draw(card_overlay)
    card_draw.rounded_rectangle(
        [(card_x, card_y), (card_x + card_w, card_y + card_h)],
        radius=10,
        fill=(15, 23, 42, 215),
        outline=(148, 163, 184, 150),
        width=1
    )
    base = Image.alpha_composite(base.convert("RGBA"), card_overlay).convert("RGB")
    draw = ImageDraw.Draw(base)

    clean_prompt = prompt.replace(
        "digital storybook illustration, vibrant cinematic lighting, soft warm palette, expressive character design, sharp focus, consistent visual art style,", ""
    ).strip()
    words = clean_prompt.split()
    line1 = " ".join(words[:8])
    line2 = " ".join(words[8:16])
    line3 = " ".join(words[16:24])

    draw.text((card_x + 16, card_y + 10), f"SCENE BEAT {shot_number}/6", fill=(245, 158, 11), font=font_title)
    draw.text((card_x + 16, card_y + 32), line1, fill=(241, 245, 249), font=font_sub)
    if line2:
        draw.text((card_x + 16, card_y + 52), line2, fill=(203, 213, 225), font=font_sub)
    if line3:
        draw.text((card_x + 16, card_y + 72), line3, fill=(148, 163, 184), font=font_sub)

    return base


def _load_diffusers_pipeline():
    """Lazy-loads and caches local stabilityai/sd-turbo pipeline."""
    global _LOCAL_DIFFUSER_PIPE
    if _LOCAL_DIFFUSER_PIPE is not None:
        return _LOCAL_DIFFUSER_PIPE

    import torch
    from diffusers import AutoPipelineForText2Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[DIFFUSERS] Loading {_LOCAL_MODEL_ID} on {device.upper()} ({torch_dtype})...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        _LOCAL_MODEL_ID,
        torch_dtype=torch_dtype,
        variant="fp16" if device == "cuda" else None
    )
    pipe.to(device)

    if device == "cuda":
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

    _LOCAL_DIFFUSER_PIPE = pipe
    return pipe


def _generate_with_hf_api(prompt: str, seed: int, token: str) -> Image.Image:
    """Uses Hugging Face Inference API as a cloud fallback."""
    import requests

    headers = {"Authorization": f"Bearer {token}"}
    api_url = f"https://router.huggingface.co/hf-inference/models/{_LOCAL_MODEL_ID}"

    payload = {
        "inputs": prompt,
        "parameters": {
            "num_inference_steps": 2,
            "guidance_scale": 0.0,
            "seed": seed
        }
    }
    response = requests.post(api_url, headers=headers, json=payload, timeout=40)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        raise RuntimeError(f"HF Router returned {response.status_code}: {response.text[:120]}")


def generate_single_frame(
    prompt: str,
    shot_number: int,
    seed: int = 42,
    mode: str = "auto",
    hf_token: Optional[str] = None
) -> Tuple[Image.Image, str]:
    """
    Generates a single frame using local diffusers, HF Inference API, or idea-themed fallback.
    Returns stamped image and execution description.
    """
    token = hf_token or os.getenv("HF_TOKEN")
    env_mode = os.getenv("IMAGE_GEN_MODE", "auto").lower()
    selected_mode = mode if mode != "auto" else env_mode
    shot_seed = int(seed) + shot_number

    # Strategy 1: Explicit mock mode
    if selected_mode == "mock":
        img = _create_synthetic_story_frame(prompt, shot_number, shot_seed)
        stamped = stamp_ai_label(img, shot_number=shot_number)
        return stamped, "Dynamic Concept Canvas (Mock Mode)"

    # Strategy 2: Local diffusers with sd-turbo (CPU or GPU)
    if selected_mode in ("auto", "local_diffusers", "sd-turbo", "sd_turbo"):
        try:
            import torch
            pipe = _load_diffusers_pipeline()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(shot_seed)

            # sd-turbo requires only 1 to 2 steps with guidance_scale=0.0
            image = pipe(
                prompt=prompt,
                num_inference_steps=1,
                guidance_scale=0.0,
                generator=generator,
                width=512,
                height=512
            ).images[0]

            stamped = stamp_ai_label(image, shot_number=shot_number)
            desc = f"Local sd-turbo ({device.upper()})"
            return stamped, desc

        except Exception as e:
            print(f"[DIFFUSERS NOTICE] Local diffusers could not complete shot {shot_number}: {e}")

    # Strategy 3: HF Inference API (if token provided)
    if token and "your_huggingface" not in token:
        try:
            raw_img = _generate_with_hf_api(prompt, shot_seed, token)
            stamped = stamp_ai_label(raw_img, shot_number=shot_number)
            return stamped, "Hugging Face Inference API"
        except Exception as api_err:
            print(f"[HF API NOTICE] HF API unavailable ({api_err}). Falling back to concept canvas.")

    # Strategy 4: Dynamic idea-themed canvas renderer
    img = _create_synthetic_story_frame(prompt, shot_number, shot_seed)
    stamped = stamp_ai_label(img, shot_number=shot_number)
    return stamped, "Concept-Themed Canvas (Offline Fallback)"


def generate_storyboard_frames(
    shots: List[Dict[str, Any]],
    seed: int = 42,
    mode: str = "auto",
    hf_token: Optional[str] = None,
    progress_callback=None
) -> Tuple[List[Image.Image], List[str]]:
    """
    Generates all 6 storyboard frames sequentially with consistent yet distinct seeds.
    Seed = base_seed + shot_number.
    """
    frames = []
    methods = []
    total = len(shots)

    print(f"\n[IMAGE GEN] Starting generation of {total} frames (Base Seed: {seed})...")

    for i, shot in enumerate(shots):
        shot_num = shot.get("shot_number", i + 1)
        prompt = shot.get("image_prompt", "")
        shot_seed = int(seed) + shot_num

        if progress_callback:
            progress_callback((i + 1) / total, f"Rendering frame {shot_num} of {total} (Seed {shot_seed})...")

        t0 = time.time()
        img, method = generate_single_frame(
            prompt=prompt,
            shot_number=shot_num,
            seed=seed,
            mode=mode,
            hf_token=hf_token
        )
        elapsed = time.time() - t0
        frames.append(img)
        methods.append(method)

        print(f"[IMAGE GEN] Frame {shot_num}/{total} | Seed: {shot_seed} | Engine: {method} | Time: {elapsed:.2f}s")
        print(f"   Prompt: {prompt}")

    return frames, methods
