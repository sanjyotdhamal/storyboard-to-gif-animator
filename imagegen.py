"""
imagegen.py - Frame Generation with Diffusers (sd-turbo) & Fallbacks
====================================================================
Generates 6 visual frames based on storyboard prompts using:
1. Primary: Hugging Face `diffusers` with `stabilityai/sd-turbo` (PyTorch).
   - Fast 1-4 step inference.
   - Fixed seed reproducibility (default seed=42).
2. Fallback 1: Hugging Face Inference API (`HF_TOKEN` from env var).
3. Fallback 2: High-aesthetic offline visual canvas renderer (for instant offline demos/testing).
4. AI-Generated Disclosure Label: Automatically stamped onto every generated frame.
"""

import os
import io
import time
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
    Stamps an official, elegant 'AI-Generated' watermark label onto the bottom of a frame.
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
    
    # Create overlay for smooth alpha transparency
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([(0, banner_y0), (width, height)], fill=(15, 23, 42, 185))

    # Optional shot badge in top-left
    if shot_number is not None:
        badge_text = f"SHOT {shot_number}/6"
        overlay_draw.rounded_rectangle(
            [(12, 12), (100, 38)],
            radius=6,
            fill=(30, 41, 59, 210),
            outline=(99, 102, 241, 230),
            width=2
        )
        overlay_draw.text((22, 18), badge_text, fill=(248, 250, 252, 255), font=font_small)

    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw label text centered inside bottom banner
    text_color = (255, 255, 255)
    accent_color = (244, 114, 182) # soft coral/pink highlight
    
    # Calculate bounding box for centering
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
    Offline synthetic renderer that creates high-aesthetic visual art frames
    with gradients, geometric scenery, and narrative atmosphere.
    Used for instant unit testing, offline environments, or CPU speed fallback.
    """
    import random
    rng = random.Random(seed + shot_number * 1000)

    # Narrative color mood palettes based on shot progression (Act 1 -> Act 2 -> Climax -> Resolution)
    palettes = [
        # Shot 1: Sunrise / beginning
        ((45, 55, 72), (237, 137, 54), (254, 235, 200)),
        # Shot 2: Discovery
        ((26, 32, 44), (79, 209, 197), (230, 255, 250)),
        # Shot 3: Action / effort
        ((44, 82, 130), (99, 179, 237), (235, 248, 255)),
        # Shot 4: Tension / turning point
        ((116, 42, 42), (237, 100, 166), (254, 215, 226)),
        # Shot 5: Climax brilliance
        ((85, 60, 154), (246, 173, 85), (254, 252, 191)),
        # Shot 6: Peaceful sunset / resolution
        ((26, 54, 93), (129, 230, 217), (255, 245, 245)),
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

    # Layer atmospheric scenic elements (mountains, celestial orbs, glowing horizon)
    # 1. Celestial orb (sun / moon / glowing core)
    orb_x = width // 2 + int(rng.uniform(-80, 80))
    orb_y = int(height * 0.35) + int(rng.uniform(-30, 30))
    orb_rad = int(width * 0.16)
    draw.ellipse(
        [(orb_x - orb_rad, orb_y - orb_rad), (orb_x + orb_rad, orb_y + orb_rad)],
        fill=c3,
        outline=(255, 255, 255, 180)
    )

    # 2. Layered mountain / landscape silhouettes
    for layer_i in range(3):
        layer_y = int(height * (0.55 + layer_i * 0.12))
        points = [(0, height)]
        step = 40
        for x in range(0, width + step, step):
            jitter = rng.randint(-25, 25)
            points.append((x, layer_y + jitter))
        points.append((width, height))
        shade_alpha = int(30 + layer_i * 45)
        layer_color = (
            max(0, c1[0] - shade_alpha),
            max(0, c1[1] - shade_alpha),
            max(0, c1[2] - shade_alpha)
        )
        draw.polygon(points, fill=layer_color)

    # 3. Vignette soft blur effect
    base = base.filter(ImageFilter.SMOOTH_MORE)
    
    # 4. Narrative prompt overlay in the center
    draw = ImageDraw.Draw(base)
    try:
        font_title = ImageFont.truetype("arial.ttf", 16)
        font_sub = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    # Draw semi-transparent center card for prompt readability
    card_w, card_h = int(width * 0.88), int(height * 0.28)
    card_x = (width - card_w) // 2
    card_y = int(height * 0.58)
    
    card_overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    card_draw = ImageDraw.Draw(card_overlay)
    card_draw.rounded_rectangle(
        [(card_x, card_y), (card_x + card_w, card_y + card_h)],
        radius=10,
        fill=(15, 23, 42, 210),
        outline=(148, 163, 184, 150),
        width=1
    )
    base = Image.alpha_composite(base.convert("RGBA"), card_overlay).convert("RGB")
    draw = ImageDraw.Draw(base)

    # Truncate and wrap text for visual appeal
    words = prompt.replace("digital storybook illustration, vibrant cinematic lighting, soft warm palette, expressive character design, sharp focus, consistent visual art style,", "").strip().split()
    line1 = " ".join(words[:7])
    line2 = " ".join(words[7:14])
    line3 = " ".join(words[14:21])

    draw.text((card_x + 16, card_y + 12), f"SCENE BEAT {shot_number}", fill=(245, 158, 11), font=font_title)
    draw.text((card_x + 16, card_y + 36), line1, fill=(241, 245, 249), font=font_sub)
    if line2:
        draw.text((card_x + 16, card_y + 56), line2, fill=(203, 213, 225), font=font_sub)
    if line3:
        draw.text((card_x + 16, card_y + 76), line3, fill=(148, 163, 184), font=font_sub)

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

    print(f"Loading {_LOCAL_MODEL_ID} on {device} ({torch_dtype})...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        _LOCAL_MODEL_ID,
        torch_dtype=torch_dtype,
        variant="fp16" if device == "cuda" else None
    )
    pipe.to(device)

    # Optimization for fast low-memory inference
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
    api_url = f"https://api-inference.huggingface.co/models/{_LOCAL_MODEL_ID}"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "num_inference_steps": 2,
            "guidance_scale": 0.0,
            "seed": seed
        }
    }
    response = requests.post(api_url, headers=headers, json=payload, timeout=45)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        raise RuntimeError(f"HF API Error {response.status_code}: {response.text}")


def generate_single_frame(
    prompt: str,
    shot_number: int,
    seed: int = 42,
    mode: str = "auto",
    hf_token: Optional[str] = None
) -> Tuple[Image.Image, str]:
    """
    Generates a single frame using local diffusers, HF Inference API, or synthetic fallback.
    Returns stamped image and execution description.
    """
    token = hf_token or os.getenv("HF_TOKEN")
    env_mode = os.getenv("IMAGE_GEN_MODE", "auto").lower()
    selected_mode = mode if mode != "auto" else env_mode

    # Strategy 1: Explicit synthetic / mock mode
    if selected_mode == "mock":
        img = _create_synthetic_story_frame(prompt, shot_number, seed)
        stamped = stamp_ai_label(img, shot_number=shot_number)
        return stamped, "Generated via Synthetic Fallback Canvas"

    # Strategy 2: Local diffusers with sd-turbo
    if selected_mode in ("auto", "local_diffusers"):
        try:
            import torch
            pipe = _load_diffusers_pipeline()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(seed + shot_number)

            # sd-turbo requires only 1 to 2 steps with guidance_scale=0.0
            image = pipe(
                prompt=prompt,
                num_inference_steps=2,
                guidance_scale=0.0,
                generator=generator,
                width=512,
                height=512
            ).images[0]

            stamped = stamp_ai_label(image, shot_number=shot_number)
            return stamped, f"Generated via Local PyTorch sd-turbo ({device.upper()})"

        except Exception as e:
            print(f"Local diffusers generation failed or GPU memory exceeded ({e}). Attempting fallback...")

    # Strategy 3: HF Inference API
    if token and "your_huggingface" not in token:
        try:
            raw_img = _generate_with_hf_api(prompt, seed + shot_number, token)
            stamped = stamp_ai_label(raw_img, shot_number=shot_number)
            return stamped, "Generated via Hugging Face Inference API"
        except Exception as api_err:
            print(f"HF Inference API error ({api_err}). Falling back to synthetic renderer.")

    # Strategy 4: High-aesthetic offline synthetic renderer
    img = _create_synthetic_story_frame(prompt, shot_number, seed)
    stamped = stamp_ai_label(img, shot_number=shot_number)
    return stamped, "Generated via Aesthetic Canvas (CPU Fallback)"


def generate_storyboard_frames(
    shots: List[Dict[str, Any]],
    seed: int = 42,
    mode: str = "auto",
    hf_token: Optional[str] = None,
    progress_callback=None
) -> Tuple[List[Image.Image], List[str]]:
    """
    Generates all 6 storyboard frames sequentially with consistent seed.
    
    Args:
        shots: List of 6 shot dictionaries with 'image_prompt'.
        seed: Fixed random seed for reproducibility.
        mode: 'auto', 'local_diffusers', 'hf_api', or 'mock'.
        hf_token: Optional Hugging Face token.
        progress_callback: Optional callable for UI progress updates.
        
    Returns:
        Tuple of (list_of_images, list_of_methods)
    """
    frames = []
    methods = []
    total = len(shots)

    for i, shot in enumerate(shots):
        shot_num = shot.get("shot_number", i + 1)
        prompt = shot.get("image_prompt", "")

        if progress_callback:
            progress_callback((i + 1) / total, f"Generating frame {shot_num} of {total}...")

        img, method = generate_single_frame(
            prompt=prompt,
            shot_number=shot_num,
            seed=seed,
            mode=mode,
            hf_token=hf_token
        )
        frames.append(img)
        methods.append(method)

    return frames, methods
