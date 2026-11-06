"""
imagegen.py - Frame Generation with Diffusers (sd-turbo) Img2Img Consistency
=============================================================================
Generates 6 distinct, visually coherent narrative frames:
1. Shot 1: Generated via Text-to-Image (stabilityai/sd-turbo) with user base seed.
2. Shots 2 to 6: Generated via Image-to-Image (AutoPipelineForImage2Image) from the
   PREVIOUS frame with configurable strength (0.55-0.65) to preserve character
   identity and setting while evolving the narrative action.
3. Same base seed is used across shots for continuity.
4. AI-Generated Disclosure Label is stamped onto every exported frame.
5. Exact prompts and MD5 hashes are logged in terminal and returned for UI inspection.
6. NO silent fallback: failures raise visible errors in the application.
"""

import os
import io
import time
import hashlib
from typing import List, Dict, Any, Tuple, Optional, Callable
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from dotenv import load_dotenv

load_dotenv()

# Global pipeline caches
_LOCAL_DIFFUSER_T2I = None
_LOCAL_DIFFUSER_I2I = None
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
    width, height = img.size

    try:
        font_large = ImageFont.truetype("arial.ttf", max(14, int(height * 0.035)))
        font_small = ImageFont.truetype("arial.ttf", max(11, int(height * 0.026)))
    except Exception:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    banner_height = int(height * 0.08)
    banner_y0 = height - banner_height

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([(0, banner_y0), (width, height)], fill=(15, 23, 42, 195))

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


def _load_diffusers_pipelines():
    """Lazy-loads and caches local stabilityai/sd-turbo Text2Image & Image2Image pipelines."""
    global _LOCAL_DIFFUSER_T2I, _LOCAL_DIFFUSER_I2I
    if _LOCAL_DIFFUSER_T2I is not None and _LOCAL_DIFFUSER_I2I is not None:
        return _LOCAL_DIFFUSER_T2I, _LOCAL_DIFFUSER_I2I

    import torch
    from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[DIFFUSERS] Loading {_LOCAL_MODEL_ID} on {device.upper()} ({torch_dtype})...")
    pipe_t2i = AutoPipelineForText2Image.from_pretrained(
        _LOCAL_MODEL_ID,
        torch_dtype=torch_dtype,
        variant="fp16" if device == "cuda" else None
    )
    pipe_t2i.to(device)

    # Share model components directly between T2I and I2I
    pipe_i2i = AutoPipelineForImage2Image.from_pipe(pipe_t2i)
    pipe_i2i.to(device)

    if device == "cuda":
        try:
            pipe_t2i.enable_attention_slicing()
            pipe_i2i.enable_attention_slicing()
        except Exception:
            pass

    _LOCAL_DIFFUSER_T2I = pipe_t2i
    _LOCAL_DIFFUSER_I2I = pipe_i2i
    return _LOCAL_DIFFUSER_T2I, _LOCAL_DIFFUSER_I2I


def _generate_with_hf_api(prompt: str, seed: int, token: str) -> Image.Image:
    """Uses Hugging Face Inference API when selected."""
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
        raise RuntimeError(f"HF Router returned {response.status_code}: {response.text[:140]}")


def _create_synthetic_story_frame(
    prompt: str,
    shot_number: int,
    seed: int,
    width: int = 512,
    height: int = 512
) -> Image.Image:
    """Dynamic concept-themed canvas for explicit mock testing."""
    import random
    effective_seed = seed + shot_number * 1000
    rng = random.Random(effective_seed)

    prompt_lower = prompt.lower()
    if any(k in prompt_lower for k in ["dragon", "snow", "village", "bread", "bakery"]):
        palettes = [
            ((30, 20, 15), (180, 80, 20), (255, 230, 180)),
            ((40, 25, 20), (200, 100, 30), (255, 240, 200)),
            ((25, 30, 45), (120, 70, 30), (240, 245, 255)),
            ((50, 20, 10), (230, 110, 20), (255, 220, 160)),
            ((60, 30, 10), (240, 140, 30), (255, 245, 210)),
            ((20, 30, 40), (180, 120, 50), (250, 240, 220)),
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

    base = Image.new("RGB", (width, height), c1)
    draw = ImageDraw.Draw(base)
    for y in range(height):
        ratio = y / height
        r = int(c1[0] * (1 - ratio) + c2[0] * ratio)
        g = int(c1[1] * (1 - ratio) + c2[1] * ratio)
        b = int(c1[2] * (1 - ratio) + c2[2] * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    orb_x = width // 2 + int(rng.uniform(-60, 60))
    orb_y = int(height * 0.30) + int(rng.uniform(-20, 20))
    orb_rad = int(width * 0.15)
    draw.ellipse(
        [(orb_x - orb_rad, orb_y - orb_rad), (orb_x + orb_rad, orb_y + orb_rad)],
        fill=c3,
        outline=(255, 255, 255, 160)
    )

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
    return base


def generate_storyboard_frames(
    shots: List[Dict[str, Any]],
    seed: int = 42,
    mode: str = "auto",
    strength: float = 0.60,
    hf_token: Optional[str] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[List[Image.Image], List[str]]:
    """
    Generates all 6 storyboard frames sequentially.
    Requirement 5:
    - Base seed is constant across all 6 shots.
    - Shot 1: Generated via Text-to-Image.
    - Shots 2-6: Generated via Image-to-Image from the PREVIOUS frame with strength (0.55-0.65).
    - Logs prompt and MD5 hash of each frame in the terminal.
    - NO silent fallback: failures raise explicit errors.
    """
    token = hf_token or os.getenv("HF_TOKEN")
    env_mode = os.getenv("IMAGE_GEN_MODE", "auto").lower()
    selected_mode = mode if mode != "auto" else env_mode

    frames = []
    methods = []
    total = len(shots)
    prev_raw_frame = None

    print(f"\n{'='*75}")
    print(f"[IMAGE GEN] Starting generation of {total} frames")
    print(f"  Mode: {selected_mode} | Base Seed: {seed} | Img2Img Strength: {strength}")
    print(f"{'='*75}")

    # Mode 1: Explicit Mock Canvas (only when user explicitly requests it)
    if selected_mode == "mock":
        for i, shot in enumerate(shots):
            shot_num = shot.get("shot_number", i + 1)
            prompt = shot.get("image_prompt", "")
            if progress_callback:
                progress_callback((i + 1) / total, f"Drawing mock frame {shot_num}/{total}...")
            raw_img = _create_synthetic_story_frame(prompt, shot_num, int(seed))
            stamped = stamp_ai_label(raw_img, shot_number=shot_num)
            md5_hash = hashlib.md5(raw_img.tobytes()).hexdigest()
            shot["md5_hash"] = md5_hash
            frames.append(stamped)
            methods.append("Mock Canvas")
            print(f"[FRAME {shot_num}] MD5: {md5_hash} | Engine: Mock Canvas")
            print(f"   Prompt: {prompt}\n")
        return frames, methods

    # Mode 2: HF API (if explicitly requested or local unavailable)
    if selected_mode == "hf_api":
        if not token or "your_huggingface" in token:
            raise RuntimeError("HF_TOKEN missing or invalid. Set HF_TOKEN to use Hugging Face Inference API.")
        for i, shot in enumerate(shots):
            shot_num = shot.get("shot_number", i + 1)
            prompt = shot.get("image_prompt", "")
            if progress_callback:
                progress_callback((i + 1) / total, f"Querying HF API for frame {shot_num}/{total}...")
            raw_img = _generate_with_hf_api(prompt, int(seed), token)
            stamped = stamp_ai_label(raw_img, shot_number=shot_num)
            md5_hash = hashlib.md5(raw_img.tobytes()).hexdigest()
            shot["md5_hash"] = md5_hash
            frames.append(stamped)
            methods.append("Hugging Face Inference API")
            print(f"[FRAME {shot_num}] MD5: {md5_hash} | Engine: Hugging Face Inference API")
            print(f"   Prompt: {prompt}\n")
        return frames, methods

    # Mode 3: Local Diffusers with sd-turbo (T2I for shot 1, I2I for shots 2-6)
    try:
        import torch
        t2i_pipe, i2i_pipe = _load_diffusers_pipelines()
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception as e:
        raise RuntimeError(f"Failed to initialize local sd-turbo diffusers: {e}. No silent fallback used.")

    for i, shot in enumerate(shots):
        shot_num = shot.get("shot_number", i + 1)
        prompt = shot.get("image_prompt", "")

        t0 = time.time()
        # Same base seed for all shots
        gen = torch.Generator(device=device).manual_seed(int(seed))

        if i == 0 or prev_raw_frame is None:
            # Shot 1: Text-to-Image
            engine_desc = f"Local sd-turbo Text2Image ({device.upper()})"
            if progress_callback:
                progress_callback((i + 0.1) / total, f"Rendering Shot 1 Text-to-Image (Seed {seed})...")
            try:
                raw_frame = t2i_pipe(
                    prompt=prompt,
                    num_inference_steps=1,
                    guidance_scale=0.0,
                    generator=gen,
                    width=512,
                    height=512
                ).images[0]
            except Exception as gen_err:
                raise RuntimeError(f"Local sd-turbo failed on Shot 1: {gen_err}")
        else:
            # Shots 2 to 6: Image-to-Image from previous frame
            engine_desc = f"Local sd-turbo Img2Img ({device.upper()}, str={strength:.2f})"
            if progress_callback:
                progress_callback((i + 0.1) / total, f"Rendering Shot {shot_num} Img2Img from Shot {shot_num-1}...")
            try:
                # With sd-turbo, 2 steps with strength 0.6 runs 1 fast step
                raw_frame = i2i_pipe(
                    prompt=prompt,
                    image=prev_raw_frame,
                    strength=float(strength),
                    num_inference_steps=2,
                    guidance_scale=0.0,
                    generator=gen
                ).images[0]
            except Exception as gen_err:
                raise RuntimeError(f"Local sd-turbo failed on Shot {shot_num}: {gen_err}")

        # Save raw frame before watermark to preserve visual purity for next img2img step
        prev_raw_frame = raw_frame

        elapsed = time.time() - t0
        md5_hash = hashlib.md5(raw_frame.tobytes()).hexdigest()
        shot["md5_hash"] = md5_hash

        # Stamp Responsible AI label
        stamped_frame = stamp_ai_label(raw_frame, shot_number=shot_num)
        frames.append(stamped_frame)
        methods.append(engine_desc)

        print(f"[FRAME {shot_num}/6] MD5: {md5_hash} | Engine: {engine_desc} | Time: {elapsed:.2f}s")
        print(f"   Prompt: {prompt}\n")

    return frames, methods
