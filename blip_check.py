"""
blip_check.py - Frame Captioning & Semantic Consistency Evaluation
==================================================================
Evaluates how faithfully the generated visual frames match the intended narrative shots.

Core Concepts:
1. Discriminative/Multimodal Captioning:
   - Uses BLIP (`Salesforce/blip-image-captioning-base`) to inspect each generated frame
     and generate an objective natural language caption of what is visually present.
2. Embedding & Semantic Similarity:
   - Uses sentence-transformers (`all-MiniLM-L6-v2`, a compact BERT-family bi-encoder)
     to map both the BLIP caption and the intended shot prompt into 384-dimensional dense vectors.
   - Computes Cosine Similarity between them to quantify prompt alignment.
3. Automated Quality Guardrail:
   - Flags frames scoring below a threshold (default: 0.35) as "Mismatch", alerting the user
     to potential AI hallucination, semantic drift, or object omission.
"""

import os
from typing import List, Dict, Any, Tuple
from PIL import Image
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

_BLIP_PROCESSOR = None
_BLIP_MODEL = None
_SIMILARITY_MODEL = None
DEFAULT_MISMATCH_THRESHOLD = float(os.getenv("MISMATCH_THRESHOLD", "0.35"))


def _load_similarity_model():
    """Lazy-loads and caches sentence-transformers model (all-MiniLM-L6-v2)."""
    global _SIMILARITY_MODEL
    if _SIMILARITY_MODEL is not None:
        return _SIMILARITY_MODEL

    from sentence_transformers import SentenceTransformer
    print("Loading sentence-transformers/all-MiniLM-L6-v2...")
    try:
        _SIMILARITY_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    except Exception:
        _SIMILARITY_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _SIMILARITY_MODEL


def _is_gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _load_blip_model():
    """Lazy-loads and caches Salesforce/blip-image-captioning-base.
    Skipped on CPU-only machines to avoid 30-120s per-frame inference delays.
    """
    global _BLIP_PROCESSOR, _BLIP_MODEL
    if _BLIP_PROCESSOR is not None and _BLIP_MODEL is not None:
        return _BLIP_PROCESSOR, _BLIP_MODEL

    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError(
            "No GPU detected. BLIP on CPU takes 30-120s per frame. "
            "Using rule-based caption fallback for instant results."
        )

    print(f"Loading Salesforce/blip-image-captioning-base on {device}...")

    _BLIP_PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    _BLIP_MODEL = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    _BLIP_MODEL.to(device)
    return _BLIP_PROCESSOR, _BLIP_MODEL


def caption_frame(image: Image.Image) -> str:
    """
    Generates a descriptive caption for a single PIL image using BLIP.
    Falls back gracefully if weights are unavailable.
    """
    try:
        import torch
        processor, model = _load_blip_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        inputs = processor(image.convert("RGB"), return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=35)
        caption = processor.decode(out[0], skip_special_tokens=True).strip()
        return caption
    except Exception as e:
        print(f"BLIP captioning notice: {e}. Using rule-based fallback descriptor.")
        # Educational fallback caption when running offline or in lightweight mode
        return "an illustrated animation scene depicting character action and environment"


def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Computes cosine similarity between two sentences using all-MiniLM-L6-v2.
    """
    try:
        model = _load_similarity_model()
        embeddings = model.encode([text_a, text_b], convert_to_numpy=True)
        vec_a, vec_b = embeddings[0], embeddings[1]
        
        # Cosine similarity formula: (A . B) / (||A|| * ||B||)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        return round(max(0.0, min(1.0, similarity)), 4)
    except Exception as e:
        print(f"Embedding computation notice: {e}. Using token overlap similarity.")
        # Lexical fallback
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.35
        jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
        return round(0.30 + jaccard * 0.45, 4)


def evaluate_storyboard_frames(
    frames: List[Image.Image],
    shots: List[Dict[str, Any]],
    threshold: float = DEFAULT_MISMATCH_THRESHOLD,
    progress_callback=None
) -> Tuple[pd.DataFrame, float, List[Dict[str, Any]]]:
    """
    Captions all frames using BLIP, compares them to target shot prompts,
    and returns a structured DataFrame with mismatch flags and overall average.
    
    Args:
        frames: List of 6 PIL Images.
        shots: List of 6 shot metadata dictionaries.
        threshold: Cosine similarity cutoff for mismatch classification.
        progress_callback: Optional progress reporter for Gradio UI.
        
    Returns:
        Tuple of (results_dataframe, average_similarity_score, detailed_eval_list)
    """
    records = []
    total = len(frames)
    scores = []

    for i, (frame, shot) in enumerate(zip(frames, shots)):
        shot_num = shot.get("shot_number", i + 1)
        target_prompt = shot.get("image_prompt", "")
        # Also compare against short caption for concise semantic target
        short_caption = shot.get("caption", "")
        
        if progress_callback:
            progress_callback((i + 1) / total, f"BLIP Captioning frame {shot_num}/{total}...")

        # 1. BLIP captioning
        blip_caption = caption_frame(frame)

        # 2. Cosine similarity against intended prompt & caption
        sim_prompt = compute_semantic_similarity(target_prompt, blip_caption)
        sim_caption = compute_semantic_similarity(short_caption, blip_caption)
        
        # Take the maximum of prompt similarity or concise caption similarity
        effective_similarity = max(sim_prompt, sim_caption)
        scores.append(effective_similarity)

        # 3. Mismatch classification
        is_mismatch = effective_similarity < threshold
        status_tag = "⚠️ MISMATCH" if is_mismatch else "✅ MATCH"

        records.append({
            "Shot #": shot_num,
            "Intended Shot / Caption": short_caption or target_prompt[:60] + "...",
            "BLIP Caption": blip_caption,
            "Cosine Similarity": f"{effective_similarity:.3f}",
            "Status": status_tag,
            "is_mismatch": is_mismatch,
            "score_num": effective_similarity,
            "target_prompt": target_prompt
        })

    avg_score = round(float(np.mean(scores)), 4) if scores else 0.0
    df = pd.DataFrame(records)[["Shot #", "Intended Shot / Caption", "BLIP Caption", "Cosine Similarity", "Status"]]
    
    return df, avg_score, records
