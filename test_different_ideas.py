import os
import json
import time
import hashlib
from dotenv import load_dotenv

load_dotenv()

from llm import generate_storyboard
from imagegen import generate_storyboard_frames
from video import create_storyboard_media
from blip_check import evaluate_storyboard_frames

CONCEPTS = [
    "a small lonely robot in a greenhouse learns to paint flowers",
    "a dragon bakes bread in a snowy village",
    "a rowing team trains at sunrise on a calm lake"
]

all_hashes = []
results_summary = []

print("=" * 80)
print("TEST_DIFFERENT_IDEAS: VERIFYING MULTI-CONCEPT RUNS WITH IMG2IMG & ANCHORS")
print("=" * 80)

for idx, idea in enumerate(CONCEPTS, 1):
    print(f"\n[{idx}/3] CONCEPT: \"{idea}\"")
    t0 = time.time()
    
    # 1. LLM Storyboard with Anchors
    sb_data, note = generate_storyboard(idea=idea, technique="few-shot")
    t_llm = time.time() - t0
    
    title = sb_data.get("title", "")
    char_anchor = sb_data.get("character_anchor", "")
    set_anchor = sb_data.get("setting_anchor", "")
    shots = sb_data.get("shots", [])
    
    print(f" -> Title: {title}")
    print(f" -> Character Anchor: {char_anchor}")
    print(f" -> Setting Anchor:   {set_anchor}")
    
    # Anchor consistency check within this concept
    for s_idx, s in enumerate(shots, 1):
        prompt = s.get("image_prompt", "")
        assert char_anchor in prompt, f"Shot {s_idx} missing character_anchor!"
        assert set_anchor in prompt, f"Shot {s_idx} missing setting_anchor!"
        print(f"    Shot {s_idx} Prompt: {prompt[:110]}...")
    
    # 2. Frames generation (sd-turbo: T2I for Shot 1, I2I for Shots 2-6 with base seed 42)
    t1 = time.time()
    frames, engines = generate_storyboard_frames(shots=shots, seed=42, strength=0.60, mode="auto")
    t_frames = time.time() - t1
    print(f" -> Generated {len(frames)} frames ({engines[0]} -> {engines[-1]}) in {t_frames:.2f}s")
    
    # 3. Media export (Unique timestamped run directory)
    t2 = time.time()
    gif_path, mp4_path, music_path, run_dir = create_storyboard_media(frames=frames)
    t_video = time.time() - t2
    print(f" -> Saved to: {run_dir}")
    print(f"    GIF: {os.path.basename(gif_path)} ({os.path.getsize(gif_path)} bytes)")
    print(f"    MP4: {os.path.basename(mp4_path)} ({os.path.getsize(mp4_path)} bytes)")
    
    # Verify neither GIF nor MP4 is in sample_output
    assert "sample_output" not in run_dir.lower(), "Must NOT use sample_output directory!"
    assert os.path.exists(gif_path) and os.path.getsize(gif_path) > 0, "GIF must exist and be non-empty!"
    assert os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0, "MP4 must exist and be non-empty!"
    
    # 4. BLIP check
    t3 = time.time()
    eval_df, avg_score, details = evaluate_storyboard_frames(frames=frames, shots=shots)
    t_blip = time.time() - t3
    print(f" -> BLIP Semantic Alignment Score: {avg_score:.4f} in {t_blip:.2f}s")
    
    concept_hashes = [s.get("md5_hash") for s in shots]
    all_hashes.extend(concept_hashes)
    
    results_summary.append({
        "idea": idea,
        "title": title,
        "char_anchor": char_anchor,
        "set_anchor": set_anchor,
        "run_dir": run_dir,
        "avg_score": avg_score,
        "hashes": concept_hashes
    })

print("\n" + "=" * 80)
print("VERIFICATION CHECKS & PROOFS")
print("=" * 80)
for r in results_summary:
    print(f"\nConcept: {r['idea']}")
    print(f"  Title:            {r['title']}")
    print(f"  Character Anchor: {r['char_anchor']}")
    print(f"  Setting Anchor:   {r['set_anchor']}")
    print(f"  Run Directory:    {r['run_dir']}")
    print(f"  BLIP Score:       {r['avg_score']:.4f}")
    print(f"  Frame MD5 Hashes:")
    for i, h in enumerate(r['hashes'], 1):
        print(f"    Shot {i}: {h}")

# Strict Assertions
print("\n" + "-" * 80)
print("ASSERTIONS:")
print(f"Total frame hashes collected: {len(all_hashes)} (expected 18)")
assert len(all_hashes) == 18, "Expected exactly 18 frame hashes!"

unique_hashes = set(all_hashes)
print(f"Unique frame hashes across all runs: {len(unique_hashes)} / 18")
assert len(unique_hashes) == 18, f"All 18 generated frames must have distinct MD5 hashes! Found duplicate hashes."

# Assert unique run directories
run_dirs = set(r["run_dir"] for r in results_summary)
print(f"Unique run output directories: {len(run_dirs)} / 3")
assert len(run_dirs) == 3, "Each concept must output to a distinct run folder!"

print("\n[VERIFICATION PASSED] All 3 concepts generated completely unique, anchor-consistent frames!")
print("=" * 80)

