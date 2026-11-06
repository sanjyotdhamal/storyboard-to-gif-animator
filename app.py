"""
app.py - Storyboard-to-GIF Animator — Clean Creator Edition (Light Theme)
========================================================================
Redesigned UI with Antigravity-inspired "Clean Creator" light aesthetic.
Backend files (safety.py, llm.py, imagegen.py, blip_check.py, video.py) are UNCHANGED.
All features preserved: idea input, few-shot/zero-shot, seed, advanced settings,
generate button, gallery, MP4 + GIF, music, BLIP table, JSON viewer, all tabs.
"""

import os
import math
import time
import gradio as gr
from dotenv import load_dotenv

from safety import check_safety
from llm import generate_storyboard, compare_prompts, DEFAULT_STYLE_PREFIX
from imagegen import generate_storyboard_frames
from blip_check import evaluate_storyboard_frames, DEFAULT_MISMATCH_THRESHOLD
from video import create_storyboard_media, get_background_music_path

load_dotenv()

# Pre-synthesize default ambient audio loop on startup
try:
    get_background_music_path()
except Exception as e:
    print(f"Audio init warning: {e}")


# ---------------------------------------------------------------------------
# Pipeline step definitions & preset ideas
# ---------------------------------------------------------------------------

PIPELINE_STEPS = [
    ("🛡️", "Safety"),
    ("✍️", "Storyboard"),
    ("🖼️", "Frames"),
    ("🎬", "Video"),
    ("🔍", "Quality"),
]

PRESET_IDEAS = [
    ("🤖 Robot paints flowers",    "A small lonely robot in a greenhouse learns to paint flowers"),
    ("🚢 Paper boat to the sea",   "A paper boat sails through a stormy city street and reaches the sea"),
    ("🐱 Astronaut cat on moon",   "A tiny astronaut cat plants a flag on a candy moon"),
    ("🚣 Rowing team at sunrise",  "A rowing team trains at sunrise on a calm lake"),
    ("🌳 Seed grows through seasons", "A seed grows into a giant tree as the seasons change"),
    ("🐳 Lighthouse meets whale",  "A lighthouse keeper discovers a glowing whale at night"),
]


# ---------------------------------------------------------------------------
# UI helper renderers
# ---------------------------------------------------------------------------

def make_pipeline_html(active_step: int = -1, done_steps: set = None) -> str:
    """Renders the 5-step pipeline progress strip with high-contrast text."""
    if done_steps is None:
        done_steps = set()
    pills = []
    for i, (icon, label) in enumerate(PIPELINE_STEPS):
        if i in done_steps:
            cls, suffix = "done", " ✓"
        elif i == active_step:
            cls, suffix = "active", ""
        else:
            cls, suffix = "idle", ""
        pills.append(
            f'<div class="cc-pipe-step {cls}">'
            f'{icon} <span class="cc-pipe-label">{label}</span>{suffix}</div>'
        )
        if i < len(PIPELINE_STEPS) - 1:
            pills.append('<span class="cc-pipe-arrow">›</span>')
    return f'<div class="cc-pipeline">{"".join(pills)}</div>'


def make_blip_ring_html(score=None) -> str:
    """Renders an SVG circular progress ring for the BLIP alignment score."""
    radius = 52
    circumference = 2 * math.pi * radius

    if score is None:
        pct, score_str, ring_color, label_color = 0.0, "—", "#E6E8EC", "#6B7280"
    else:
        pct = min(1.0, max(0.0, float(score)))
        score_str = f"{pct:.3f}"
        if pct >= 0.50:
            ring_color, label_color = "#10B981", "#047857"
        elif pct >= 0.35:
            ring_color, label_color = "#F59E0B", "#B45309"
        else:
            ring_color, label_color = "#EF4444", "#B91C1C"

    dash_offset = circumference * (1 - pct)

    return f"""
<div style="display:flex;flex-direction:column;align-items:center;padding:16px 0 8px;">
  <div style="position:relative;width:130px;height:130px;">
    <svg width="130" height="130" viewBox="0 0 130 130"
         style="transform:rotate(-90deg);display:block;">
      <circle cx="65" cy="65" r="{radius}" fill="none"
              stroke="#E6E8EC" stroke-width="10"/>
      <circle cx="65" cy="65" r="{radius}" fill="none"
              stroke="{ring_color}" stroke-width="10" stroke-linecap="round"
              stroke-dasharray="{circumference:.2f}"
              stroke-dashoffset="{dash_offset:.2f}"
              style="transition:stroke-dashoffset 0.9s cubic-bezier(0.22,1,0.36,1);"/>
    </svg>
    <div style="position:absolute;inset:0;display:flex;flex-direction:column;
                align-items:center;justify-content:center;gap:2px;">
      <span style="font-size:1.85rem;font-weight:700;color:#0B0B0F;
                   letter-spacing:-0.04em;line-height:1;">{score_str}</span>
      <span style="font-size:0.75rem;font-weight:700;text-transform:uppercase;
                   letter-spacing:0.08em;color:{label_color};">BLIP</span>
    </div>
  </div>
  <p style="font-size:0.82rem;color:#3C4043;font-weight:500;text-align:center;margin-top:10px;
             line-height:1.45;max-width:180px;">
    Average semantic alignment (Cosine Similarity)
  </p>
</div>
"""


# ---------------------------------------------------------------------------
# Pipeline Generator (real-time progress updates)
# ---------------------------------------------------------------------------

def _run_pipeline_generator(
    idea: str,
    technique: str,
    seed: int,
    style_prefix: str,
    mismatch_threshold: float,
    image_mode: str,
    api_key_override: str,
    hf_token_override: str,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generator yielding UI updates after each pipeline stage.
    Outputs (9): pipeline_html, status_md, json_viewer, gallery,
                 eval_table, blip_ring_html, gif_image, video_player, audio_player
    """
    # ── Step 0: Safety ──────────────────────────────────────────────────────
    t_start = time.time()
    yield (make_pipeline_html(0), "🛡️ **Checking content safety guardrails...**",
           None, None, None, make_blip_ring_html(), None, None, None)

    t0 = time.time()
    is_safe, safety_msg, _ = check_safety(idea)
    t_safety = time.time() - t0
    if not is_safe:
        yield (make_pipeline_html(),
               f"🚫 **Safety block:** {safety_msg}",
               None, None, None, make_blip_ring_html(), None, None, None)
        return

    # ── Step 1: LLM Storyboard ───────────────────────────────────────────
    yield (make_pipeline_html(1, {0}),
           f"✍️ **Generating `{technique}` storyboard with Gemini...**",
           None, None, None, make_blip_ring_html(), None, None, None)

    t0 = time.time()
    clean_prefix = style_prefix.strip() or DEFAULT_STYLE_PREFIX
    storyboard_data, llm_note = generate_storyboard(
        idea=idea, technique=technique, style_prefix=clean_prefix,
        api_key=api_key_override.strip() if api_key_override else None,
    )
    t_llm = time.time() - t0
    shots = storyboard_data.get("shots", [])

    is_fallback = storyboard_data.get("is_fallback", False)
    fallback_reason = storyboard_data.get("fallback_reason", "")
    llm_model = storyboard_data.get("model", "gemini-3-flash-preview")

    # ── Step 2: Frame Generation ─────────────────────────────────────────
    yield (make_pipeline_html(2, {0, 1}),
           f"🖼️ **Generating {len(shots)} visual frames (seed={seed})...**",
           storyboard_data, None, None, make_blip_ring_html(), None, None, None)

    t0 = time.time()
    frames, gen_methods = generate_storyboard_frames(
        shots=shots, seed=int(seed), mode=image_mode,
        hf_token=hf_token_override.strip() if hf_token_override else None,
        progress_callback=lambda p, msg: progress(0.35 + p * 0.30, desc=msg),
    )
    t_frames = time.time() - t0

    # Show frames immediately while video renders
    yield (make_pipeline_html(3, {0, 1, 2}),
           "🎬 **Rendering cross-fade transitions and merging music...**",
           storyboard_data, frames, None, make_blip_ring_html(), None, None, None)

    # ── Step 3: Video ────────────────────────────────────────────────────
    t0 = time.time()
    gif_path, mp4_path, music_path, run_dir = create_storyboard_media(
        frames=frames, fps_video=24, fps_gif=15,
        hold_frames=12, transition_frames=8,
        progress_callback=lambda p, msg: progress(0.65 + p * 0.20, desc=msg),
    )
    t_video = time.time() - t0

    # ── Step 4: BLIP Quality Check ───────────────────────────────────────
    yield (make_pipeline_html(4, {0, 1, 2, 3}),
           "🔍 **Running BLIP multimodal quality verification...**",
           storyboard_data, frames, None, make_blip_ring_html(),
           gif_path, mp4_path, music_path)

    t0 = time.time()
    eval_df, avg_score, eval_details = evaluate_storyboard_frames(
        frames=frames, shots=shots, threshold=float(mismatch_threshold),
        progress_callback=lambda p, msg: progress(0.85 + p * 0.15, desc=msg),
    )
    t_quality = time.time() - t0
    t_total = time.time() - t_start

    mismatch_count = sum(1 for d in eval_details if d.get("is_mismatch"))
    engine = gen_methods[0] if gen_methods else "Local sd-turbo"

    if is_fallback:
        warning_banner = (
            f'<div style="background-color:#FEE2E2;border:2px solid #EF4444;color:#991B1B;'
            f'padding:14px 18px;border-radius:14px;font-weight:600;margin-bottom:16px;'
            f'box-shadow:0 2px 8px rgba(239,68,68,0.12);">'
            f'<div style="font-size:1.05rem;display:flex;align-items:center;gap:8px;">'
            f'⚠️ <span>Gemini failed, using offline fallback. Results are NOT based on your idea</span></div>'
            f'<div style="font-size:0.85rem;font-weight:400;margin-top:6px;color:#7F1D1D;">'
            f'<strong>Error details:</strong> {fallback_reason}</div>'
            f'</div>'
        )
    else:
        warning_banner = (
            f'<div style="background-color:#ECFDF5;border:1.5px solid #10B981;color:#065F46;'
            f'padding:10px 16px;border-radius:12px;font-size:0.9rem;font-weight:600;margin-bottom:14px;'
            f'display:flex;align-items:center;gap:8px;">'
            f'✓ <span>Live Gemini LLM (<code>{llm_model}</code>) &amp; {engine} active</span>'
            f'</div>'
        )

    summary = (
        f"{warning_banner}\n\n"
        f"### 🎉 Storyboard Generated Successfully!\n"
        f"- **LLM Storyboard**: `{llm_model}` ({t_llm:.2f}s) &nbsp;·&nbsp; **Technique**: `{technique}`\n"
        f"- **Image Engine**: `{engine}` ({t_frames:.2f}s) &nbsp;·&nbsp; **Seed**: `{seed}` (Frames seeded `{seed}+1`–`{seed}+6`)\n"
        f"- **Video & Audio**: Cross-fade Transitions + Synthesized Audio ({t_video:.2f}s)\n"
        f"- **BLIP Quality**: Avg `{avg_score:.3f}` &nbsp;·&nbsp; `{mismatch_count}` potential mismatch flag(s) ({t_quality:.2f}s)\n"
        f"- **Run Artifacts**: Saved to `{run_dir}` &nbsp;·&nbsp; **Total Pipeline Time**: `{t_total:.2f}s`\n"
        f"- **AI Disclosure**: Verified 'AI-Generated' watermark stamped onto all frames."
    )

    yield (
        make_pipeline_html(-1, {0, 1, 2, 3, 4}),
        summary,
        storyboard_data,
        frames,
        eval_df,
        make_blip_ring_html(avg_score),
        gif_path,
        mp4_path,
        music_path,
    )


def run_technique_comparison(idea: str, style_prefix: str, api_key_override: str):
    """Executes both Zero-Shot and Few-Shot prompting side by side."""
    is_safe, safety_msg, _ = check_safety(idea)
    if not is_safe:
        return safety_msg, None, None
    clean_prefix = style_prefix.strip() or DEFAULT_STYLE_PREFIX
    zero_data, few_data, summary = compare_prompts(
        idea=idea, style_prefix=clean_prefix,
        api_key=api_key_override.strip() if api_key_override else None,
    )
    return summary, zero_data, few_data


# ---------------------------------------------------------------------------
# CSS — Clean Creator Light Theme with complete Gradio variable overrides
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* ── Fonts ───────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

/* ── Force Light Theme and Override Gradio Variables ──── */
:root, html, body, .gradio-container, .dark, .light {
  color-scheme: light !important;
  --body-background-fill: #FFFFFF !important;
  --background-fill-primary: #FFFFFF !important;
  --background-fill-secondary: #F8F9FA !important;
  --block-background-fill: #FFFFFF !important;
  --panel-background-fill: #FFFFFF !important;
  --input-background-fill: #FFFFFF !important;
  --button-secondary-background-fill: #F8F9FA !important;
  --body-text-color: #0B0B0F !important;
  --block-label-text-color: #0B0B0F !important;
  --block-title-text-color: #0B0B0F !important;
  --input-text-color: #0B0B0F !important;
  --border-color-primary: #E6E8EC !important;
  --border-color-secondary: #E6E8EC !important;
  --block-border-color: #E6E8EC !important;
  --table-border-color: #E6E8EC !important;
  --button-secondary-text-color: #0B0B0F !important;
  --button-secondary-border-color: #E6E8EC !important;

  /* Custom palette */
  --white:   #FFFFFF;
  --bg:      #F8F9FA;
  --black:   #0B0B0F;
  --text:    #0B0B0F;
  --muted:   #4B5563;
  --border:  #E6E8EC;
  --radius-card: 24px;
  --radius-pill: 999px;
  --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
  --shadow-md: 0 4px 20px rgba(0,0,0,0.08);

  /* Soft pastels */
  --lavender:      #E8E3FF;
  --lavender-mid:  #C4B5FD;
  --mint:          #DDF5EA;
  --mint-mid:      #86EFAC;
  --peach:         #FFE8DB;
  --sky:           #DCEBFF;
  --butter:        #FFF3CC;
}

/* Hard reset for dark mode overrides */
.dark, .dark * {
  --body-background-fill: #FFFFFF !important;
  --background-fill-primary: #FFFFFF !important;
  --background-fill-secondary: #F8F9FA !important;
  --block-background-fill: #FFFFFF !important;
  --panel-background-fill: #FFFFFF !important;
  --input-background-fill: #FFFFFF !important;
  --button-secondary-background-fill: #F8F9FA !important;
  --body-text-color: #0B0B0F !important;
  --block-label-text-color: #0B0B0F !important;
  --block-title-text-color: #0B0B0F !important;
  --input-text-color: #0B0B0F !important;
  --border-color-primary: #E6E8EC !important;
}

body, .gradio-container {
  max-width: 100% !important;
  padding: 0 !important;
  margin: 0 !important;
  background-color: #FFFFFF !important;
  font-family: 'DM Sans', Inter, -apple-system, BlinkMacSystemFont, sans-serif !important;
  font-size: 16px !important;
  color: #0B0B0F !important;
}

footer.svelte-1rjryqp { display: none !important; }

/* ── Top Bar (Logo & Project Name only) ───────────────── */
.cc-nav {
  position: fixed;
  top: 0; left: 0; right: 0;
  z-index: 999;
  height: 60px;
  display: flex; align-items: center; justify-content: flex-start;
  padding: 0 36px;
  background: rgba(255,255,255,0.92);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border-bottom: 1px solid var(--border);
}
.cc-nav-brand {
  font-size: 1.05rem; font-weight: 700; color: #0B0B0F !important;
  letter-spacing: -0.02em; display: flex; align-items: center; gap: 10px;
  text-decoration: none;
}
.cc-nav-brand span { font-size: 1.35rem; }
.cc-nav-spacer { height: 60px; }

/* ── Hero ────────────────────────────────────────────── */
.cc-hero {
  position: relative; overflow: hidden;
  padding: 64px 32px 48px; text-align: center;
  background: var(--white);
}
.cc-hero-inner { position: relative; z-index: 1; }
.cc-hero-eyebrow {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--lavender); color: #3B1B7D !important;
  border-radius: var(--radius-pill); padding: 5px 14px;
  font-size: 0.8rem; font-weight: 700; letter-spacing: 0.06em;
  text-transform: uppercase; margin-bottom: 18px;
}
.cc-hero-h1 {
  font-size: clamp(2.4rem, 5.5vw, 3.75rem);
  font-weight: 700; color: #0B0B0F !important;
  letter-spacing: -0.035em; line-height: 1.08;
  margin-bottom: 12px;
}
.cc-hero-h1 em { color: #5F6368 !important; font-style: normal; }
.cc-hero-sub {
  font-size: 1.1rem; color: #3C4043 !important; line-height: 1.6;
  max-width: 580px; margin: 0 auto 28px; font-weight: 400;
}

/* ── Floating blobs (subtle & light) ──────────────────── */
.cc-blobs { position: absolute; inset: 0; pointer-events: none; overflow: hidden; }
.cc-blob { position: absolute; border-radius: 50%; filter: blur(72px); opacity: 0.45; }
.cc-blob-1 { width:380px; height:380px; background:var(--lavender); top:-120px; left:-80px; animation: blobA 26s ease-in-out infinite; }
.cc-blob-2 { width:300px; height:300px; background:var(--mint); top:-80px; right:-60px; animation: blobB 22s ease-in-out infinite; }
.cc-blob-3 { width:260px; height:260px; background:var(--peach); bottom:-100px; left:35%; animation: blobC 30s ease-in-out infinite; }
@keyframes blobA { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(28px,18px) scale(1.06)} }
@keyframes blobB { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(-22px,14px) scale(1.07)} }
@keyframes blobC { 0%,100%{transform:translate(0,0) scale(1)} 50%{transform:translate(18px,-14px) scale(1.04)} }

/* ── Idea input card ─────────────────────────────────── */
.cc-idea-card {
  max-width: 760px; margin: 0 auto; position: relative; z-index: 1;
}
.cc-idea-card textarea {
  font-size: 1.05rem !important;
  font-family: 'DM Sans', sans-serif !important;
  border-radius: 18px !important;
  border: 1.5px solid var(--border) !important;
  padding: 18px 22px !important;
  background: #FFFFFF !important;
  box-shadow: var(--shadow-sm) !important;
  color: #0B0B0F !important;
  line-height: 1.6 !important;
}
.cc-idea-card textarea:focus {
  border-color: #0B0B0F !important;
  box-shadow: 0 0 0 3px rgba(11,11,15,0.08) !important;
  outline: none !important;
}
.cc-idea-card label, .cc-idea-card span {
  color: #0B0B0F !important; font-weight: 600 !important; opacity: 1 !important;
}

/* ── Preset chips ────────────────────────────────────── */
.cc-chips {
  display: flex; flex-wrap: wrap; justify-content: center;
  gap: 8px; margin: 12px auto 0; max-width: 760px;
  position: relative; z-index: 1;
}
.cc-chip button {
  background: #FFFFFF !important;
  color: #1A1A2E !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-pill) !important;
  padding: 7px 16px !important;
  font-size: 0.85rem !important; font-weight: 600 !important;
  font-family: 'DM Sans', sans-serif !important;
  cursor: pointer !important; white-space: nowrap !important;
  box-shadow: var(--shadow-sm) !important;
  opacity: 1 !important;
  transition: all 0.18s ease !important;
}
.cc-chip button:hover {
  background: var(--lavender) !important;
  border-color: var(--lavender-mid) !important;
  color: #0B0B0F !important;
  transform: translateY(-2px) !important;
}

/* ── Advanced Settings bar ───────────────────────────── */
.cc-advanced-accordion {
  max-width: 760px; margin: 14px auto 0 !important;
  background: #FFFFFF !important;
  border: 1px solid var(--border) !important;
  border-radius: 18px !important;
  overflow: hidden !important;
}
.cc-advanced-accordion details {
  border: none !important;
  background: #FFFFFF !important;
}
.cc-advanced-accordion summary {
  padding: 12px 20px !important;
  background: #F8F9FA !important;
  color: #0B0B0F !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  opacity: 1 !important;
}
.cc-advanced-accordion label,
.cc-advanced-accordion span {
  color: #0B0B0F !important;
  font-weight: 600 !important;
  opacity: 1 !important;
}
.cc-advanced-accordion input,
.cc-advanced-accordion select,
.cc-advanced-accordion .gr-input {
  background: #FFFFFF !important;
  color: #0B0B0F !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}

/* ── Generate button (Black Pill) ────────────────────── */
.cc-gen-btn { max-width: 760px; margin: 16px auto 0; position: relative; z-index: 1; }
.cc-gen-btn button {
  background: #0B0B0F !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: var(--radius-pill) !important;
  padding: 15px 36px !important;
  font-size: 1.05rem !important; font-weight: 600 !important;
  font-family: 'DM Sans', sans-serif !important;
  width: 100% !important; cursor: pointer !important;
  box-shadow: 0 4px 16px rgba(11,11,15,0.22) !important;
  transition: transform 0.18s, box-shadow 0.18s !important;
  letter-spacing: -0.01em !important;
  opacity: 1 !important;
}
.cc-gen-btn button:hover {
  transform: scale(1.02) !important;
  box-shadow: 0 6px 24px rgba(11,11,15,0.30) !important;
}

/* ── Section wrapper ─────────────────────────────────── */
.cc-section {
  max-width: 1200px; margin: 0 auto; padding: 0 32px;
}
.cc-section-gap { height: 40px; }

/* ── Tabs (Main Navigation) ──────────────────────────── */
.gradio-container .tabs > .tab-nav {
  background: #FFFFFF !important;
  border-bottom: 1px solid var(--border) !important;
  padding: 0 32px !important;
}
.gradio-container .tabs > .tab-nav button {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.95rem !important; font-weight: 600 !important;
  color: #3C4043 !important;
  opacity: 1 !important;
  padding: 14px 22px !important; border-radius: 0 !important;
  border-bottom: 2px solid transparent !important;
  background: transparent !important;
  transition: color 0.18s, border-color 0.18s !important;
}
.gradio-container .tabs > .tab-nav button:hover {
  color: #0B0B0F !important;
}
.gradio-container .tabs > .tab-nav button.selected,
.gradio-container .tabs > .tab-nav button[aria-selected="true"] {
  color: #0B0B0F !important;
  border-bottom: 2px solid #0B0B0F !important;
  font-weight: 700 !important;
}

/* ── Pipeline progress strip ─────────────────────────── */
.cc-pipeline {
  display: flex; align-items: center; flex-wrap: wrap;
  gap: 6px; padding: 18px 0 10px;
}
.cc-pipe-step {
  display: flex; align-items: center; gap: 6px;
  padding: 8px 18px; border-radius: var(--radius-pill);
  font-size: 0.85rem; font-weight: 600; white-space: nowrap;
  border: 1.5px solid transparent;
  opacity: 1 !important;
  transition: all 0.25s ease;
}
.cc-pipe-step.idle   { background: #F8F9FA !important; color: #3C4043 !important; border-color: var(--border) !important; }
.cc-pipe-step.active { background: var(--lavender) !important; color: #0B0B0F !important; border-color: #A78BFA !important; font-weight: 700 !important; }
.cc-pipe-step.done   { background: var(--mint) !important; color: #064E3B !important; border-color: #86EFAC !important; font-weight: 700 !important; }
.cc-pipe-arrow       { color: #6B7280 !important; font-size: 1.2rem; flex-shrink: 0; font-weight: 700; opacity: 1 !important; }
.cc-pipe-label       { color: inherit !important; opacity: 1 !important; }

/* ── Hint status line ────────────────────────────────── */
.cc-status-md, .cc-status-md p, .cc-status-md em, .cc-status-md strong {
  color: #3C4043 !important;
  font-size: 0.95rem !important;
  opacity: 1 !important;
  line-height: 1.5 !important;
}

/* ── Cards ───────────────────────────────────────────── */
.cc-card {
  background: #FFFFFF !important;
  border-radius: var(--radius-card) !important;
  border: 1px solid var(--border) !important;
  padding: 24px !important;
  box-shadow: var(--shadow-sm) !important;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  overflow: hidden;
}
.cc-card:hover { transform: translateY(-3px); box-shadow: var(--shadow-md) !important; }
.cc-card-lavender { background: linear-gradient(140deg, var(--lavender) 0%, #FAF8FF 100%) !important; }
.cc-card-sky      { background: linear-gradient(140deg, var(--sky) 0%, #F5F9FF 100%) !important; }
.cc-card-mint     { background: linear-gradient(140deg, var(--mint) 0%, #F4FBF7 100%) !important; }
.cc-card-peach    { background: linear-gradient(140deg, var(--peach) 0%, #FFF9F6 100%) !important; }
.cc-card-butter   { background: linear-gradient(140deg, var(--butter) 0%, #FFFCF2 100%) !important; }

.cc-card-label {
  font-size: 0.75rem !important; font-weight: 700 !important; letter-spacing: 0.08em !important;
  text-transform: uppercase !important; color: #4B5563 !important; margin-bottom: 4px !important;
  opacity: 1 !important;
}
.cc-card-title {
  font-size: 1.2rem !important; font-weight: 700 !important; color: #0B0B0F !important;
  letter-spacing: -0.02em !important; margin-bottom: 16px !important;
  opacity: 1 !important;
}

/* ── Audio Card ──────────────────────────────────────── */
.cc-audio-card {
  background: #FFFFFF !important;
  border: 1px solid var(--border) !important;
  border-radius: 18px !important;
  padding: 16px !important;
}
.cc-audio-card label, .cc-audio-card span {
  color: #0B0B0F !important; font-weight: 600 !important; opacity: 1 !important;
}

/* ── Results heading ─────────────────────────────────── */
.cc-results-heading {
  font-size: 1.5rem; font-weight: 700; color: #0B0B0F !important;
  letter-spacing: -0.025em; margin: 36px 0 16px;
  display: flex; align-items: center; gap: 10px;
}

/* ── Secondary button ────────────────────────────────── */
.cc-sec-btn button {
  background: #FFFFFF !important;
  color: #0B0B0F !important;
  border: 1.5px solid var(--border) !important;
  border-radius: var(--radius-pill) !important;
  padding: 10px 24px !important;
  font-size: 0.9375rem !important; font-weight: 600 !important;
  font-family: 'DM Sans', sans-serif !important;
  box-shadow: var(--shadow-sm) !important;
  opacity: 1 !important;
}
.cc-sec-btn button:hover {
  border-color: #0B0B0F !important;
  transform: scale(1.02) !important;
}

/* ── Animation opacity safety ────────────────────────── */
.cc-fade-up, .cc-fade-up-1, .cc-fade-up-2, .cc-fade-up-3, .cc-fade-up-4 {
  opacity: 1 !important;
  animation: fadeUp 0.5s cubic-bezier(0.22,1,0.36,1) forwards !important;
}
@keyframes fadeUp {
  from { opacity: 0.85; transform: translateY(12px); }
  to   { opacity: 1 !important; transform: translateY(0); }
}

/* ── Responsible AI banner ───────────────────────────── */
.cc-disclosure {
  display: flex; align-items: flex-start; gap: 12px;
  background: var(--lavender) !important;
  border: 1px solid var(--lavender-mid) !important;
  border-radius: 14px !important;
  padding: 14px 20px !important;
  font-size: 0.9375rem !important;
  color: #0B0B0F !important;
  line-height: 1.55 !important;
  max-width: 1200px !important;
  margin: 18px auto 0 !important;
  opacity: 1 !important;
}
.cc-disclosure strong { color: #0B0B0F !important; font-weight: 700 !important; }
.cc-disclosure em { color: #3C4043 !important; font-weight: 600 !important; }

/* ── Dataframe table light mode ──────────────────────── */
.gradio-container table,
.gradio-container tbody,
.gradio-container tr,
.gradio-container td {
  background-color: #FFFFFF !important;
  color: #0B0B0F !important;
  border-color: var(--border) !important;
}
.gradio-container th {
  background-color: #F8F9FA !important;
  color: #0B0B0F !important;
  font-weight: 700 !important;
  border-color: var(--border) !important;
}

/* ── Footer ──────────────────────────────────────────── */
.cc-footer {
  text-align: center; padding: 32px 24px;
  font-size: 0.86rem; color: #4B5563 !important; line-height: 1.6;
  border-top: 1px solid var(--border);
  margin-top: 56px; background: #F8F9FA;
  border-radius: 24px 24px 0 0;
}
.cc-footer strong { color: #0B0B0F !important; }

/* ── Diagnostics ─────────────────────────────────────── */
.cc-diag-rule { border:none; border-top:1px solid var(--border); margin:28px 0; }

/* ── Reduced motion ──────────────────────────────────── */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
"""

# ---------------------------------------------------------------------------
# Static HTML constants & scripts
# ---------------------------------------------------------------------------

_FORCE_LIGHT_JS = """
<script>
(function() {
  try {
    var url = new URL(window.location.href);
    if (url.searchParams.get('__theme') !== 'light') {
      url.searchParams.set('__theme', 'light');
      window.location.replace(url.toString());
      return;
    }
  } catch (e) {}
  function applyLight() {
    document.documentElement.classList.remove('dark');
    document.documentElement.classList.add('light');
    if (document.body) {
      document.body.classList.remove('dark');
      document.body.classList.add('light');
    }
  }
  applyLight();
  if (window.MutationObserver) {
    new MutationObserver(applyLight).observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class']
    });
  }
})();
</script>
"""

_NAV_HTML = """
<div class="cc-nav">
  <a class="cc-nav-brand" href="#" onclick="return false;">
    <span>🎬</span> Storyboard Animator
  </a>
</div>
<div class="cc-nav-spacer"></div>
"""

_HERO_HTML = """
<div class="cc-hero cc-fade-up">
  <div class="cc-blobs">
    <div class="cc-blob cc-blob-1"></div>
    <div class="cc-blob cc-blob-2"></div>
    <div class="cc-blob cc-blob-3"></div>
  </div>
  <div class="cc-hero-inner">
    <div class="cc-hero-eyebrow cc-fade-up cc-fade-up-1">✨ Generative AI · Story Studio</div>
    <h1 class="cc-hero-h1 cc-fade-up cc-fade-up-2">
      Turn any idea into<br>
      <em>a 6-shot animated story.</em>
    </h1>
    <p class="cc-hero-sub cc-fade-up cc-fade-up-3">
      Describe your concept below. Gemini writes the narrative beats,
      SD-Turbo renders the frames, and BLIP verifies semantic quality.
    </p>
  </div>
</div>
"""

_DISCLOSURE_HTML = """
<div class="cc-disclosure cc-fade-up cc-fade-up-4">
  <span style="font-size:1.35rem;flex-shrink:0;">🛡️</span>
  <div>
    <strong>Responsible AI Disclosure:</strong>
    All visual imagery, story beats, and captions are
    <strong>synthetically generated by AI</strong>
    (Gemini LLM · SD-Turbo diffusion · BLIP captioning).
    An indelible <em>"AI-Generated"</em> watermark is stamped on every media frame.
  </div>
</div>
"""

_FOOTER_HTML = """
<div class="cc-footer">
  <p>
    All visuals are AI-generated (Gemini · SD-Turbo · BLIP). Every frame carries an
    <strong>AI-Generated</strong> label. &nbsp;·&nbsp;
    Built with Gradio · Google Generative AI · Hugging Face Transformers
  </p>
</div>
"""


# ---------------------------------------------------------------------------
# App Construction
# ---------------------------------------------------------------------------

def build_app():
    with gr.Blocks(title="Storyboard-to-GIF Animator") as demo:

        # ── Force light mode JS & Nav (logo + project name only) ─────────────
        gr.HTML(_FORCE_LIGHT_JS + _NAV_HTML)

        # ── Navigation Tabs ─────────────────────────────────────────────────
        with gr.Tabs(elem_classes=["cc-tabs"]):

            # ================================================================
            # TAB 1 · Storyboard Studio
            # ================================================================
            with gr.TabItem("🎬 Studio"):

                # Hero section & Responsible AI banner
                gr.HTML(_HERO_HTML)
                gr.HTML(_DISCLOSURE_HTML)

                # ── Idea input & settings ───────────────────────────────────
                with gr.Column(elem_classes=["cc-section"]):
                    gr.HTML('<div style="height:24px;"></div>')

                    user_idea_input = gr.Textbox(
                        label="Your Story Concept",
                        placeholder="e.g. A small brass robot discovers an easel and paints sunflowers in a greenhouse at dawn…",
                        value="A small lonely robot in a greenhouse learns to paint flowers",
                        lines=3,
                        elem_classes=["cc-idea-card"],
                    )

                    # Preset chips
                    with gr.Row(elem_classes=["cc-chips"]):
                        chips = []
                        for label, _ in PRESET_IDEAS:
                            chips.append(
                                gr.Button(label, elem_classes=["cc-chip"], size="sm")
                            )

                    # Advanced settings accordion
                    with gr.Accordion("🎨  Advanced Settings", open=False, elem_classes=["cc-advanced-accordion"]):
                        gr.HTML('<div style="height:8px;"></div>')
                        with gr.Row():
                            technique_radio = gr.Radio(
                                choices=["few-shot", "zero-shot"],
                                value="few-shot",
                                label="Prompt Engineering Technique",
                                info="Few-shot: in-context shot-list examples · Zero-shot: direct instructions",
                            )
                            seed_input = gr.Number(
                                label="Fixed Seed (Reproducibility)",
                                value=42, precision=0, step=1,
                            )
                        style_prefix_input = gr.Textbox(
                            label="Shared Artistic Style Prefix (prepended to each image prompt)",
                            value=DEFAULT_STYLE_PREFIX,
                            lines=2,
                        )
                        with gr.Row():
                            mismatch_threshold_slider = gr.Slider(
                                minimum=0.10, maximum=0.70,
                                value=DEFAULT_MISMATCH_THRESHOLD, step=0.05,
                                label="Mismatch Detection Cutoff (Cosine Similarity)",
                            )
                            image_mode_select = gr.Dropdown(
                                choices=["auto", "local_diffusers", "hf_api", "mock"],
                                value="auto",
                                label="Image Engine Mode",
                                info="'auto' tries local GPU, then HF API, then canvas fallback.",
                            )

                    # Generate button (Solid black pill)
                    with gr.Column(elem_classes=["cc-gen-btn"]):
                        generate_btn = gr.Button(
                            "🚀  Generate 6-Shot Story  →", size="lg"
                        )

                # ── Pipeline progress strip ────────────────────────────────
                with gr.Column(elem_classes=["cc-section"]):
                    pipeline_progress_html = gr.HTML(make_pipeline_html())

                # ── Status hint & Audio card ───────────────────────────────
                with gr.Row(elem_classes=["cc-section"]):
                    with gr.Column(scale=7):
                        status_markdown = gr.Markdown(
                            "💡 *Enter a story idea above and press **Generate** to begin.*",
                            elem_classes=["cc-status-md"]
                        )
                    with gr.Column(scale=3, elem_classes=["cc-audio-card"]):
                        audio_player = gr.Audio(
                            label="🎵 Background Music Track",
                            type="filepath",
                        )

                gr.HTML('<div class="cc-section-gap"></div>')

                # ── Results: 3 pastel cards ────────────────────────────────
                with gr.Column(elem_classes=["cc-section"]):
                    gr.HTML('<div class="cc-results-heading">✨ Generated Storyboard & Animations</div>')

                with gr.Row(elem_classes=["cc-section"], equal_height=False):

                    # Card 1 · Animation export
                    with gr.Column(scale=4, elem_classes=["cc-card", "cc-card-lavender"]):
                        gr.HTML(
                            '<div class="cc-card-label">Output</div>'
                            '<div class="cc-card-title">Your Animation</div>'
                        )
                        with gr.Tabs():
                            with gr.TabItem("▶ MP4 Video"):
                                video_player = gr.Video(
                                    label="Story Animation (with music)",
                                    autoplay=False,
                                )
                            with gr.TabItem("🎞 GIF"):
                                gif_image = gr.Image(
                                    label="Animated GIF (seamless loop)",
                                )

                    # Card 2 · Storyboard frames
                    with gr.Column(scale=5, elem_classes=["cc-card", "cc-card-sky"]):
                        gr.HTML(
                            '<div class="cc-card-label">Frames</div>'
                            '<div class="cc-card-title">6-Shot Storyboard</div>'
                        )
                        frames_gallery = gr.Gallery(
                            label="Storyboard Frames (AI-labelled)",
                            columns=3, rows=2, height=340,
                            object_fit="contain",
                        )

                    # Card 3 · Quality check
                    with gr.Column(scale=3, elem_classes=["cc-card", "cc-card-mint"]):
                        gr.HTML(
                            '<div class="cc-card-label">Quality</div>'
                            '<div class="cc-card-title">BLIP Score</div>'
                        )
                        blip_ring_html = gr.HTML(make_blip_ring_html())

                gr.HTML('<div style="height:16px;"></div>')

                # ── Eval table ─────────────────────────────────────────────
                with gr.Column(elem_classes=["cc-section"]):
                    gr.HTML(
                        '<div style="font-size:0.95rem;color:#3C4043;font-weight:500;margin-bottom:12px;">'
                        'Each frame is captioned by <strong>Salesforce/blip-image-captioning-base</strong> '
                        'and compared to the intended prompt via <strong>sentence-transformers (all-MiniLM-L6-v2)</strong>.'
                        '</div>'
                    )
                    eval_table = gr.Dataframe(
                        headers=["Shot #", "Intended Shot / Caption", "BLIP Caption",
                                 "Cosine Similarity", "Status"],
                        datatype=["number", "str", "str", "str", "str"],
                        label="Per-Frame Semantic Alignment",
                        interactive=False,
                    )

                gr.HTML('<div style="height:12px;"></div>')

                # ── Raw JSON accordion ─────────────────────────────────────
                with gr.Column(elem_classes=["cc-section"]):
                    with gr.Accordion("📄  View Raw LLM Storyboard JSON", open=False):
                        json_viewer = gr.JSON(label="LLM Structured Output")

                gr.HTML('<div style="height:32px;"></div>')

            # ================================================================
            # TAB 2 · Prompt Technique Comparison
            # ================================================================
            with gr.TabItem("⚖️ Compare"):
                with gr.Column(elem_classes=["cc-section"]):
                    gr.HTML('<div style="height:32px;"></div>')
                    gr.HTML(
                        '<h2 style="font-size:1.6rem;font-weight:700;color:#0B0B0F;'
                        'letter-spacing:-0.025em;margin-bottom:8px;">Prompt Technique Comparison</h2>'
                        '<p style="color:#3C4043;font-size:1rem;margin-bottom:28px;">'
                        'See how <strong>Few-Shot</strong> (in-context examples) vs '
                        '<strong>Zero-Shot</strong> (direct instructions) change the LLM\'s '
                        'camera language, rhythm, and descriptive grounding.</p>'
                    )

                    compare_concept_input = gr.Textbox(
                        label="Concept to Compare",
                        value="A timid deep-sea octopus invents a bioluminescent lantern",
                        lines=2,
                    )
                    with gr.Row():
                        with gr.Column(elem_classes=["cc-sec-btn"]):
                            compare_btn = gr.Button("🔬  Run Side-by-Side Comparison")

                    compare_summary_md = gr.Markdown()

                    with gr.Row():
                        with gr.Column(elem_classes=["cc-card", "cc-card-mint"]):
                            gr.HTML('<div class="cc-card-label">Technique</div>'
                                    '<div class="cc-card-title">🟢 Zero-Shot Output</div>')
                            zero_shot_json = gr.JSON(label="Zero-Shot Storyboard")
                        with gr.Column(elem_classes=["cc-card", "cc-card-lavender"]):
                            gr.HTML('<div class="cc-card-label">Technique</div>'
                                    '<div class="cc-card-title">🟣 Few-Shot Output</div>')
                            few_shot_json = gr.JSON(label="Few-Shot Storyboard")
                    gr.HTML('<div style="height:32px;"></div>')

            # ================================================================
            # TAB 3 · Responsible AI & Diagnostics
            # ================================================================
            with gr.TabItem("🛡️ Responsible AI"):
                with gr.Column(elem_classes=["cc-section"]):
                    gr.HTML('<div style="height:32px;"></div>')
                    gr.HTML(
                        '<h2 style="font-size:1.6rem;font-weight:700;color:#0B0B0F;'
                        'letter-spacing:-0.025em;margin-bottom:24px;">Responsible AI Guardrail Suite</h2>'
                    )

                    with gr.Row():
                        with gr.Column(elem_classes=["cc-card", "cc-card-peach"]):
                            gr.HTML('<div class="cc-card-label">Test</div>'
                                    '<div class="cc-card-title">1. Safety Guardrail Tester</div>')
                            guardrail_test_input = gr.Textbox(
                                label="Test a prompt against safety policies",
                                value="Donald Trump and Batman in a violent gunfight",
                                lines=2,
                            )
                            with gr.Row(elem_classes=["cc-sec-btn"]):
                                guardrail_check_btn = gr.Button("🔍  Check Guardrails")
                            guardrail_result_md = gr.Markdown()

                        with gr.Column(elem_classes=["cc-card", "cc-card-butter"]):
                            gr.HTML('<div class="cc-card-label">Policy</div>'
                                    '<div class="cc-card-title">2. Guardrail Rules</div>')
                            gr.Markdown("""
- **Violence & Harm** — Blocks gore, weapons, mass casualty, or self-harm.
- **Hate Speech** — Blocks slurs, supremacy, xenophobia, and hate incidents.
- **Sexual Content** — Blocks explicit adult or erotic depictions.
- **Real-Person Deepfakes** — Protects public figures & living individuals.
- **Trademarked IP** — Prevents commercial copyright infringement (Disney, Marvel, etc.).
                            """)

                    gr.HTML('<hr class="cc-diag-rule">')

                    gr.HTML(
                        '<h3 style="font-size:1.2rem;font-weight:700;color:#0B0B0F;'
                        'letter-spacing:-0.02em;margin-bottom:14px;">Failure Case & Semantic Drift Analysis</h3>'
                    )
                    gr.Markdown("""
In Generative AI pipelines, text-to-image models can suffer from **object omission**,
**attribute binding errors**, or **semantic hallucination**.

- **Example Failure Case**: The prompt asked for *"a brass robot holding a paintbrush"*, but the model generated a generic landscape where the robot appeared as a rock or was missing entirely.
- **BLIP Detection**: BLIP caption was *"a scenic mountain path with boulders"* → cosine similarity **0.24** (< 0.35 cutoff).
- **Mitigation**: The system flags this as **⚠️ MISMATCH**, prompting the user to adjust or regenerate with an alternate seed.
                    """)
                    gr.HTML('<div style="height:32px;"></div>')

            # ================================================================
            # TAB 4 · Secrets & Config
            # ================================================================
            with gr.TabItem("⚙️ Config"):
                with gr.Column(elem_classes=["cc-section"]):
                    gr.HTML('<div style="height:32px;"></div>')
                    gr.HTML(
                        '<h2 style="font-size:1.6rem;font-weight:700;color:#0B0B0F;'
                        'letter-spacing:-0.025em;margin-bottom:8px;">API Credentials & Config</h2>'
                        '<p style="color:#3C4043;font-size:1rem;margin-bottom:28px;">'
                        'Set via environment variables (<code>GEMINI_API_KEY</code>, <code>HF_TOKEN</code>) '
                        'or use the session overrides below. <em>Keys are never logged or saved to disk.</em></p>'
                    )
                    with gr.Row():
                        with gr.Column(elem_classes=["cc-card"]):
                            ui_gemini_key = gr.Textbox(
                                label="Gemini API Key (Optional Override)",
                                placeholder="AIzaSy...",
                                type="password",
                            )
                        with gr.Column(elem_classes=["cc-card"]):
                            ui_hf_token = gr.Textbox(
                                label="Hugging Face Token (Optional Override)",
                                placeholder="hf_...",
                                type="password",
                            )
                    gr.Markdown(
                        "ℹ️ When deploying to **Hugging Face Spaces**, add `GEMINI_API_KEY` "
                        "and `HF_TOKEN` under **Settings › Variables and secrets**."
                    )
                    gr.HTML('<div style="height:32px;"></div>')

        # ── Footer ──────────────────────────────────────────────────────────
        gr.HTML(_FOOTER_HTML)

        # ================================================================
        # Event wiring
        # ================================================================

        # Preset chip click handlers
        for chip_btn, (_, full_idea) in zip(chips, PRESET_IDEAS):
            chip_btn.click(
                fn=lambda v=full_idea: v,
                outputs=user_idea_input,
            )

        # Main pipeline
        generate_btn.click(
            fn=_run_pipeline_generator,
            inputs=[
                user_idea_input,
                technique_radio,
                seed_input,
                style_prefix_input,
                mismatch_threshold_slider,
                image_mode_select,
                ui_gemini_key,
                ui_hf_token,
            ],
            outputs=[
                pipeline_progress_html,
                status_markdown,
                json_viewer,
                frames_gallery,
                eval_table,
                blip_ring_html,
                gif_image,
                video_player,
                audio_player,
            ],
        )

        # Prompt comparison
        compare_btn.click(
            fn=run_technique_comparison,
            inputs=[compare_concept_input, style_prefix_input, ui_gemini_key],
            outputs=[compare_summary_md, zero_shot_json, few_shot_json],
        )

        # Guardrail tester
        def _test_guardrail(prompt):
            _, msg, _ = check_safety(prompt)
            return msg

        guardrail_check_btn.click(
            fn=_test_guardrail,
            inputs=[guardrail_test_input],
            outputs=[guardrail_result_md],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def create_light_theme():
    """Explicit light theme based on Base."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.neutral,
        secondary_hue=gr.themes.colors.neutral,
        neutral_hue=gr.themes.colors.neutral,
    ).set(
        body_background_fill="#FFFFFF",
        body_text_color="#0B0B0F",
        background_fill_primary="#FFFFFF",
        background_fill_secondary="#F8F9FA",
        block_background_fill="#FFFFFF",
        panel_background_fill="#FFFFFF",
        input_background_fill="#FFFFFF",
        button_secondary_background_fill="#F8F9FA",
        block_label_text_color="#0B0B0F",
        block_title_text_color="#0B0B0F",
        border_color_primary="#E6E8EC",
    )


if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=create_light_theme(),
        css=CUSTOM_CSS,
    )
