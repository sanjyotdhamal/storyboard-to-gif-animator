"""
app.py - Storyboard-to-GIF Animator — Clean Creator Edition
=============================================================
Redesigned UI with Antigravity-inspired "Clean Creator" aesthetic.
Backend files (safety.py, llm.py, imagegen.py, blip_check.py, video.py) are UNCHANGED.
All features preserved: idea input, few-shot/zero-shot, seed, advanced settings,
generate button, gallery, MP4 + GIF, music, BLIP table, JSON viewer, all tabs.
"""

import os
import math
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
    """Renders the 5-step pipeline progress strip."""
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
        pct, score_str, ring_color, label_color = 0.0, "—", "#E6E8EC", "#9CA3AF"
    else:
        pct = min(1.0, max(0.0, float(score)))
        score_str = f"{pct:.3f}"
        if pct >= 0.50:
            ring_color, label_color = "#34D399", "#059669"
        elif pct >= 0.35:
            ring_color, label_color = "#FBBF24", "#D97706"
        else:
            ring_color, label_color = "#F87171", "#DC2626"

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
      <span style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                   letter-spacing:0.08em;color:{label_color};">BLIP</span>
    </div>
  </div>
  <p style="font-size:0.78rem;color:#6B7280;text-align:center;margin-top:10px;
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
    Generator that yields partial UI state after each pipeline stage.
    Outputs (9): pipeline_html, status_md, json_viewer, gallery,
                 eval_table, blip_ring_html, gif_image, video_player, audio_player
    """
    _EMPTY = (make_pipeline_html(), None, None, None, None, make_blip_ring_html(), None, None, None)

    # ── Step 0: Safety ──────────────────────────────────────────────────────
    yield (make_pipeline_html(0), "🛡️ Checking content safety...",
           None, None, None, make_blip_ring_html(), None, None, None)

    is_safe, safety_msg, _ = check_safety(idea)
    if not is_safe:
        yield (make_pipeline_html(),
               f"🚫 **Safety block:** {safety_msg}",
               None, None, None, make_blip_ring_html(), None, None, None)
        return

    # ── Step 1: LLM Storyboard ───────────────────────────────────────────
    yield (make_pipeline_html(1, {0}),
           f"✍️ Generating `{technique}` storyboard with Gemini…",
           None, None, None, make_blip_ring_html(), None, None, None)

    clean_prefix = style_prefix.strip() or DEFAULT_STYLE_PREFIX
    storyboard_data, llm_note = generate_storyboard(
        idea=idea, technique=technique, style_prefix=clean_prefix,
        api_key=api_key_override.strip() if api_key_override else None,
    )
    shots = storyboard_data.get("shots", [])

    # ── Step 2: Frame Generation ─────────────────────────────────────────
    yield (make_pipeline_html(2, {0, 1}),
           f"🖼️ Generating {len(shots)} frames (seed={seed})…",
           storyboard_data, None, None, make_blip_ring_html(), None, None, None)

    frames, gen_methods = generate_storyboard_frames(
        shots=shots, seed=int(seed), mode=image_mode,
        hf_token=hf_token_override.strip() if hf_token_override else None,
        progress_callback=lambda p, msg: progress(0.35 + p * 0.30, desc=msg),
    )

    # Show frames immediately while video renders
    yield (make_pipeline_html(3, {0, 1, 2}),
           "🎬 Rendering cross-fade animation with music…",
           storyboard_data, frames, None, make_blip_ring_html(), None, None, None)

    # ── Step 3: Video ────────────────────────────────────────────────────
    gif_path, mp4_path, music_path = create_storyboard_media(
        frames=frames, fps_video=24, fps_gif=15,
        hold_frames=12, transition_frames=8,
        progress_callback=lambda p, msg: progress(0.65 + p * 0.20, desc=msg),
    )

    # ── Step 4: BLIP Quality Check ───────────────────────────────────────
    yield (make_pipeline_html(4, {0, 1, 2, 3}),
           "🔍 Running BLIP semantic quality check…",
           storyboard_data, frames, None, make_blip_ring_html(),
           gif_path, mp4_path, music_path)

    eval_df, avg_score, eval_details = evaluate_storyboard_frames(
        frames=frames, shots=shots, threshold=float(mismatch_threshold),
        progress_callback=lambda p, msg: progress(0.85 + p * 0.15, desc=msg),
    )

    mismatch_count = sum(1 for d in eval_details if d.get("is_mismatch"))
    engine = gen_methods[0] if gen_methods else "Canvas"
    summary = (
        f"### 🎉 Storyboard Generated!\n"
        f"**Technique**: `{technique}` &nbsp;·&nbsp; **Engine**: `{engine}` "
        f"&nbsp;·&nbsp; **Seed**: `{seed}`  \n"
        f"**BLIP Score**: `{avg_score:.3f}` &nbsp;·&nbsp; "
        f"**Flags**: `{mismatch_count}` mismatch(es) (threshold: `{mismatch_threshold}`)  \n"
        f"*AI-Generated watermark applied to every frame.*"
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
# CSS — Clean Creator / Antigravity-inspired light theme
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* ── Fonts ───────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

/* ── Design tokens ───────────────────────────────────── */
:root {
  --white:   #FFFFFF;
  --bg:      #F8F9FA;
  --black:   #0B0B0F;
  --text:    #1A1A2E;
  --muted:   #6B7280;
  --border:  #E6E8EC;
  --radius-card: 24px;
  --radius-pill: 999px;
  --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
  --shadow-md: 0 4px 20px rgba(0,0,0,0.08);
  --shadow-lg: 0 8px 36px rgba(0,0,0,0.12);
  /* Pastel palette */
  --lavender:      #E8E3FF;
  --lavender-mid:  #C4B5FD;
  --lavender-dark: #7C3AED;
  --mint:          #DDF5EA;
  --mint-mid:      #86EFAC;
  --mint-dark:     #15803D;
  --peach:         #FFE8DB;
  --peach-dark:    #EA580C;
  --sky:           #DCEBFF;
  --sky-dark:      #2563EB;
  --butter:        #FFF3CC;
  --butter-dark:   #D97706;
}

/* ── Gradio global resets ────────────────────────────── */
.gradio-container {
  max-width: 100% !important;
  padding: 0 !important;
  margin: 0 !important;
  background: var(--white) !important;
  font-family: 'DM Sans', Inter, -apple-system, BlinkMacSystemFont, sans-serif !important;
  font-size: 16px !important;
  color: var(--text) !important;
}
footer.svelte-1rjryqp { display: none !important; }

/* ── Fixed Nav ───────────────────────────────────────── */
.cc-nav {
  position: fixed;
  top: 0; left: 0; right: 0;
  z-index: 999;
  height: 60px;
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 36px;
  background: rgba(255,255,255,0.88);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border-bottom: 1px solid var(--border);
}
.cc-nav-brand {
  font-size: 1.0rem; font-weight: 700; color: var(--black);
  letter-spacing: -0.02em; display: flex; align-items: center; gap: 8px;
  text-decoration: none;
}
.cc-nav-brand span { font-size: 1.3rem; }
.cc-nav-links { display: flex; gap: 2px; }
.cc-nav-links button {
  background: none; border: none; cursor: pointer;
  font-size: 0.875rem; font-weight: 500;
  color: var(--muted); padding: 7px 14px;
  border-radius: var(--radius-pill);
  font-family: 'DM Sans', sans-serif;
  transition: color 0.18s ease, background 0.18s ease;
}
.cc-nav-links button:hover { color: var(--black); background: var(--bg); }

/* ── Top spacer (push content below fixed nav) ───────── */
.cc-nav-spacer { height: 60px; }

/* ── Hero ────────────────────────────────────────────── */
.cc-hero {
  position: relative; overflow: hidden;
  padding: 72px 32px 60px; text-align: center;
  background: var(--white);
}
.cc-hero-inner { position: relative; z-index: 1; }
.cc-hero-eyebrow {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--lavender); color: var(--lavender-dark);
  border-radius: var(--radius-pill); padding: 5px 14px;
  font-size: 0.8rem; font-weight: 700; letter-spacing: 0.06em;
  text-transform: uppercase; margin-bottom: 20px;
}
.cc-hero-h1 {
  font-size: clamp(2.4rem, 5.5vw, 3.75rem);
  font-weight: 700; color: var(--black);
  letter-spacing: -0.035em; line-height: 1.07;
  margin-bottom: 10px;
}
.cc-hero-h1 em { color: var(--muted); font-style: normal; }
.cc-hero-sub {
  font-size: 1.1rem; color: var(--muted); line-height: 1.65;
  max-width: 540px; margin: 0 auto 36px; font-weight: 400;
}

/* ── Floating blobs ──────────────────────────────────── */
.cc-blobs {
  position: absolute; inset: 0; pointer-events: none; overflow: hidden;
}
.cc-blob {
  position: absolute; border-radius: 50%; filter: blur(72px); opacity: 0.42;
}
.cc-blob-1 {
  width:380px; height:380px; background:var(--lavender);
  top:-120px; left:-80px;
  animation: blobA 26s ease-in-out infinite;
}
.cc-blob-2 {
  width:300px; height:300px; background:var(--mint);
  top:-80px; right:-60px;
  animation: blobB 22s ease-in-out infinite;
}
.cc-blob-3 {
  width:260px; height:260px; background:var(--peach);
  bottom:-100px; left:35%;
  animation: blobC 30s ease-in-out infinite;
}
@keyframes blobA {
  0%,100%{transform:translate(0,0) scale(1)}
  40%{transform:translate(28px,18px) scale(1.06)}
  70%{transform:translate(-16px,8px) scale(0.96)}
}
@keyframes blobB {
  0%,100%{transform:translate(0,0) scale(1)}
  35%{transform:translate(-22px,14px) scale(1.07)}
  65%{transform:translate(14px,-10px) scale(0.95)}
}
@keyframes blobC {
  0%,100%{transform:translate(0,0) scale(1)}
  30%{transform:translate(18px,-14px) scale(1.04)}
  60%{transform:translate(-10px,18px) scale(0.97)}
}

/* sparkle dots */
.cc-sparkle {
  position: absolute; border-radius: 50%; background: var(--lavender-mid);
  animation: sparkle linear infinite; opacity: 0;
}
@keyframes sparkle {
  0%   { opacity:0; transform:translateY(0) scale(0); }
  10%  { opacity:0.9; transform:translateY(-8px) scale(1); }
  85%  { opacity:0.3; transform:translateY(-70px) scale(0.7); }
  100% { opacity:0; transform:translateY(-90px) scale(0); }
}

/* ── Idea input card ─────────────────────────────────── */
.cc-idea-card {
  max-width: 760px; margin: 0 auto; position: relative; z-index: 1;
}
.cc-idea-card textarea {
  font-size: 1.0625rem !important;
  font-family: 'DM Sans', sans-serif !important;
  border-radius: 18px !important;
  border: 1.5px solid var(--border) !important;
  padding: 20px 22px !important;
  background: var(--white) !important;
  box-shadow: var(--shadow-md) !important;
  color: var(--black) !important;
  line-height: 1.6 !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
.cc-idea-card textarea:focus {
  border-color: var(--lavender-mid) !important;
  box-shadow: 0 0 0 4px rgba(196,181,253,0.18), var(--shadow-md) !important;
  outline: none !important;
}

/* ── Preset chips ────────────────────────────────────── */
.cc-chips {
  display: flex; flex-wrap: wrap; justify-content: center;
  gap: 8px; margin: 14px auto 0; max-width: 760px;
  position: relative; z-index: 1;
}
.cc-chip button {
  background: var(--bg) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-pill) !important;
  padding: 7px 16px !important;
  font-size: 0.84rem !important; font-weight: 500 !important;
  font-family: 'DM Sans', sans-serif !important;
  cursor: pointer !important; white-space: nowrap !important;
  box-shadow: none !important;
  transition: background 0.18s, border-color 0.18s, transform 0.18s, box-shadow 0.18s !important;
}
.cc-chip button:hover {
  background: var(--lavender) !important;
  border-color: var(--lavender-mid) !important;
  transform: translateY(-2px) !important;
  box-shadow: var(--shadow-sm) !important;
}
.cc-chip button:active { transform: translateY(0) !important; }

/* ── Generate button ─────────────────────────────────── */
.cc-gen-btn { max-width: 760px; margin: 14px auto 0; position: relative; z-index: 1; }
.cc-gen-btn button {
  background: var(--black) !important;
  color: var(--white) !important;
  border: none !important;
  border-radius: var(--radius-pill) !important;
  padding: 15px 36px !important;
  font-size: 1.0625rem !important; font-weight: 600 !important;
  font-family: 'DM Sans', sans-serif !important;
  width: 100% !important; cursor: pointer !important;
  box-shadow: 0 4px 16px rgba(11,11,15,0.22) !important;
  transition: transform 0.18s, box-shadow 0.18s !important;
  letter-spacing: -0.01em !important;
}
.cc-gen-btn button:hover {
  transform: scale(1.025) !important;
  box-shadow: 0 6px 24px rgba(11,11,15,0.30) !important;
}
.cc-gen-btn button:active { transform: scale(0.99) !important; }

/* ── Section wrapper ─────────────────────────────────── */
.cc-section {
  max-width: 1200px; margin: 0 auto; padding: 0 32px;
}
.cc-section-gap { height: 48px; }

/* ── Tabs ────────────────────────────────────────────── */
.gradio-container .tabs > .tab-nav {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  padding: 0 32px !important;
}
.gradio-container .tabs > .tab-nav button {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.9rem !important; font-weight: 500 !important;
  color: var(--muted) !important;
  padding: 14px 18px !important; border-radius: 0 !important;
  border-bottom: 2px solid transparent !important;
  background: transparent !important;
  transition: color 0.18s, border-color 0.18s !important;
}
.gradio-container .tabs > .tab-nav button.selected,
.gradio-container .tabs > .tab-nav button[aria-selected="true"] {
  color: var(--black) !important;
  border-bottom-color: var(--black) !important;
  font-weight: 600 !important;
}

/* ── Pipeline progress strip ─────────────────────────── */
.cc-pipeline {
  display: flex; align-items: center; flex-wrap: wrap;
  gap: 5px; padding: 20px 0 10px;
}
.cc-pipe-step {
  display: flex; align-items: center; gap: 5px;
  padding: 7px 15px; border-radius: var(--radius-pill);
  font-size: 0.82rem; font-weight: 600; white-space: nowrap;
  border: 1.5px solid transparent;
  transition: all 0.3s cubic-bezier(0.22,1,0.36,1);
}
.cc-pipe-step.idle   { background:var(--bg); color:var(--muted); border-color:var(--border); }
.cc-pipe-step.active { background:var(--lavender); color:var(--lavender-dark); border-color:var(--lavender-mid); animation:pulsePill 1.6s ease-in-out infinite; }
.cc-pipe-step.done   { background:var(--mint); color:var(--mint-dark); border-color:var(--mint-mid); }
.cc-pipe-arrow       { color:var(--border); font-size:1.1rem; flex-shrink:0; }
.cc-pipe-label       { display:inline; }
@keyframes pulsePill {
  0%,100%{opacity:1; transform:scale(1)}
  50%{opacity:0.72; transform:scale(1.05)}
}

/* ── Cards ───────────────────────────────────────────── */
.cc-card {
  background: var(--white);
  border-radius: var(--radius-card) !important;
  border: 1px solid var(--border);
  padding: 24px !important;
  box-shadow: var(--shadow-sm);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  overflow: hidden;
}
.cc-card:hover { transform: translateY(-3px); box-shadow: var(--shadow-md); }
.cc-card-lavender { background: linear-gradient(140deg, var(--lavender) 0%, #F5F3FF 100%) !important; }
.cc-card-sky      { background: linear-gradient(140deg, var(--sky) 0%, #EFF6FF 100%) !important; }
.cc-card-mint     { background: linear-gradient(140deg, var(--mint) 0%, #F0FDF4 100%) !important; }
.cc-card-peach    { background: linear-gradient(140deg, var(--peach) 0%, #FFF7F5 100%) !important; }
.cc-card-label {
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--muted); margin-bottom: 4px;
}
.cc-card-title {
  font-size: 1.15rem; font-weight: 700; color: var(--black);
  letter-spacing: -0.02em; margin-bottom: 16px;
}

/* ── Results heading ─────────────────────────────────── */
.cc-results-heading {
  font-size: 1.5rem; font-weight: 700; color: var(--black);
  letter-spacing: -0.025em; margin: 44px 0 20px;
  display: flex; align-items: center; gap: 10px;
}

/* ── Secondary button ────────────────────────────────── */
.cc-sec-btn button {
  background: var(--white) !important;
  color: var(--black) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: var(--radius-pill) !important;
  padding: 10px 24px !important;
  font-size: 0.9375rem !important; font-weight: 600 !important;
  font-family: 'DM Sans', sans-serif !important;
  transition: border-color 0.18s, transform 0.18s, box-shadow 0.18s !important;
  box-shadow: none !important;
}
.cc-sec-btn button:hover {
  border-color: var(--black) !important;
  transform: scale(1.02) !important;
  box-shadow: var(--shadow-sm) !important;
}

/* ── Fade-up entrance ────────────────────────────────── */
.cc-fade-up { animation: fadeUp 0.6s cubic-bezier(0.22,1,0.36,1) both; }
.cc-fade-up-1 { animation-delay: 0.06s; }
.cc-fade-up-2 { animation-delay: 0.13s; }
.cc-fade-up-3 { animation-delay: 0.20s; }
.cc-fade-up-4 { animation-delay: 0.28s; }
@keyframes fadeUp {
  from { opacity:0; transform:translateY(22px); }
  to   { opacity:1; transform:translateY(0);    }
}

/* ── Shimmer skeleton ────────────────────────────────── */
.cc-skeleton {
  background: linear-gradient(90deg,#F3F4F6 25%,#E9EAEC 50%,#F3F4F6 75%);
  background-size: 200% 100%;
  animation: shimmer 1.4s infinite;
  border-radius: 14px; min-height: 180px;
}
@keyframes shimmer {
  0%  { background-position: 200% 0; }
  100%{ background-position: -200% 0; }
}

/* ── Gradio form elements ────────────────────────────── */
.gradio-container label {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.875rem !important; font-weight: 600 !important;
  color: var(--text) !important;
}
.gradio-container .gr-input,
.gradio-container input[type="number"],
.gradio-container select {
  border-radius: 10px !important;
  border: 1.5px solid var(--border) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.9375rem !important;
}
.gradio-container details > summary {
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important; font-size: 0.9rem !important;
}

/* ── Gallery ─────────────────────────────────────────── */
.gradio-container .svelte-1oiizq5,
.gradio-container [data-testid="gallery"] {
  gap: 8px !important; border-radius: 14px !important;
}

/* ── Dataframe ───────────────────────────────────────── */
.gradio-container table {
  font-family: 'DM Sans', sans-serif !important;
  font-size: 0.875rem !important; border-radius: 10px !important;
}
.gradio-container th {
  background: var(--bg) !important;
  font-weight: 700 !important; color: var(--text) !important;
}

/* ── Audio / Video ───────────────────────────────────── */
.gradio-container audio { border-radius: 12px !important; width:100%!important; }
.gradio-container video { border-radius: 14px !important; }

/* ── Markdown body ───────────────────────────────────── */
.gradio-container .md p, .gradio-container .md li {
  font-size: 1rem !important; line-height: 1.65 !important;
  color: var(--text) !important;
}
.gradio-container .md h1,.gradio-container .md h2,.gradio-container .md h3 {
  font-family: 'DM Sans', sans-serif !important;
  letter-spacing: -0.02em !important; color: var(--black) !important;
}

/* ── AI disclosure banner ────────────────────────────── */
.cc-disclosure {
  display: flex; align-items: flex-start; gap: 12px;
  background: var(--lavender); border: 1px solid var(--lavender-mid);
  border-radius: 14px; padding: 14px 20px;
  font-size: 0.875rem; color: var(--lavender-dark); line-height: 1.55;
  max-width: 1200px; margin: 24px auto 0;
}

/* ── Footer ──────────────────────────────────────────── */
.cc-footer {
  text-align: center; padding: 32px 24px;
  font-size: 0.84rem; color: var(--muted); line-height: 1.6;
  border-top: 1px solid var(--border);
  margin-top: 64px; background: var(--bg);
  border-radius: 24px 24px 0 0;
}

/* ── Diagnostic section ──────────────────────────────── */
.cc-diag-rule { border:none; border-top:1px solid var(--border); margin:28px 0; }

/* ── Responsive ──────────────────────────────────────── */
@media (max-width: 768px) {
  .cc-nav { padding: 0 20px; }
  .cc-nav-links { display: none; }
  .cc-hero { padding: 48px 20px 40px; }
  .cc-section { padding: 0 20px; }
  .cc-hero-h1 { font-size: 2rem; }
  .cc-pipe-label { display: none; }
  .cc-pipe-step { padding: 6px 10px; font-size: 0.75rem; }
}

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
# Static HTML constants
# ---------------------------------------------------------------------------

_NAV_HTML = """
<div class="cc-nav">
  <a class="cc-nav-brand" href="#" onclick="return false;">
    <span>🎬</span> Storyboard Animator
  </a>
  <div class="cc-nav-links">
    <button onclick="(function(){var btns=document.querySelectorAll('.tab-nav button');if(btns[0])btns[0].click();})()">Studio</button>
    <button onclick="(function(){var btns=document.querySelectorAll('.tab-nav button');if(btns[1])btns[1].click();})()">Compare</button>
    <button onclick="(function(){var btns=document.querySelectorAll('.tab-nav button');if(btns[2])btns[2].click();})()">Responsible AI</button>
    <button onclick="(function(){var btns=document.querySelectorAll('.tab-nav button');if(btns[3])btns[3].click();})()">Config</button>
  </div>
</div>
<div class="cc-nav-spacer"></div>
"""

_HERO_HTML = """
<div class="cc-hero cc-fade-up">
  <!-- Floating blobs -->
  <div class="cc-blobs">
    <div class="cc-blob cc-blob-1"></div>
    <div class="cc-blob cc-blob-2"></div>
    <div class="cc-blob cc-blob-3"></div>
  </div>
  <!-- Sparkles via inline JS -->
  <script>
  (function(){
    var container = document.currentScript.parentElement;
    for(var i=0;i<14;i++){
      var d=document.createElement('div');
      d.className='cc-sparkle';
      d.style.cssText='width:'+(3+Math.random()*3)+'px;height:'+(3+Math.random()*3)+'px;'
        +'left:'+(5+Math.random()*90)+'%;top:'+(10+Math.random()*80)+'%;'
        +'animation-duration:'+(6+Math.random()*8)+'s;'
        +'animation-delay:'+(Math.random()*8)+'s;';
      container.appendChild(d);
    }
  })();
  </script>
  <!-- Content -->
  <div class="cc-hero-inner">
    <div class="cc-hero-eyebrow cc-fade-up cc-fade-up-1">✨ Generative AI · Story Studio</div>
    <h1 class="cc-hero-h1 cc-fade-up cc-fade-up-2">
      Turn any idea into<br>
      <em>a 6-shot animated story.</em>
    </h1>
    <p class="cc-hero-sub cc-fade-up cc-fade-up-3">
      Describe your concept below. Gemini writes the shots,
      SD-Turbo draws the frames, and BLIP checks the quality — all in one click.
    </p>
  </div>
</div>
"""

_DISCLOSURE_HTML = """
<div class="cc-disclosure cc-fade-up cc-fade-up-4">
  <span style="font-size:1.3rem;flex-shrink:0;">🛡️</span>
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

        # ── Fixed nav ───────────────────────────────────────────────────────
        gr.HTML(_NAV_HTML)

        # ── Hero (outside tabs so it's always visible on the studio page) ──
        # We place it inside Tab 1 instead so other tabs are clean
        with gr.Tabs(elem_classes=["cc-tabs"]):

            # ================================================================
            # TAB 1 · Storyboard Studio
            # ================================================================
            with gr.TabItem("🎬 Studio"):

                # Hero section
                gr.HTML(_HERO_HTML)
                gr.HTML(_DISCLOSURE_HTML)

                # ── Idea input ─────────────────────────────────────────────
                with gr.Column(elem_classes=["cc-section"]):
                    gr.HTML('<div style="height:28px;"></div>')

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
                    with gr.Accordion("🎨  Advanced Settings", open=False):
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

                    # Generate button
                    with gr.Column(elem_classes=["cc-gen-btn"]):
                        generate_btn = gr.Button(
                            "🚀  Generate 6-Shot Story  →", size="lg"
                        )

                # ── Pipeline progress strip ────────────────────────────────
                with gr.Column(elem_classes=["cc-section"]):
                    pipeline_progress_html = gr.HTML(make_pipeline_html())

                # ── Status + audio ─────────────────────────────────────────
                with gr.Row(elem_classes=["cc-section"]):
                    with gr.Column(scale=7):
                        status_markdown = gr.Markdown(
                            "💡 *Enter a story idea above and press **Generate** to begin.*"
                        )
                    with gr.Column(scale=3):
                        audio_player = gr.Audio(
                            label="🎵 Background Music",
                            type="filepath",
                        )

                gr.HTML('<div class="cc-section-gap"></div>')

                # ── Results: 3 pastel cards ────────────────────────────────
                with gr.Column(elem_classes=["cc-section"]):
                    gr.HTML('<div class="cc-results-heading">✨ Your Results</div>')

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

                # ── Eval table (full width) ────────────────────────────────
                with gr.Column(elem_classes=["cc-section"]):
                    gr.HTML(
                        '<div style="font-size:0.95rem;color:var(--muted);margin-bottom:12px;">'
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
                        '<p style="color:#6B7280;font-size:1rem;margin-bottom:28px;">'
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
                        '<p style="color:#6B7280;font-size:1rem;margin-bottom:28px;">'
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

if __name__ == "__main__":
    demo = build_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Base(),
        css=CUSTOM_CSS,
    )
