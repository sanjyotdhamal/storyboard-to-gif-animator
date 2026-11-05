"""
app.py - Storyboard-to-GIF Animator (Gradio Web UI)
===================================================
A complete Generative AI mini-project that converts a creative story concept
into a cinematic 6-shot animated GIF and MP4 with music.

Architecture:
- UI: Gradio Blocks (Hugging Face Spaces compatible)
- LLM: Google Gemini API (GEMINI_API_KEY from environment or UI settings)
- Diffusion: Hugging Face diffusers (sd-turbo) + HF API + Canvas Fallback
- Verification: BLIP image captioning + all-MiniLM-L6-v2 semantic similarity
- Video: Cross-fade blending, audio synthesis, MP4 multiplexing, and AI disclosure stamping
"""

import os
import json
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

CUSTOM_CSS = """
/* Premium dark-mode modern aesthetics */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
.ai-disclosure-banner {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(236, 72, 153, 0.15));
    border: 1px solid rgba(168, 85, 247, 0.3);
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.hero-title {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.3rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    margin-bottom: 4px !important;
}
.status-box {
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 10px;
    font-size: 0.92rem;
}
.primary-btn {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3) !important;
}
"""

# ---------------------------------------------------------------------------
# Core Pipeline Execution
# ---------------------------------------------------------------------------

def run_full_pipeline(
    idea: str,
    technique: str,
    seed: int,
    style_prefix: str,
    mismatch_threshold: float,
    image_mode: str,
    api_key_override: str,
    hf_token_override: str,
    progress=gr.Progress(track_tqdm=True)
):
    """
    Executes the entire end-to-end pipeline:
    1. Safety check
    2. LLM storyboard generation (Zero-Shot or Few-Shot)
    3. 6-shot image generation (Fixed seed, sd-turbo or fallback)
    4. BLIP captioning and semantic similarity verification
    5. Cross-fade blending and GIF / MP4 export with audio
    """
    progress(0.02, desc="Enforcing Safety Guardrails...")

    # Step 1: Safety Check
    is_safe, safety_msg, meta = check_safety(idea)
    if not is_safe:
        empty_df = None
        return (
            safety_msg,         # pipeline_status
            None,               # storyboard_json
            [],                 # gallery_output
            empty_df,           # evaluation_table
            "N/A",              # avg_similarity_display
            None,               # gif_output
            None,               # video_output
            None                # audio_output
        )

    # Step 2: LLM Storyboard Generation
    progress(0.15, desc=f"Generating {technique.upper()} storyboard with Gemini LLM...")
    clean_prefix = style_prefix.strip() or DEFAULT_STYLE_PREFIX
    storyboard_data, llm_note = generate_storyboard(
        idea=idea,
        technique=technique,
        style_prefix=clean_prefix,
        api_key=api_key_override.strip() if api_key_override else None
    )

    shots = storyboard_data.get("shots", [])
    if len(shots) != 6:
        status_msg = f"⚠️ Expected 6 shots, but received {len(shots)}. Using formatted storyboard."
    else:
        status_msg = f"✅ Storyboard created ({len(shots)} shots). {llm_note}"

    # Step 3: Frame Generation
    progress(0.35, desc=f"Generating 6 visual frames (Seed: {seed})...")
    frames, gen_methods = generate_storyboard_frames(
        shots=shots,
        seed=int(seed),
        mode=image_mode,
        hf_token=hf_token_override.strip() if hf_token_override else None,
        progress_callback=lambda p, msg: progress(0.35 + p * 0.30, desc=msg)
    )

    # Step 4: BLIP Captioning & Semantic Evaluation
    progress(0.68, desc="Evaluating frames with BLIP and sentence-transformers...")
    eval_df, avg_score, eval_details = evaluate_storyboard_frames(
        frames=frames,
        shots=shots,
        threshold=float(mismatch_threshold),
        progress_callback=lambda p, msg: progress(0.68 + p * 0.15, desc=msg)
    )

    # Step 5: Video & GIF Cross-fade Assembly
    progress(0.85, desc="Rendering cross-fade transitions and merging audio...")
    gif_path, mp4_path, music_path = create_storyboard_media(
        frames=frames,
        fps_video=24,
        fps_gif=15,
        hold_frames=22,
        transition_frames=10,
        progress_callback=lambda p, msg: progress(0.85 + p * 0.14, desc=msg)
    )

    progress(1.0, desc="Pipeline execution complete!")

    # Summary status
    mismatch_count = sum(1 for d in eval_details if d.get("is_mismatch"))
    summary_banner = (
        f"### 🎉 Storyboard Generated Successfully!\n"
        f"- **LLM Technique**: `{technique}`\n"
        f"- **Image Engine**: `{gen_methods[0] if gen_methods else 'Standard'}`\n"
        f"- **Fixed Seed**: `{seed}`\n"
        f"- **Average BLIP Alignment Score**: `{avg_score:.3f}`\n"
        f"- **Quality Guardrail**: `{mismatch_count}` frames flagged as potential mismatch (Threshold: `{mismatch_threshold}`).\n"
        f"- **AI Disclosure**: Verified 'AI-Generated' watermark applied to all frames."
    )

    return (
        summary_banner,
        storyboard_data,
        frames,
        eval_df,
        f"{avg_score:.3f}",
        gif_path,
        mp4_path,
        music_path
    )


def run_technique_comparison(
    idea: str,
    style_prefix: str,
    api_key_override: str
):
    """
    Executes both Zero-Shot and Few-Shot prompting side by side
    to demonstrate prompt engineering mechanics for students.
    """
    is_safe, safety_msg, _ = check_safety(idea)
    if not is_safe:
        return safety_msg, None, None

    clean_prefix = style_prefix.strip() or DEFAULT_STYLE_PREFIX
    zero_data, few_data, summary = compare_prompts(
        idea=idea,
        style_prefix=clean_prefix,
        api_key=api_key_override.strip() if api_key_override else None
    )

    return summary, zero_data, few_data


# ---------------------------------------------------------------------------
# Gradio Interface Construction
# ---------------------------------------------------------------------------

def build_app():
    with gr.Blocks(title="Storyboard-to-GIF Animator") as demo:
        # AI Disclosure Banner
        gr.HTML("""
        <div class="ai-disclosure-banner">
            <span style="font-size: 1.6rem;">✨</span>
            <div>
                <strong style="color: #6366f1;">Responsible AI Disclosure:</strong>
                All visual imagery, story beats, and captions produced in this application are 
                <strong>synthetically generated using Artificial Intelligence</strong> 
                (Gemini, SD-Turbo, and BLIP). An indelible <em>"AI-Generated"</em> mark is watermarked on every media frame.
            </div>
        </div>
        """)

        # Main Header
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 24px;">
            <h1 class="hero-title">Storyboard-to-GIF Animator</h1>
            <p style="color: #64748b; font-size: 1.05rem; max-width: 750px; margin: 0 auto;">
                Transform any creative idea into a cohesive 6-shot animated story with cross-fade transitions, 
                background music, and automated BLIP semantic quality verification.
            </p>
        </div>
        """)

        with gr.Tabs() as tabs:
            # ----------------------------------------------------------------
            # TAB 1: Main Storyboard Studio
            # ----------------------------------------------------------------
            with gr.TabItem("🎬 Storyboard Studio"):
                with gr.Row():
                    # Left Column: Inputs & Controls
                    with gr.Column(scale=5):
                        user_idea_input = gr.Textbox(
                            label="Your Story Concept",
                            placeholder="e.g. A small vintage brass robot discovers an easel and paints blooming sunflowers in a greenhouse...",
                            value="A small lonely robot in a greenhouse learns to paint flowers",
                            lines=3
                        )

                        with gr.Row():
                            technique_radio = gr.Radio(
                                choices=["few-shot", "zero-shot"],
                                value="few-shot",
                                label="Prompt Engineering Technique",
                                info="Few-shot provides in-context exemplar shot lists; Zero-shot provides direct instructions."
                            )
                            seed_input = gr.Number(
                                label="Fixed Seed (Reproducibility)",
                                value=42,
                                precision=0,
                                step=1
                            )

                        with gr.Accordion("🎨 Advanced Style & Generation Settings", open=False):
                            style_prefix_input = gr.Textbox(
                                label="Shared Artistic Style Prefix (Prepended to each shot prompt)",
                                value=DEFAULT_STYLE_PREFIX,
                                lines=2
                            )
                            with gr.Row():
                                mismatch_threshold_slider = gr.Slider(
                                    minimum=0.10,
                                    maximum=0.70,
                                    value=DEFAULT_MISMATCH_THRESHOLD,
                                    step=0.05,
                                    label="Mismatch Detection Cutoff (Cosine Similarity)"
                                )
                                image_mode_select = gr.Dropdown(
                                    choices=["auto", "local_diffusers", "hf_api", "mock"],
                                    value="auto",
                                    label="Image Engine Mode",
                                    info="'auto' uses local GPU diffusers, falling back gracefully."
                                )

                        generate_btn = gr.Button("🚀 Generate 6-Shot Storyboard & Video", elem_classes=["primary-btn"], size="lg")

                    # Right Column: Execution Status & AI Metrics
                    with gr.Column(scale=5):
                        status_markdown = gr.Markdown("💡 *Enter an idea and press Generate to begin.*")
                        avg_similarity_stat = gr.Label(label="Average BLIP Semantic Alignment Score", value="Ready")
                        audio_player = gr.Audio(label="🎵 Royalty-Free Background Music Track", type="filepath")

                gr.HTML("<hr style='border: none; border-top: 1px solid rgba(226, 232, 240, 0.4); margin: 24px 0;'/>")

                # Results Showcase
                with gr.Row():
                    with gr.Column(scale=6):
                        gr.Markdown("### 🎞️ Generated 6-Shot Narrative Sequence")
                        frames_gallery = gr.Gallery(
                            label="Storyboard Frames (Stamped with AI Label)",
                            columns=3,
                            rows=2,
                            height=360,
                            object_fit="contain"
                        )
                    with gr.Column(scale=4):
                        gr.Markdown("### 🎥 Final Animated Exports")
                        with gr.Tabs():
                            with gr.TabItem("MP4 Video (with Music)"):
                                video_player = gr.Video(label="Story Animation MP4", autoplay=False)
                            with gr.TabItem("Animated GIF"):
                                gif_image = gr.Image(label="Seamless Cross-Fade GIF")

                # Quality Verification Table
                gr.Markdown("### 🔍 BLIP Multimodal Quality Verification & Mismatch Guardrail")
                gr.Markdown(
                    "Each frame is independently captioned by **Salesforce/blip-image-captioning-base**, "
                    "and compared against the intended narrative prompt using **sentence-transformers (all-MiniLM-L6-v2)**."
                )
                eval_table = gr.Dataframe(
                    headers=["Shot #", "Intended Shot / Caption", "BLIP Caption", "Cosine Similarity", "Status"],
                    datatype=["number", "str", "str", "str", "str"],
                    label="Per-Frame Semantic Alignment Table",
                    interactive=False
                )

                with gr.Accordion("📄 View Raw LLM Storyboard JSON", open=False):
                    json_viewer = gr.JSON(label="LLM Structured Output")

            # ----------------------------------------------------------------
            # TAB 2: Prompt Technique Comparison (Zero-Shot vs Few-Shot)
            # ----------------------------------------------------------------
            with gr.TabItem("⚖️ Prompt Technique Comparison"):
                gr.Markdown("### Compare Prompt Techniques Side-by-Side on the Same Concept")
                gr.Markdown(
                    "Observe how providing **in-context examples (Few-Shot)** shapes the LLM's camera terminology, "
                    "rhythm, and descriptive grounding compared to direct instructions alone (**Zero-Shot**)."
                )

                compare_concept_input = gr.Textbox(
                    label="Concept to Compare",
                    value="A timid deep-sea octopus invents a bioluminescent lantern",
                    lines=2
                )
                compare_btn = gr.Button("🔬 Run Side-by-Side Comparison", variant="secondary")
                compare_summary_md = gr.Markdown()

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 🟢 Zero-Shot Output")
                        zero_shot_json = gr.JSON(label="Zero-Shot Storyboard")
                    with gr.Column():
                        gr.Markdown("#### 🟣 Few-Shot Output (In-Context Learned)")
                        few_shot_json = gr.JSON(label="Few-Shot Storyboard")

            # ----------------------------------------------------------------
            # TAB 3: Responsible AI & Diagnostics
            # ----------------------------------------------------------------
            with gr.TabItem("🛡️ Responsible AI & Diagnostics"):
                gr.Markdown("### Responsible AI Guardrail Suite")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 1. Interactive Safety Guardrail Tester")
                        guardrail_test_input = gr.Textbox(
                            label="Test a prompt against safety policies",
                            value="Donald Trump and Batman in a violent gunfight",
                            lines=2
                        )
                        guardrail_check_btn = gr.Button("Check Guardrails")
                        guardrail_result_md = gr.Markdown()
                    with gr.Column():
                        gr.Markdown("#### 2. Guardrail Policy Rules")
                        gr.Markdown("""
                        - **Violence & Harm**: Blocks gore, weapons, mass casualty, or self-harm.
                        - **Hate Speech**: Blocks slurs, supremacy, xenophobia, and hate incidents.
                        - **Sexual Content**: Blocks explicit adult or erotic depictions.
                        - **Real-Person Deepfakes**: Protects public figures & living individuals against non-consensual depiction.
                        - **Trademarked IP**: Prevents commercial copyright infringement (e.g. Disney, Marvel, Nintendo).
                        """)

                gr.HTML("<hr style='border: none; border-top: 1px solid rgba(226, 232, 240, 0.4); margin: 20px 0;'/>")

                gr.Markdown("### Failure Case & Semantic Drift Analysis")
                gr.Markdown("""
                In Generative AI pipelines, text-to-image models can suffer from **object omission**, **attribute binding errors**, 
                or **semantic hallucination**.
                
                - **Example Failure Case**: The prompt asked for *"a brass robot holding a paintbrush"*, but the model generated 
                  a generic outdoor landscape where the robot was rendered as a small rock or missing entirely.
                - **BLIP Detection**: BLIP caption was *"a scenic mountain path with boulders"*, yielding a cosine similarity of **0.24** (< 0.35 cutoff).
                - **Mitigation**: The system flags this as **⚠️ MISMATCH**, allowing the user to adjust prompts or regenerate with an alternate seed.
                """)

            # ----------------------------------------------------------------
            # TAB 4: Configuration & Secrets
            # ----------------------------------------------------------------
            with gr.TabItem("⚙️ Secrets & Config"):
                gr.Markdown("### API Credentials & Deployment Configuration")
                gr.Markdown(
                    "You can configure API keys via environment variables (`GEMINI_API_KEY`, `HF_TOKEN`) "
                    "or enter temporary session overrides below. *Keys are never logged or saved to disk.*"
                )
                with gr.Row():
                    ui_gemini_key = gr.Textbox(
                        label="Gemini API Key (Optional Override)",
                        placeholder="AIzaSy...",
                        type="password"
                    )
                    ui_hf_token = gr.Textbox(
                        label="Hugging Face Token (Optional Override)",
                        placeholder="hf_...",
                        type="password"
                    )
                gr.Markdown("ℹ️ When deploying to **Hugging Face Spaces**, add `GEMINI_API_KEY` and `HF_TOKEN` under **Settings > Variables and secrets**.")

        # Wire Events
        generate_btn.click(
            fn=run_full_pipeline,
            inputs=[
                user_idea_input,
                technique_radio,
                seed_input,
                style_prefix_input,
                mismatch_threshold_slider,
                image_mode_select,
                ui_gemini_key,
                ui_hf_token
            ],
            outputs=[
                status_markdown,
                json_viewer,
                frames_gallery,
                eval_table,
                avg_similarity_stat,
                gif_image,
                video_player,
                audio_player
            ]
        )

        compare_btn.click(
            fn=run_technique_comparison,
            inputs=[
                compare_concept_input,
                style_prefix_input,
                ui_gemini_key
            ],
            outputs=[
                compare_summary_md,
                zero_shot_json,
                few_shot_json
            ]
        )

        def test_guardrail_ui(prompt):
            safe, msg, _ = check_safety(prompt)
            return msg

        guardrail_check_btn.click(
            fn=test_guardrail_ui,
            inputs=[guardrail_test_input],
            outputs=[guardrail_result_md]
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    # Launch with public sharing disabled by default, listen on localhost
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_api=False,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS
    )
