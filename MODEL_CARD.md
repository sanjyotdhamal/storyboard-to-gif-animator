# Model Card: Storyboard-to-GIF Animator Pipeline

This document details the machine learning models integrated into the **Storyboard-to-GIF Animator** system, their intended purposes, technical configurations, ethical considerations, and licensing.

---

## 1. Overview of Models in the Pipeline

| Pipeline Stage | Model Name | Model Family / Architecture | Primary Role | Source / Provider | License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Storyboard LLM** | `gemini-1.5-flash` | Multimodal Transformer LLM | Decomposes story idea into 6 structured cinematic narrative beats (JSON) | Google DeepMind | Google Gemini API Terms |
| **Image Generation** | `stabilityai/sd-turbo` | Adversarial Latent Diffusion (SDXL distilled) | Generates 512x512 visual story frames in 1–2 inference steps | Stability AI / Hugging Face | Stability AI Non-Commercial Research / Community |
| **Frame Captioning** | `blip-image-captioning-base` | Vision-Language Transformer (ViT + Text Decoder) | Generates objective natural language captions describing visual frame contents | Salesforce Research | BSD 3-Clause |
| **Semantic Scoring** | `all-MiniLM-L6-v2` | BERT-family MiniLM Transformer (Bi-Encoder) | Computes 384-dimensional dense semantic vectors and cosine similarity | Hugging Face / Microsoft | Apache 2.0 |

---

## 2. Intended Use & Target Users

- **Intended Use**: Educational exploration of generative storytelling, prompt engineering techniques (zero-shot vs. few-shot), automated multimodal quality verification, and responsible AI safety guardrails.
- **Target Audience**: Students, researchers, and developers studying Generative AI pipelines and automated alignment mechanisms.
- **Out-of-Scope Uses**:
  - Commercial production without licensing review.
  - Generating political disinformation, impersonations, deepfakes, or copyrighted character infringements.
  - Safety-critical or automated surveillance applications.

---

## 3. Detailed Model Breakdown

### A. Google Gemini 1.5 Flash (LLM Narrative Planner)
- **Role**: Takes a high-level creative seed (e.g., *"a robot learns to paint"*) and plans a 6-beat narrative arc:
  1. Introduction / Establishing shot
  2. Inciting incident / Discovery
  3. Rising action / Initial attempt
  4. Turning point / Dramatic tension
  5. Climax / Creative breakthrough
  6. Heartwarming resolution
- **Techniques Supported**: Zero-Shot (zero reference examples) and Few-Shot (in-context exemplars).
- **Style Consistency Enforcement**: Automatically incorporates a user-configurable style prefix (e.g., *digital storybook illustration, vibrant cinematic lighting*) across every shot.

### B. Stability AI SD-Turbo (Adversarial Diffusion Frame Generator)
- **Role**: Synthesizes 512×512 resolution frames with low latency using Adversarial Diffusion Distillation (ADD).
- **Inference Profile**: Generates coherent images in only 1 to 2 denoising steps (guidance scale = 0.0), making it ideal for interactive web environments.
- **Reproducibility**: Enforces a user-configurable fixed seed (default `42`) to demonstrate algorithmic determinism in generative workflows.
- **Fallback Options**:
  - Hugging Face Inference API for CPU environments.
  - High-aesthetic Canvas Synthetic Renderer for instant offline unit testing.

### C. Salesforce BLIP (`blip-image-captioning-base`)
- **Role**: Acts as an objective multimodal "judge." It views the synthesized image without prior knowledge of the prompt and generates an independent descriptive caption.
- **Purpose**: Evaluates whether the diffusion model actually rendered what was requested, or if it suffered from object omission or background hallucinations.

### D. Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Role**: Encodes both the BLIP visual caption and the intended shot prompt into dense 384-dimensional semantic embeddings.
- **Metric**: Computes the Cosine Similarity:
  $$\text{Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
- **Threshold**: Flagged as `⚠️ MISMATCH` if the score drops below the threshold (default: `0.35`).

---

## 4. Limitations, Biases & Hallucinations

1. **Character Consistency**: Diffusion models without fine-tuned subject checkpoints (like DreamBooth or LoRA) will display slight visual variations in character clothing, exact facial features, or robotic joints across consecutive shots.
2. **Adjective & Attribute Binding**: Diffusion models may occasionally misbind colors or objects (e.g., painting the robot yellow instead of the flower).
3. **BLIP Domain Sensitivity**: BLIP is trained primarily on natural photographic datasets (COCO, Conceptual Captions). When captioning abstract or stylized storybook frames, it may use generic terms (e.g., calling an easel *"a wooden table"*).
4. **Safety Filter Boundaries**: The rule-based safety guardrail balances precision and recall; highly metaphorical negative language may require future neural classification additions.

---

## 5. Responsible AI Disclosures & Labelling

- **Visual Watermark**: Every frame rendered by the pipeline is stamped with an explicit, high-contrast banner: `✨ AI-Generated | Storyboard-to-GIF`.
- **UI Disclosure**: The Gradio web interface displays a persistent Responsible AI notice advising users that all media is synthetic.
- **Audio Licensing**: Procedural ambient audio loop generated via clean harmonic synthesis and released under **CC0 1.0 Universal (Public Domain)**.
