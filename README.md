# 🎬 Storyboard-to-GIF Animator

> **Course**: Generative AI Mini-Project  
> An end-to-end multimodal Generative AI application that turns a creative story idea into a cohesive 6-shot animated GIF and MP4 with background music, automated semantic alignment verification via BLIP, and strict Responsible AI guardrails.

---

## ✨ Features & Architecture

- **Narrative Decomposition**: Uses **Google Gemini 1.5 Flash** to break any creative concept into 6 cinematic narrative beats (Introduction, Inciting Incident, Rising Action, Turning Point, Climax, Resolution).
- **Prompt Engineering**: Demonstrates and compares **Zero-Shot** vs. **Few-Shot (In-Context)** prompting side-by-side.
- **Visual Style Consistency**: Injects a shared style prefix into every shot prompt for visual coherence.
- **Fast Frame Synthesis**: Generates 6 frames at 512×512 using **PyTorch + Hugging Face Diffusers (`stabilityai/sd-turbo`)** with fixed seed reproducibility (`seed=42`).
- **Resilient Fallbacks**: Automatic fallback to **Hugging Face Inference API** or an offline aesthetic canvas renderer if GPU/connectivity is unavailable.
- **Multimodal Quality Auditing**: Employs **Salesforce BLIP** (`blip-image-captioning-base`) to independently caption each generated frame, and **Sentence-Transformers** (`all-MiniLM-L6-v2`) to compute semantic cosine similarity against intended prompts. Flags semantic mismatches below cutoff.
- **Cinematic Media Assembly**: Smooth alpha-blended cross-fading (~10 transition frames per shot) exported as an animated **GIF** and **MP4**.
- **Audio Synthesis & Synchronization**: Accompanied by an ambient music track, synthesized or loaded from `/assets`.
- **Responsible AI Watermarking**: Stamped visible `✨ AI-Generated` watermark badge on every frame of the exported media and persistent disclosure in the UI.

---

## 🛠️ Tech Stack

| Component | Library / Framework | Model / Engine |
| :--- | :--- | :--- |
| **User Interface** | Gradio 4.x (Blocks) | Python Web App |
| **LLM Storyboard** | `google-generativeai` | Gemini 1.5 Flash |
| **Text-to-Image** | `diffusers`, `torch` | `stabilityai/sd-turbo` |
| **Image Captioning** | `transformers` | `Salesforce/blip-image-captioning-base` |
| **Semantic Similarity** | `sentence-transformers` | `all-MiniLM-L6-v2` |
| **Media & Audio** | `Pillow`, `imageio`, `moviepy`, `wave` | Cross-fade interpolation & AAC/MP4 |

---

## 🚀 Quickstart: Running Locally

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 (Recommended).
- (Optional) NVIDIA GPU with CUDA for local `sd-turbo` acceleration.

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-username/storyboard-to-gif-animator.git
cd storyboard-to-gif-animator

# Create and activate virtual environment
python -m venv venv

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Secrets (.env)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` and insert your API keys:
```env
# Google Gemini API Key (Required for LLM generation)
# Get your free key at: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Hugging Face Access Token (Optional: for HF Cloud Inference fallback)
# Get your token at: https://huggingface.co/settings/tokens
HF_TOKEN=your_hf_token_here

# Image generation mode: 'auto', 'local_diffusers', 'hf_api', or 'mock'
IMAGE_GEN_MODE=auto

# Cosine similarity mismatch threshold
MISMATCH_THRESHOLD=0.35
```

> **Note**: If `GEMINI_API_KEY` is omitted, the app will run in **Educational Fallback Mode**, providing deterministic sample storyboards so you can still test the entire UI and video pipeline offline without an API key!

### 4. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:7860
```

---

## ☁️ Deploying on Hugging Face Spaces

You can deploy this project to Hugging Face Spaces in under 3 minutes:

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Select **Gradio** as the Space SDK and choose **Python 3.10 / 3.11**.
3. Choose hardware:
   - **T4 Small GPU** (Recommended for fast local `sd-turbo` inference), OR
   - **Free CPU Tier** (Set `IMAGE_GEN_MODE=hf_api` or `mock` in Space secrets).
4. In your Space repository, upload all files (`app.py`, `safety.py`, `llm.py`, `imagegen.py`, `blip_check.py`, `video.py`, `requirements.txt`, `assets/`, `results/`, `docs/`).
5. Configure Secrets in Spaces:
   - Navigate to **Settings** > **Variables and secrets**.
   - Under **Secrets**, add:
     - `GEMINI_API_KEY`: Your Gemini API key.
     - `HF_TOKEN`: Your Hugging Face read token.
6. The Space will build automatically and launch at your public URL!

---

## 🛡️ Responsible AI Policies & Guardrails

1. **Safety Filter**: Proactively blocks harmful prompt categories before LLM invocation:
   - Violence, gore, weapons, and self-harm.
   - Hate speech, slurs, and harassment.
   - Sexually explicit and adult themes.
   - Real-person deepfakes (living public figures and political leaders).
   - Trademarked / commercial IP characters (e.g., Marvel, Disney, Nintendo).
2. **Quality Auditing via BLIP**: Captions generated frames and scores prompt fidelity using BERT cosine similarity to flag semantic drift, object omission, or hallucination.
3. **Disclosure Stamping**: Stamped banner `✨ AI-Generated | Storyboard-to-GIF` permanently affixed to every exported video frame.

---

## 📜 Intellectual Property & Licenses

- **Code**: Licensed under the **MIT License**.
- **SD-Turbo Weights**: Stability AI Non-Commercial Research Community License.
- **BLIP Weights**: BSD 3-Clause License (Salesforce Research).
- **all-MiniLM-L6-v2**: Apache 2.0 License.
- **Audio Assets**: The procedural ambient track (`assets/ambient_story_theme.wav`) is synthesized procedurally and dedicated to the **CC0 1.0 Universal Public Domain**.
- **Generated Media**: All synthesized output images and videos are synthetic artifacts intended strictly for educational coursework and non-commercial research.
