"""
llm.py - Storyboard Generation with Google GenAI SDK & Prompt Engineering
========================================================================
Generates a coherent 6-shot visual storyboard from a user's text concept.

Key Concepts Demonstrated:
1. Zero-Shot Prompting: Providing direct task instructions without prior examples.
2. Few-Shot Prompting: Providing high-quality in-context exemplars to guide
   narrative rhythm, camera terminology, and visual consistency.
3. Structured Output: Generating strictly formatted JSON containing 6 distinct narrative beats.
4. Shared Style Prefix: Prepended to every frame prompt to maintain artistic coherence across the clip.
5. Dynamic Idea-Grounded Fallback: If API encounters errors, builds narrative directly from the user's concept.
"""

import os
import json
import re
import time
from typing import Dict, Any, List, Tuple, Optional
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

DEFAULT_STYLE_PREFIX = (
    "digital storybook illustration, vibrant cinematic lighting, soft warm palette, "
    "expressive character design, sharp focus, consistent visual art style"
)

# ---------------------------------------------------------------------------
# Few-Shot Demonstrations for the LLM
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = """
[EXAMPLE 1]
User Concept: "A lonely robot in a greenhouse learns to paint flowers"
Output JSON:
{
  "title": "The Mechanical Artist",
  "style_prefix": "digital storybook illustration, vibrant cinematic lighting, soft warm palette, expressive character design, sharp focus, consistent visual art style",
  "shots": [
    {
      "shot_number": 1,
      "camera": "Wide establishing shot",
      "image_prompt": "digital storybook illustration, vibrant cinematic lighting, soft warm palette, expressive character design, sharp focus, consistent visual art style, a rusty vintage brass robot standing inside an overgrown glass greenhouse filled with ferns and sunlight filtering through cracked panes",
      "caption": "In an abandoned sunlit greenhouse, Rustle the robot tended to silent plants."
    },
    {
      "shot_number": 2,
      "camera": "Over-the-shoulder medium shot",
      "image_prompt": "digital storybook illustration, vibrant cinematic lighting, soft warm palette, expressive character design, sharp focus, consistent visual art style, the brass robot discovers an old wooden painter's easel and dusty tubes of vivid oil paint tucked behind flowerpots",
      "caption": "One afternoon, Rustle uncovered a forgotten artist's palette and colorful pigments."
    },
    {
      "shot_number": 3,
      "camera": "Close-up macro shot",
      "image_prompt": "digital storybook illustration, vibrant cinematic lighting, soft warm palette, expressive character design, sharp focus, consistent visual art style, close-up of a metallic robotic finger gently dabbing bright turquoise paint onto a canvas, trembling with mechanical curiosity",
      "caption": "With gentle gears turning, Rustle placed the first hesitant stroke on the canvas."
    },
    {
      "shot_number": 4,
      "camera": "Low angle dramatic shot",
      "image_prompt": "digital storybook illustration, vibrant cinematic lighting, soft warm palette, expressive character design, sharp focus, consistent visual art style, dynamic view of the robot enthusiastically mixing red and yellow paint on a wooden board, splatters of color surrounding the workspace",
      "caption": "Colors swirled as the robot studied the curves and petals of blooming orchids."
    },
    {
      "shot_number": 5,
      "camera": "Three-quarter cinematic shot",
      "image_prompt": "digital storybook illustration, vibrant cinematic lighting, soft warm palette, expressive character design, sharp focus, consistent visual art style, the robot standing back proudly admiring a finished vibrant canvas of glowing sunflowers, surrounded by curious butterflies fluttering in the greenhouse",
      "caption": "Before Rustle stood a luminous masterpiece, glowing like real morning light."
    },
    {
      "shot_number": 6,
      "camera": "Wide heartwarming resolution shot",
      "image_prompt": "digital storybook illustration, vibrant cinematic lighting, soft warm palette, expressive character design, sharp focus, consistent visual art style, wide shot of the greenhouse adorned with dozens of colorful painted canvases on the walls, sunlight streaming down as the robot smiles with glowing blue optic eyes",
      "caption": "The quiet greenhouse had transformed into a sanctuary of life and art."
    }
  ]
}

[EXAMPLE 2]
User Concept: "A small adventurous turtle finds a lost celestial star"
Output JSON:
{
  "title": "The Star Bearer",
  "style_prefix": "digital storybook illustration, vibrant cinematic lighting, soft warm palette, expressive character design, sharp focus, consistent visual art style",
  "shots": [
    {
      "shot_number": 1,
      "camera": "Wide shoreline shot at dusk",
      "image_prompt": "digital storybook illustration, vibrant cinematic lighting, soft warm palette, expressive character design, sharp focus, consistent visual art style, a tiny sea turtle with a patterned mossy shell crawling along a purple twilight beach under twinkling constellations",
      "caption": "Pip the turtle loved listening to the whispered lullabies of the night tide."
    },
    {
      "shot_number": 2,
      "camera": "Close-up discovery shot",
      "image_prompt": "digital storybook illustration, vibrant cinematic lighting, soft warm palette, expressive character design, sharp focus, consistent visual art style, a tiny glowing fallen baby star resting inside a mother-of-pearl seashell, illuminating the turtle's wide sparkling eyes",
      "caption": "Tucked inside a spiral shell lay a fallen star, shimmering like honey and gold."
    },
    {
      "shot_number": 3,
      "camera": "Medium companionship shot",
      "image_prompt": "digital storybook illustration, vibrant cinematic lighting, soft warm palette, expressive character design, sharp focus, consistent visual art style, the little turtle carrying the glowing star safely on top of its shell as it paddles into phosphorescent bioluminescent ocean waves",
      "caption": "Pip carried the trembling star upon its shell into the glowing ocean depths."
    },
    {
      "shot_number": 4,
      "camera": "Underwater wide adventure shot",
      "image_prompt": "digital storybook illustration, vibrant cinematic lighting, soft warm palette, expressive character design, sharp focus, consistent visual art style, deep underwater reef with glowing coral arches, schools of translucent jellyfish guiding the turtle towards an underwater mountain peak",
      "caption": "Guiding spirits of the deep illuminated their journey to the Highest Reef."
    },
    {
      "shot_number": 5,
      "camera": "Climactic ascending shot",
      "image_prompt": "digital storybook illustration, vibrant cinematic lighting, soft warm palette, expressive character design, sharp focus, consistent visual art style, the turtle reaching the peak of an oceanic coral spire, the star bursting into radiant golden rays as it begins to levitate into the sky",
      "caption": "At the highest peak, the star gathered warmth and began to float skyward."
    },
    {
      "shot_number": 6,
      "camera": "Cosmic resolution wide shot",
      "image_prompt": "digital storybook illustration, vibrant cinematic lighting, soft warm palette, expressive character design, sharp focus, consistent visual art style, wide shot of the ocean meeting the night sky, the bright star restored in the galaxy above, sending a beam of sparkling light down to the smiling turtle on the water",
      "caption": "Pip looked up happily as his little friend shone bright in the northern constellation."
    }
  ]
}
"""


def _clean_json_response(raw_text: str) -> Dict[str, Any]:
    """Extracts and parses JSON object from an LLM response string."""
    # Find markdown JSON blocks or bare json brackets
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if json_match:
        content = json_match.group(1)
    else:
        json_match_bare = re.search(r"(\{.*\})", raw_text, re.DOTALL)
        content = json_match_bare.group(1) if json_match_bare else raw_text

    return json.loads(content)


def _extract_subject_and_action(idea: str) -> Tuple[str, str]:
    """Extracts a concise subject and scene setting from the user's idea string."""
    clean = idea.strip()
    words = clean.split()
    subject = " ".join(words[:4]) if len(words) >= 4 else clean
    return subject, clean


def _generate_mock_storyboard(
    idea: str,
    technique: str,
    style_prefix: str,
    reason: str = "Fallback mode"
) -> Dict[str, Any]:
    """
    Builds an offline fallback storyboard that is deeply grounded in the user's idea text.
    Every shot explicitly mentions the subject, setting, and action described by the user.
    """
    cleaned_idea = idea.strip() or "A magical adventure"
    subject, full_concept = _extract_subject_and_action(cleaned_idea)

    # Narrative progression grounded in the user's concept
    shots = [
        {
            "shot_number": 1,
            "camera": "Wide establishing cinematic shot",
            "image_prompt": (
                f"{style_prefix}, wide establishing shot of {cleaned_idea}, "
                f"introducing the primary protagonist and the surrounding environment, vibrant atmospheric depth"
            ),
            "caption": f"In the opening scene, {cleaned_idea} begins in a calm, vivid environment."
        },
        {
            "shot_number": 2,
            "camera": "Medium discovery shot",
            "image_prompt": (
                f"{style_prefix}, medium shot featuring {subject}, discovering a hidden opportunity or spark "
                f"directly tied to {full_concept}, expressive eyes and curious posture"
            ),
            "caption": f"A sudden curious discovery draws {subject} into the journey of {full_concept}."
        },
        {
            "shot_number": 3,
            "camera": "Close-up action engagement shot",
            "image_prompt": (
                f"{style_prefix}, close-up focused shot of {subject} actively engaged in {full_concept}, "
                f"hands or paws or mechanical tools in motion, rich textural details and warm directional lighting"
            ),
            "caption": f"With deep focus and care, {subject} takes on the challenge of {full_concept}."
        },
        {
            "shot_number": 4,
            "camera": "Low angle dramatic tension shot",
            "image_prompt": (
                f"{style_prefix}, low angle dramatic shot of {cleaned_idea}, experiencing a crucial turning point, "
                f"dynamic visual tension, swirling color and lighting contrast"
            ),
            "caption": f"At the turning point, {subject} overcomes an unexpected test while pursuing {full_concept}."
        },
        {
            "shot_number": 5,
            "camera": "Climactic high-energy spectacle shot",
            "image_prompt": (
                f"{style_prefix}, climactic cinematic spectacle of {cleaned_idea}, peak visual breakthrough, "
                f"glowing illumination, radiant particles, celebratory atmosphere"
            ),
            "caption": f"In a triumphant breakthrough, {subject} brings {full_concept} to dazzling fruition!"
        },
        {
            "shot_number": 6,
            "camera": "Wide heartwarming resolution shot",
            "image_prompt": (
                f"{style_prefix}, wide panoramic heartwarming shot concluding {cleaned_idea}, "
                f"peaceful golden hour lighting, lasting fulfillment and visual harmony"
            ),
            "caption": f"The story concludes with {cleaned_idea} leaving a joyful, lasting mark on the world."
        }
    ]

    return {
        "title": f"Story of {subject[:35]}",
        "technique": technique,
        "style_prefix": style_prefix,
        "shots": shots,
        "source": "educational_concept_fallback",
        "is_fallback": True,
        "fallback_reason": reason
    }


def _get_candidate_models() -> List[str]:
    """Returns an ordered list of Gemini Flash models to try."""
    env_model = os.getenv("GEMINI_MODEL", "").strip()
    candidates = []
    if env_model:
        candidates.append(env_model)

    # Standard currently supported Gemini Flash models
    defaults = [
        "gemini-3-flash-preview",
        "gemini-3.8-flash",
        "gemini-3.1-flash-lite-preview",
        "gemini-flash-latest"
    ]
    for d in defaults:
        if d not in candidates:
            candidates.append(d)
    return candidates


def generate_storyboard(
    idea: str,
    technique: str = "few-shot",
    style_prefix: str = DEFAULT_STYLE_PREFIX,
    api_key: Optional[str] = None
) -> Tuple[Dict[str, Any], str]:
    """
    Generates a 6-shot visual storyboard using Google GenAI SDK (google.genai).
    Implements model discovery, retry logic, and concept-grounded fallback.
    
    Args:
        idea: The user's prompt / narrative concept.
        technique: 'zero-shot' or 'few-shot'
        style_prefix: Visual style prepended to all image prompts for aesthetic consistency.
        api_key: Optional explicit Gemini API key (defaults to GEMINI_API_KEY from .env).
        
    Returns:
        Tuple of (storyboard_dict, status_message)
    """
    active_key = api_key or os.getenv("GEMINI_API_KEY")

    # If no key configured, fallback with explicit warning
    if not active_key or active_key.strip() == "" or "your_gemini_api_key" in active_key:
        err_msg = "No GEMINI_API_KEY configured in environment or UI settings."
        print(f"[LLM WARNING] {err_msg}")
        fallback_data = _generate_mock_storyboard(idea, technique, style_prefix, reason=err_msg)
        return fallback_data, f"⚠️ Gemini API key missing. Generated concept-grounded fallback storyboard."

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=active_key.strip())
        candidate_models = _get_candidate_models()

        system_instruction = (
            "You are a professional animation director and visual storyboard artist. "
            "Your task is to take a user story concept and produce a visually compelling, "
            "coherent 6-shot animated story sequence. Every shot must strictly follow a 6-beat narrative arc:\n"
            "Beat 1: Establishing the world and protagonist\n"
            "Beat 2: Inciting incident / curious discovery\n"
            "Beat 3: First attempt / hands-on engagement\n"
            "Beat 4: Turning point / emotional intensity\n"
            "Beat 5: Visual climax / breakthrough\n"
            "Beat 6: Heartwarming resolution / lasting impact\n\n"
            "CRITICAL FORMAT RULES:\n"
            "1. Output MUST be strictly valid JSON with keys 'title' and 'shots'. No preamble or conversational markdown outside JSON.\n"
            "2. Exactly 6 shots in an array named 'shots'.\n"
            f"3. Every shot's 'image_prompt' MUST begin with the exact shared style prefix: '{style_prefix}' "
            "followed by a vivid visual scene description that explicitly includes the user concept's protagonist, action, and setting.\n"
            "4. Each shot must include: 'shot_number' (1-6), 'camera' (e.g. Wide shot, Close-up macro), 'image_prompt', and 'caption' (1 clear narrative sentence)."
        )

        if technique.lower() == "few-shot":
            user_prompt = (
                f"{system_instruction}\n\n"
                f"### IN-CONTEXT EXAMPLES FOR GUIDANCE:\n{FEW_SHOT_EXAMPLES}\n\n"
                f"### NOW GENERATE THE STORYBOARD FOR THIS CONCEPT:\n"
                f"User Concept: \"{idea}\"\n"
                f"Output JSON:"
            )
        else:  # zero-shot
            user_prompt = (
                f"{system_instruction}\n\n"
                f"User Concept: \"{idea}\"\n"
                f"Output JSON:"
            )

        last_error = None
        response_text = None
        successful_model = None

        # Try models in order with exponential backoff on transient errors
        for model_name in candidate_models:
            print(f"[LLM] Attempting generation with model: '{model_name}'...")
            for attempt in range(1, 4):
                try:
                    resp = client.models.generate_content(
                        model=model_name,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.7,
                        )
                    )
                    if resp and resp.text:
                        response_text = resp.text
                        successful_model = model_name
                        break
                except Exception as call_err:
                    err_str = str(call_err)
                    last_error = err_str
                    print(f"[LLM] Attempt {attempt} for '{model_name}' failed: {err_str[:160]}")
                    # If 404 model not found, don't retry same model; break to try next candidate
                    if "404" in err_str or "NOT_FOUND" in err_str:
                        break
                    time.sleep(1.2 * attempt)

            if response_text:
                break

        if not response_text:
            raise RuntimeError(f"All Gemini models exhausted. Last error: {last_error}")

        # Print raw LLM response as required by Requirement 1
        print(f"\n[LLM RAW OUTPUT - Model: {successful_model}]\n{response_text}\n")

        parsed = _clean_json_response(response_text)

        # Ensure style prefix is cleanly prepended and subject is preserved on every shot
        subject, _ = _extract_subject_and_action(idea)
        if "shots" in parsed and isinstance(parsed["shots"], list):
            for i, s in enumerate(parsed["shots"]):
                s["shot_number"] = i + 1
                prompt = s.get("image_prompt", "")
                # Ensure style prefix is present
                if not prompt.startswith(style_prefix[:25]):
                    prompt = f"{style_prefix}, {prompt}"
                # Ensure the user's idea subject is mentioned in the prompt (Requirement 4)
                if subject.lower() not in prompt.lower():
                    prompt = f"{prompt}, featuring {subject}"
                s["image_prompt"] = prompt

        parsed["technique"] = technique
        parsed["style_prefix"] = style_prefix
        parsed["source"] = f"gemini ({successful_model})"
        parsed["is_fallback"] = False
        parsed["model"] = successful_model
        parsed["model_used"] = successful_model

        note = f"✅ Storyboard generated by Gemini ({successful_model})."
        return parsed, note

    except Exception as e:
        error_msg = str(e)
        print(f"[LLM ERROR] Gemini API call failed: {error_msg}")
        fallback_data = _generate_mock_storyboard(idea, technique, style_prefix, reason=error_msg)
        fallback_data["api_error"] = error_msg
        note = f"⚠️ Gemini failed ({error_msg[:120]}), using offline fallback. Results are NOT based on live Gemini output."
        return fallback_data, note


def compare_prompts(
    idea: str,
    style_prefix: str = DEFAULT_STYLE_PREFIX,
    api_key: Optional[str] = None
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """
    Executes both Zero-Shot and Few-Shot techniques on the same concept
    and returns both storyboards for side-by-side comparison.
    """
    zero_shot, note_zero = generate_storyboard(idea, "zero-shot", style_prefix, api_key)
    few_shot, note_few = generate_storyboard(idea, "few-shot", style_prefix, api_key)
    comparison_summary = (
        f"**Zero-Shot Status**: {note_zero}\n\n"
        f"**Few-Shot Status**: {note_few}\n\n"
        f"*Observe how Few-Shot prompting typically yields richer cinematic framing words "
        f"(e.g., macro shot, Dutch angle) and tighter emotional narrative beats compared to Zero-Shot.*"
    )
    return zero_shot, few_shot, comparison_summary
