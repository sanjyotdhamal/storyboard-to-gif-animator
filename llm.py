"""
llm.py - Storyboard Generation with Gemini API & Prompt Engineering
===================================================================
Generates a coherent 6-shot visual storyboard from a user's text concept.

Key Concepts Demonstrated:
1. Zero-Shot Prompting: Providing direct task instructions without prior examples.
2. Few-Shot Prompting: Providing high-quality input-output examples in context to guide
   narrative rhythm, cinematic camera terminology, and visual consistency.
3. Structured Output: Generating strictly formatted JSON containing 6 distinct narrative beats.
4. Shared Style Prefix: Prepended to every frame prompt to maintain artistic coherence across the clip.
"""

import os
import json
import re
from typing import Dict, Any, List, Tuple
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


def _generate_mock_storyboard(idea: str, technique: str, style_prefix: str) -> Dict[str, Any]:
    """
    Fallback deterministic storyboard generator if Gemini API key is missing or offline.
    Ensures complete end-to-end functionality and student testing without quota limits.
    """
    cleaned_idea = idea.strip() or "A creative adventure"
    
    # Enrich slightly based on technique to demonstrate difference
    extra_detail = "with rich cinematic framing and character nuance" if technique == "few-shot" else "with direct narrative progression"
    
    shots = [
        {
            "shot_number": 1,
            "camera": "Wide Establishing Shot",
            "image_prompt": f"{style_prefix}, wide establishing shot of {cleaned_idea}, setting the peaceful world, cinematic environment, {extra_detail}",
            "caption": f"Introduction: The story begins with {cleaned_idea} in its original setting."
        },
        {
            "shot_number": 2,
            "camera": "Medium Discovery Shot",
            "image_prompt": f"{style_prefix}, medium shot focusing on the main character discovering an intriguing mystery related to {cleaned_idea}, expressive eyes, {extra_detail}",
            "caption": "Inciting Incident: An unexpected discovery sparks a journey of transformation."
        },
        {
            "shot_number": 3,
            "camera": "Close-up Action Shot",
            "image_prompt": f"{style_prefix}, close-up dynamic shot of the first attempt and challenge during {cleaned_idea}, detailed textures and vibrant lighting, {extra_detail}",
            "caption": "Rising Action: Working through the initial challenge with steady determination."
        },
        {
            "shot_number": 4,
            "camera": "Low-angle Dramatic Shot",
            "image_prompt": f"{style_prefix}, low angle dramatic perspective of the turning point in {cleaned_idea}, emotional intensity, dynamic color contrast, {extra_detail}",
            "caption": "Turning Point: A breakthrough moment where effort and inspiration collide."
        },
        {
            "shot_number": 5,
            "camera": "Climactic High-Energy Shot",
            "image_prompt": f"{style_prefix}, climactic scene reaching peak realization in {cleaned_idea}, sparkling particles, radiant visual spectacle, {extra_detail}",
            "caption": "Climax: The ultimate achievement unfolds in full splendor."
        },
        {
            "shot_number": 6,
            "camera": "Wide Heartwarming Shot",
            "image_prompt": f"{style_prefix}, wide heartwarming resolution shot concluding {cleaned_idea}, golden hour light, sense of fulfillment, {extra_detail}",
            "caption": "Resolution: A peaceful celebration of the completed journey."
        }
    ]

    return {
        "title": f"The Tale of: {cleaned_idea[:30]}",
        "technique": technique,
        "style_prefix": style_prefix,
        "shots": shots,
        "source": "educational_mock_generator"
    }


def generate_storyboard(
    idea: str,
    technique: str = "few-shot",
    style_prefix: str = DEFAULT_STYLE_PREFIX,
    api_key: str = None
) -> Tuple[Dict[str, Any], str]:
    """
    Generates a 6-shot visual storyboard using Gemini API (or structured educational fallback).
    
    Args:
        idea: The user's prompt / narrative concept.
        technique: 'zero-shot' or 'few-shot'
        style_prefix: Visual style prepended to all image prompts for aesthetic consistency.
        api_key: Optional explicit Gemini API key (defaults to os.environ['GEMINI_API_KEY']).
        
    Returns:
        Tuple of (storyboard_dict, system_note)
    """
    active_key = api_key or os.getenv("GEMINI_API_KEY")

    # If no Gemini API key is configured, fallback to high-quality educational generator
    if not active_key or active_key.strip() == "" or "your_gemini_api_key" in active_key:
        note = "ℹ️ Running in Educational Fallback mode (No GEMINI_API_KEY provided in .env). Generated deterministic 6-shot storyboard."
        return _generate_mock_storyboard(idea, technique, style_prefix), note

    try:
        import google.generativeai as genai
        genai.configure(api_key=active_key)
        
        # Select current fast and capable Gemini model
        model = genai.GenerativeModel("gemini-1.5-flash")

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
            "1. Output MUST be strictly valid JSON with no conversational preamble.\n"
            "2. Exactly 6 shots in an array named 'shots'.\n"
            f"3. Every shot's 'image_prompt' MUST begin with the exact shared style prefix: '{style_prefix}' followed by the scene description.\n"
            "4. Each shot must include: 'shot_number' (1-6), 'camera' (e.g., Wide shot, Close-up), 'image_prompt', and 'caption' (1 clear narrative sentence)."
        )

        if technique.lower() == "few-shot":
            user_prompt = (
                f"{system_instruction}\n\n"
                f"### IN-CONTEXT EXAMPLES FOR GUIDANCE:\n{FEW_SHOT_EXAMPLES}\n\n"
                f"### NOW GENERATE THE STORYBOARD FOR THIS CONCEPT:\n"
                f"User Concept: \"{idea}\"\n"
                f"Output JSON:"
            )
        else: # zero-shot
            user_prompt = (
                f"{system_instruction}\n\n"
                f"User Concept: \"{idea}\"\n"
                f"Output JSON:"
            )

        response = model.generate_content(
            user_prompt,
            generation_config={"temperature": 0.7, "response_mime_type": "application/json"}
        )

        parsed = _clean_json_response(response.text)
        
        # Ensure style prefix is cleanly prepended on each shot if model omitted it
        if "shots" in parsed and isinstance(parsed["shots"], list):
            for s in parsed["shots"]:
                p = s.get("image_prompt", "")
                if not p.startswith(style_prefix[:20]):
                    s["image_prompt"] = f"{style_prefix}, {p}"
                    
        parsed["technique"] = technique
        parsed["style_prefix"] = style_prefix
        parsed["source"] = "gemini-1.5-flash"
        return parsed, "✅ Storyboard successfully generated using Gemini 1.5 Flash."

    except Exception as e:
        error_msg = str(e)
        note = f"⚠️ Gemini API encounter: {error_msg}. Using offline fallback generator."
        fallback_data = _generate_mock_storyboard(idea, technique, style_prefix)
        fallback_data["api_error"] = error_msg
        return fallback_data, note


def compare_prompts(
    idea: str,
    style_prefix: str = DEFAULT_STYLE_PREFIX,
    api_key: str = None
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
