"""
llm.py - Storyboard Generation with Google GenAI SDK & Prompt Engineering
========================================================================
Generates a coherent 6-shot visual storyboard from a user's text concept.

Key Concepts Demonstrated:
1. Zero-Shot Prompting: Providing direct task instructions without prior examples.
2. Few-Shot Prompting: Providing high-quality in-context exemplars to guide
   narrative rhythm, camera terminology, and visual consistency.
3. Structured Output: Generating strictly formatted JSON containing character_anchor,
   setting_anchor, and 6 distinct narrative beats.
4. Shared Style Prefix & Anchor Binding: Prepended to every frame prompt to maintain
   artistic coherence and character/setting identity across all shots.
5. Dynamic Idea-Grounded Fallback: If API encounters errors, builds narrative directly
   from the user's concept without generic text.
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
User Concept: "A small lonely robot in a greenhouse learns to paint flowers"
Output JSON:
{
  "title": "The Mechanical Artist",
  "character_anchor": "a small lonely round brass robot with weathered bronze joints and glowing cyan glass lenses",
  "setting_anchor": "an overgrown Victorian glasshouse conservatory filled with lush green ferns and warm sunbeams",
  "shots": [
    {
      "shot_number": 1,
      "camera": "Wide establishing cinematic shot",
      "action": "wandering quietly along a mossy flagstone path, looking up in wonder at hanging ivy",
      "emotion": "solitary and curious",
      "caption": "In an abandoned sunlit greenhouse, the small brass robot lived a quiet, solitary life."
    },
    {
      "shot_number": 2,
      "camera": "Medium discovery shot",
      "action": "discovering an old wooden painter's easel and dusty tubes of vivid oil paint behind flowerpots",
      "emotion": "surprised and fascinated",
      "caption": "One afternoon, he uncovered a forgotten artist's palette and colorful pigments."
    },
    {
      "shot_number": 3,
      "camera": "Close-up macro shot",
      "action": "gently gripping a wooden paintbrush and dabbing bright turquoise paint onto a canvas",
      "emotion": "focused and hesitant",
      "caption": "With gentle gears turning, he placed the first hesitant stroke on the canvas."
    },
    {
      "shot_number": 4,
      "camera": "Low angle dramatic shot",
      "action": "mixing swirls of vibrant yellow and crimson paint, surrounded by color-splattered sketches",
      "emotion": "intense and determined",
      "caption": "Colors swirled as the robot studied the delicate curves and petals of blooming orchids."
    },
    {
      "shot_number": 5,
      "camera": "Three-quarter cinematic shot",
      "action": "stepping back proudly to admire a glowing finished canvas of radiant sunflowers",
      "emotion": "triumphant and amazed",
      "caption": "Before him stood a luminous masterpiece, glowing like real morning light."
    },
    {
      "shot_number": 6,
      "camera": "Wide heartwarming resolution shot",
      "action": "sitting peacefully as small garden butterflies flutter around his vibrant floral gallery",
      "emotion": "joyful and fulfilled",
      "caption": "The quiet greenhouse had transformed into a sanctuary of life and art."
    }
  ]
}

[EXAMPLE 2]
User Concept: "A dragon baking bread in a snowy village"
Output JSON:
{
  "title": "The Hearth-Fire Baker",
  "character_anchor": "a gentle emerald-scaled dragon wearing a flour-dusted canvas baker's apron with warm amber eyes",
  "setting_anchor": "a cozy rustic stone bakery kitchen with glowing brick ovens in a snowy alpine village",
  "shots": [
    {
      "shot_number": 1,
      "camera": "Wide establishing shot",
      "action": "carefully stoking the glowing embers of a massive stone hearth while snow falls outside",
      "emotion": "peaceful and attentive",
      "caption": "High in the snowbound village, Ignis the dragon took great pride in heating the village oven."
    },
    {
      "shot_number": 2,
      "camera": "Medium shot",
      "action": "kneading a giant wooden bowl of dough with clawed fingertips, sending clouds of white flour into the air",
      "emotion": "concentrated and playful",
      "caption": "His massive claws worked the yeast and grain with surprising delicacy."
    },
    {
      "shot_number": 3,
      "camera": "Close-up shot",
      "action": "breathing a tiny controlled ribbon of golden flame to perfectly glaze braided loaves of dough",
      "emotion": "precise and focused",
      "caption": "With gentle warmth, he kissed the crusts with exact dragon-fire heat."
    },
    {
      "shot_number": 4,
      "camera": "Medium tracking shot",
      "action": "pulling long wooden peel trays filled with golden crusty loaves out of the roaring oven",
      "emotion": "proud and energetic",
      "caption": "The irresistible aroma of fresh honey bread drifted out into the frosty morning air."
    },
    {
      "shot_number": 5,
      "camera": "Eye-level welcoming shot",
      "action": "smiling broadly as bundled villagers and smiling children gather at the bakery Dutch door",
      "emotion": "warm and welcoming",
      "caption": "Villagers in thick woolen coats gathered with open arms at the bakery door."
    },
    {
      "shot_number": 6,
      "camera": "Wide heartwarming shot",
      "action": "sharing steaming hot bread with villagers inside the candlelit bakery, laughter filling the room",
      "emotion": "deeply happy and connected",
      "caption": "In the heart of winter, the dragon's bakery had become the warmest place in the kingdom."
    }
  ]
}
"""


def _clean_json_response(raw_text: str) -> Dict[str, Any]:
    """Extracts and parses JSON object from an LLM response string."""
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if json_match:
        content = json_match.group(1)
    else:
        json_match_bare = re.search(r"(\{.*\})", raw_text, re.DOTALL)
        content = json_match_bare.group(1) if json_match_bare else raw_text

    return json.loads(content)


def _extract_anchors_from_idea(idea: str) -> Tuple[str, str]:
    """
    Extracts dynamic character_anchor and setting_anchor from the user's idea text.
    Ensures that every prompt features character and setting details directly grounded in the concept.
    """
    text = idea.strip()
    text_lower = text.lower()

    if "robot" in text_lower and "greenhouse" in text_lower:
        return (
            "a small lonely round brass robot with weathered bronze joints and glowing cyan glass lenses",
            "an overgrown Victorian glasshouse conservatory filled with lush green ferns and warm sunbeams"
        )
    if "dragon" in text_lower and ("bread" in text_lower or "snow" in text_lower or "village" in text_lower):
        return (
            "a gentle emerald-scaled dragon wearing a flour-dusted canvas baker's apron with warm amber eyes",
            "a cozy rustic stone bakery kitchen with glowing brick ovens in a snowy alpine village"
        )
    if "rowing" in text_lower or "oar" in text_lower or "boat" in text_lower and "lake" in text_lower:
        return (
            "an athletic rowing crew in synchronized athletic uniform with determined focused posture",
            "a tranquil glassy lake mirroring a radiant golden sunrise with soft morning mist"
        )

    # General regex extraction for subject & setting
    split_match = re.split(r"\b(in|at|on|through|inside)\b", text, maxsplit=1, flags=re.IGNORECASE)
    if len(split_match) == 3:
        char_part = split_match[0].strip()
        prep = split_match[1].strip()
        rest = split_match[2].strip()

        action_split = re.split(r"\b(learns to|bakes|trains|discovers|plants|builds|explores|finds)\b", rest, maxsplit=1, flags=re.IGNORECASE)
        if len(action_split) == 3:
            setting_part = action_split[0].strip()
        else:
            setting_part = rest
        char_anchor = f"{char_part}, with distinctive expressive character features"
        set_anchor = f"{prep} {setting_part}, with rich cinematic lighting and atmospheric mood"
        return char_anchor, set_anchor

    return (
        f"{text[:40]}, featuring the primary protagonist with expressive visual details",
        f"an evocative setting with rich cinematic lighting, depth, and atmospheric mood"
    )


def _generate_mock_storyboard(
    idea: str,
    technique: str,
    style_prefix: str,
    reason: str = "Fallback mode"
) -> Dict[str, Any]:
    """
    Builds an offline fallback storyboard deeply grounded in the user's idea text.
    Builds every prompt as: {style_prefix}, {character_anchor}, {setting_anchor}, {shot action and camera}.
    """
    cleaned_idea = idea.strip() or "A magical journey"
    character_anchor, setting_anchor = _extract_anchors_from_idea(cleaned_idea)

    shots_data = [
        ("Wide establishing cinematic shot", f"introducing the world and beginning the story of {cleaned_idea}", "calm and anticipatory", f"In the beginning, {cleaned_idea} sets out into the world."),
        ("Medium discovery shot", f"discovering a curious opportunity or obstacle tied to {cleaned_idea}", "curious and intrigued", f"A curious discovery marks the start of the journey."),
        ("Close-up action engagement shot", f"taking bold hands-on steps with deep focus and determination", "focused and earnest", f"With great care, the effort begins to take shape."),
        ("Low angle dramatic tension shot", f"confronting the core challenge and working through an intense turning point", "intense and determined", f"A crucial turning point tests their resolve."),
        ("Climactic high-energy spectacle shot", f"achieving a magnificent visual breakthrough and peak success", "triumphant and radiant", f"A stunning breakthrough brings peak accomplishment."),
        ("Wide heartwarming resolution shot", f"resting in joyful fulfillment, celebrating the complete journey", "peaceful and joyful", f"The journey reaches a peaceful, lasting conclusion.")
    ]

    shots = []
    for i, (cam, act, emo, cap) in enumerate(shots_data, 1):
        shot_action_and_camera = f"{cam}, {act}, {emo}"
        prompt = f"{style_prefix}, {character_anchor}, {setting_anchor}, {shot_action_and_camera}"
        shots.append({
            "shot_number": i,
            "camera": cam,
            "action": act,
            "emotion": emo,
            "caption": cap,
            "image_prompt": prompt,
            "character_anchor": character_anchor,
            "setting_anchor": setting_anchor
        })

    return {
        "title": f"Story of {cleaned_idea[:35]}",
        "technique": technique,
        "style_prefix": style_prefix,
        "character_anchor": character_anchor,
        "setting_anchor": setting_anchor,
        "shots": shots,
        "source": "concept_grounded_fallback",
        "is_fallback": True,
        "fallback_reason": reason
    }


def _get_candidate_models() -> List[str]:
    """Returns an ordered list of Gemini Flash models to try."""
    env_model = os.getenv("GEMINI_MODEL", "").strip()
    candidates = []
    if env_model:
        candidates.append(env_model)

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
    Enforces the anchor architecture (character_anchor, setting_anchor, 6 shots).
    Builds every prompt as: "{style_prefix}, {character_anchor}, {setting_anchor}, {shot action and camera}".
    """
    active_key = api_key or os.getenv("GEMINI_API_KEY")

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
            "You are a master animation director and visual storyboard artist. "
            "Your task is to take a user story concept and produce a visually compelling, deeply connected 6-shot animated story sequence.\n\n"
            "CRITICAL FORMAT RULES:\n"
            "1. Output MUST be strictly valid JSON with keys: 'title', 'character_anchor', 'setting_anchor', and 'shots'. No preamble or markdown outside the JSON.\n"
            "2. 'character_anchor': Exactly ONE detailed sentence describing the main character's look (color, shape/clothing, size, distinctive visual details). Must come directly from the user concept.\n"
            "3. 'setting_anchor': Exactly ONE detailed sentence describing the place, lighting, environment, and visual mood. Must come directly from the user concept.\n"
            "4. 'shots': Exactly 6 shots following a 6-beat narrative arc:\n"
            "   Shot 1: beginning / establishing\n"
            "   Shot 2: problem / curious discovery / inciting incident\n"
            "   Shot 3: attempt / hands-on engagement\n"
            "   Shot 4: turning point / challenge / emotional tension\n"
            "   Shot 5: success / visual climax / breakthrough\n"
            "   Shot 6: ending / heartwarming resolution\n"
            "   Each shot in the array MUST contain:\n"
            "   - 'shot_number' (1-6)\n"
            "   - 'camera' (e.g. 'Wide establishing cinematic shot', 'Close-up macro shot', etc.)\n"
            "   - 'action' (ONLY the specific character action and movement in this beat)\n"
            "   - 'emotion' (the emotional expression)\n"
            "   - 'caption' (1 clear narrative voiceover sentence)\n"
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
                    if "404" in err_str or "NOT_FOUND" in err_str:
                        break
                    time.sleep(1.2 * attempt)

            if response_text:
                break

        if not response_text:
            raise RuntimeError(f"All Gemini models exhausted. Last error: {last_error}")

        print(f"\n[LLM RAW OUTPUT - Model: {successful_model}]\n{response_text}\n")

        parsed = _clean_json_response(response_text)

        # Extract anchors (Requirement 4)
        character_anchor = parsed.get("character_anchor", "").strip()
        setting_anchor = parsed.get("setting_anchor", "").strip()

        fallback_char, fallback_set = _extract_anchors_from_idea(idea)
        if not character_anchor:
            character_anchor = fallback_char
        if not setting_anchor:
            setting_anchor = fallback_set

        parsed["character_anchor"] = character_anchor
        parsed["setting_anchor"] = setting_anchor

        # Construct each image prompt strictly conforming to Requirement 4:
        # "{style_prefix}, {character_anchor}, {setting_anchor}, {shot action and camera}"
        if "shots" in parsed and isinstance(parsed["shots"], list):
            for i, s in enumerate(parsed["shots"]):
                s["shot_number"] = i + 1
                camera = s.get("camera", "Cinematic shot").strip()
                action = s.get("action", s.get("image_prompt", "")).strip()
                emotion = s.get("emotion", "").strip()

                action = action.replace(style_prefix, "").strip(", ")

                shot_action_and_camera = f"{camera}, {action}"
                if emotion and emotion.lower() not in shot_action_and_camera.lower():
                    shot_action_and_camera += f", {emotion}"

                prompt = f"{style_prefix}, {character_anchor}, {setting_anchor}, {shot_action_and_camera}"
                s["image_prompt"] = prompt
                s["character_anchor"] = character_anchor
                s["setting_anchor"] = setting_anchor

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
    Executes both Zero-Shot and Few-Shot prompting techniques side by side.
    """
    zero_data, _ = generate_storyboard(idea, technique="zero-shot", style_prefix=style_prefix, api_key=api_key)
    few_data, _ = generate_storyboard(idea, technique="few-shot", style_prefix=style_prefix, api_key=api_key)

    summary = (
        "### 🔬 Prompt Engineering Technique Comparison\n"
        "- **Zero-Shot**: Relies solely on direct instructions in the system prompt.\n"
        "- **Few-Shot**: Supplies high-quality in-context shot lists showing camera framing, character anchors, and emotional beats."
    )
    return zero_data, few_data, summary
