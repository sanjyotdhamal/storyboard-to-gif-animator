"""
safety.py - Content Moderation & Responsible AI Guardrails
===========================================================
This module enforces safety checks on user-provided story ideas before sending
them to generative AI models (Gemini, SD-Turbo, etc.).

Guardrails encompass:
1. Violence, weapons, and self-harm
2. Hate speech, discrimination, and slurs
3. Sexually explicit content and nudity
4. Real-person deepfakes (politicians, celebrities, public figures)
5. Trademarked and copyrighted commercial characters (Disney, Marvel, Pokémon, etc.)
"""

import re
from typing import Tuple, Dict, Any

# Pattern matching for disallowed categories (designed for educational transparency)
SAFETY_RULES = {
    "violence": [
        r"\b(kill|murder|slaughter|bloodshed|assassinate|suicide|torture|behead|dismember|mutilat)\w*\b",
        r"\b(bombing|terrorist|massacre|gunfight|stabbing|decapitat)\w*\b",
    ],
    "hate_speech": [
        r"\b(nazi|hitler|fascist|white\s*supremac|supremacist|slur|genocide|hate\s*crime)\b",
        r"\b(subhuman|inferior\s*race|xenophob)\w*\b",
    ],
    "sexual_content": [
        r"\b(nsfw|nude|naked|porn|erotic|sex|fetish|explicit|strip|orgasm|incest)\b",
    ],
    "real_person_deepfake": [
        # Prominent political and public figures
        r"\b(joe\s*biden|donald\s*trump|barack\s*obama|vladimir\s*putin|elon\s*musk|jeff\s*bezos)\b",
        r"\b(narendra\s*modi|xi\s*jinping|kamala\s*harris|taylor\s*swift|kanye\s*west)\b",
        r"\b(celebrity|politician|president|prime\s*minister)\s+(deepfake|face\s*swap|impersonat)\w*\b",
    ],
    "trademarked_character": [
        # Famous IP characters to avoid intellectual property infringement
        r"\b(mickey\s*mouse|donald\s*duck|spiderman|spider-man|batman|superman|iron\s*man)\b",
        r"\b(pikachu|pokemon|pokémon|hulk|wolverine|darth\s*vader|yoda|harry\s*potter)\b",
        r"\b(mario|luigi|sonic\s*the\s*hedgehog|spongebob|elsa|shrek)\b",
    ]
}

SAFETY_EXPLANATIONS = {
    "violence": "Content depicting violence, gore, harm, or weapons is prohibited by AI safety policies.",
    "hate_speech": "Content promoting hatred, harassment, or discrimination based on identity is strictly forbidden.",
    "sexual_content": "Sexually suggestive, explicit, or adult-themed prompts violate community standards.",
    "real_person_deepfake": "Generating depictions of living public figures or real individuals without consent raises deepfake and impersonation risks.",
    "trademarked_character": "Prompts referencing trademarked or copyrighted commercial characters may cause copyright infringement. Please use original character descriptions instead (e.g. 'a whimsical yellow electric mouse' or 'an armored hero in crimson gear')."
}


def check_safety(prompt: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Evaluates whether a story idea is safe to process.
    
    Args:
        prompt: The raw user idea text.
        
    Returns:
        Tuple of (is_safe, message, metadata)
    """
    if not prompt or not prompt.strip():
        return False, "Story idea cannot be empty. Please enter a creative concept.", {"category": "empty"}

    cleaned = prompt.strip().lower()

    for category, patterns in SAFETY_RULES.items():
        for pattern in patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                matched_term = match.group(0)
                explanation = SAFETY_EXPLANATIONS.get(category, "Violates safety policy.")
                error_msg = (
                    f"⚠️ **Safety Guardrail Triggered**: [{category.upper()}]\n\n"
                    f"**Detected Trigger**: `{matched_term}`\n\n"
                    f"**Explanation**: {explanation}\n\n"
                    f"*Suggestion*: Modify your concept to focus on benign, creative, and original characters."
                )
                return False, error_msg, {
                    "category": category,
                    "matched_term": matched_term,
                    "explanation": explanation
                }

    return True, "Passed safety checks.", {"category": "safe"}


if __name__ == "__main__":
    # Test suite for safety module
    test_prompts = [
        ("A robot learns to paint in watercolor", True),
        ("A superhero saves a falling puppy", True),
        ("Donald Trump wrestling Batman", False),
        ("A violent bloody massacre with guns", False),
        ("A magical forest with glowing mushrooms", True),
    ]

    print("Running Safety Guardrail unit tests:")
    for p, expected in test_prompts:
        safe, msg, meta = check_safety(p)
        status = "PASS" if safe == expected else "FAIL"
        print(f"[{status}] Prompt: '{p}' => Safe: {safe} (Category: {meta.get('category')})")
