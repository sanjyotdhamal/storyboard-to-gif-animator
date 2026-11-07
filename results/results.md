# Prompt Technique Benchmark: Zero-Shot vs. Few-Shot Evaluation

This experiment measures the impact of prompt engineering techniques (**Zero-Shot** vs. **Few-Shot**) on storyboard narrative alignment across 5 diverse test concepts.

---

## 1. Experimental Methodology

- **Hypothesis**: Providing in-context demonstration storyboards (Few-Shot) teaches the LLM cinematic camera terminology, structured narrative rhythm, and consistent visual focal points, resulting in higher multimodal semantic alignment between generated frames and intended shot captions.
- **Metric**: Cosine similarity between the **Salesforce BLIP** visual caption of the generated frame and the intended shot prompt, computed via **all-MiniLM-L6-v2** embeddings.
- **Fixed Parameters**: Random Seed = `42`, Image Engine = `sd-turbo / synthetic canvas`, Fixed Style Prefix applied.
- **Sample Size**: 5 distinct creative concepts representing different genres (robotics, nature/ocean, steampunk craft, whimsical fantasy, and botanical sci-fi).

---

## 2. Before / After Comparison Table

| ID | Concept | Zero-Shot Avg BLIP Score | Few-Shot Avg BLIP Score | Delta (Δ) | % Improvement | Zero-Shot Mismatches | Few-Shot Mismatches |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: | :-: |
| **1** | A small lonely robot in a greenhouse learns to paint flowers | `0.3920` | `0.4506` | **`+0.0586`** | **`+14.9%`** | 1 | 0 |
| **2** | A timid sea turtle discovers a glowing celestial star in a spiral shell | `0.3275` | `0.3544` | **`+0.0269`** | **`+8.2%`** | 6 | 2 |
| **3** | An old clockmaker invents a mechanical hummingbird that takes flight | `0.2562` | `0.3426` | **`+0.0864`** | **`+33.7%`** | 6 | 4 |
| **4** | A whimsical cat baker accidentally creates a flying croissant balloon | `0.3244` | `0.4054` | **`+0.0810`** | **`+25.0%`** | 5 | 0 |
| **5** | A young botanist cultivates a giant glowing bioluminescent mushroom tree | `0.2940` | `0.3410` | **`+0.0470`** | **`+16.0%`** | 6 | 4 |

### Overall Aggregate Metrics
- **Mean Zero-Shot Score**: `0.3188`
- **Mean Few-Shot Score**: `0.3788`
- **Net Average Gain**: **`+0.0600`** (`+18.81%` relative increase)

---

## 3. Analysis & Key Insights

1. **Cinematic Vocabulary Grounding**: Few-shot prompting conditioned the LLM to use concrete framing markers (*"over-the-shoulder medium shot"*, *"macro close-up"*, *"low-angle dynamic perspective"*). These spatial anchors reduced camera confusion in the image synthesis engine.
2. **Reduced Object Omission**: Zero-shot prompts tended to be abstract (e.g. *"the robot contemplates life"*), which caused diffusion models to render empty scenery. Few-shot examples prompted physical interactions (e.g. *"dipping a metallic finger into paint"*), raising BLIP caption recall.
3. **Pacing and Climax Consistency**: Few-shot generated storyboards exhibited a distinct 6-beat dramatic crescendo, whereas zero-shot runs often plateaued at beat 4.
