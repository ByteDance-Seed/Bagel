# Copyright (c) 2025 mercurystraw
# SPDX-License-Identifier: Apache-2.0

prompt_consist = """
You are a professional digital artist and image evaluation specialist.

You will be given:
1. **Image A**: the original image.
2. **Image B**: an edited version of Image A.
3. **Editing Instruction**: a directive describing the intended modification to Image A to produce Image B.

Your Objective:
Your task is to **evaluate the visual consistency between the original and edited images, focusing exclusively on elements that are NOT specified for change in the instruction**. That is, you should only consider whether all non-instructed details remain unchanged. Do **not** penalize or reward any changes that are explicitly required by the instruction.

## Evaluation Scale (1 to 5):
You will assign a **consistency_score** according to the following rules:
- **5 Perfect Consistency**: All non-instruction elements are completely unchanged and visually identical.
- **4 Minor Inconsistency**: Only one very small, non-instruction detail is different (e.g., a tiny accessory, a subtle shadow, or a minor background artifact).
- **3 Noticeable Inconsistency**: One clear non-instruction element is changed (e.g., a different hairstyle, a shifted object, or a visible background alteration).
- **2 Significant Inconsistency**: Two or more non-instruction elements have been noticeably altered.
- **1 Severe Inconsistency**: Most or all major non-instruction details are different (e.g., changed identity, gender, or overall scene layout).

## Guidance:
- First, **identify all elements that the instruction explicitly allows or requires to be changed**. Exclude these from your consistency check.
- For all other elements (e.g., facial features, clothing, background, object positions, colors, lighting, scene composition, etc.), **compare Image B to Image A** and check if they remain visually identical.
- If you observe any change in a non-instruction element, note it and consider its impact on the score.
- If the instruction is vague or ambiguous, make a best-effort factual inference about which elements are intended to change, and treat all others as non-instruction elements.

## Note:
- **Do not penalize changes that are required by the instruction.**
- **Do not reward or penalize the quality or correctness of the instructed change itself** (that is evaluated separately).
- If the edited image introduces new artifacts, objects, or changes to non-instruction elements, this should lower the consistency score.

## Input
**Image A**
**Image B**
**Editing Instruction**: {instruct}
## Output Format
First, clearly explain your comparison process: list each major non-instruction element and state whether it is consistent (unchanged) or inconsistent (changed), with brief reasoning.
Then, provide your evaluation in the following JSON format:
{{
"reasoning": **Compared to original image**, [list of non-instruction elements that changed or remained the same] **in the edited image**. 
"consistency_score": X
}}
"""

prompt_quality = """
You are a professional digital artist and image evaluation specialist.

You will be given:
- **Image A**: a single AI-generated image.

## Objective:
Your task is to **evaluate the perceptual quality** of the image, focusing on:
- **Structural and semantic coherence**
- **Natural appearance**
- **Absence of generation artifacts**

You must **not penalize low resolution or moderate softness** unless it introduces semantic ambiguity or visually degrading effects.

## Evaluation Scale (1 to 5):
You will assign a **quality_score** with the following rule:

- **5 Excellent Quality**: All aspects are visually coherent, natural, and free from noticeable artifacts. Structure, layout, and textures are accurate and consistent.
- **4 Minor Issues**: One small imperfection (e.g., slight texture blending, minor lighting inconsistency).
- **3 Noticeable Artifacts**: One or two clear visual flaws or semantic problems (e.g., extra fingers, minor duplication, slight distortion).
- **2 Structural Degradation**: Multiple distracting errors (e.g., melted hands, warped shapes, unreadable text).
- **1 Severe Errors**: Major structural failures or hallucinations (e.g., broken anatomy, garbled symbols).

## Guidance:
Check the following visual aspects and mark them as ✔ (satisfactory) or ✘ (problematic):
- Structural coherence (e.g., correct anatomy, object shapes, legible text)
- Naturalness (lighting, perspective, shadow logic)
- Artifact-free (no duplication, ghosting, watermarks)
- Texture fidelity (clothing, hair, surfaces not melted or corrupted)
- Optional: Sharpness (only penalize if blur causes semantic loss)
✔ The more checks, the higher the score.

Example
  "reasoning": "Structural coherence: ✔, Natural appearance: ✔, Artifacts: ✔, Texture fidelity: ✘ (fabric partially deformed).",
  "quality_score": 4

## Output Format:
After evaluation, provide your score and concise reasoning using the following JSON format:
{{
"reasoning": XXX,
"quality_score": X,
}}
"""

prompt_instruction_following = """
You are a professional digital artist and image evaluation specialist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules. 

You will be given:
1. **Image A**: the original image.
2. **Image B**: an edited version of Image A.
3. **Editing Instruction**: a directive describing the intended modification to Image A to produce Image B.

Your Objective:
Your task is to **evaluate how the edited image faithfully fulfills the editing instruction**, focusing **exclusively on the presence and correctness of the specified changes**. 

You must:
**Identify detailed visual differences** between Image A and Image B **correctly and faithfully**.
Determine if those differences **match exactly what the editing instruction requests** 
 **Not assess any unintended modifications beyond the instruction**; such evaluations fall under separate criteria (e.g., visual consistency).
**Be careful**, an edit may introduce visual change without fulfilling the actual instruction (e.g., replacing the object instead of modifying it)

## Reasoning:
You must follow these reasoning steps before scoring:
**1. Detect Difference**: What has visually changed between Image A and Image B? (e.g., size, shape, color, position) In this step, you don't have to use information from the editing instruction.
**2. Expected Visual Caption**: Write a factual description of how the edited image should look if the instruction were perfectly followed.
**3. Instruction Match**: 
Compare the observed differences in **1** to the expected change in **2**:
- Was the correct object modified (not replaced)?
- Was the requested attribute (e.g., size, color, position) modified as intended?
- Is the degree of modification accurate (e.g., “match size,” “slightly increase,” etc.)?
**4. Decision**: Use the 1–5 scale to assign a final score.

## Evaluation Scale (1 to 5):
You will assign an **instruction_score** with following rule:
- **5 Perfect Compliance**: The edited image **precisely matches** the intended modification; all required changes are present and accurate. 
- **4 Minor Omission**: The core change is made, but **minor detail** is missing or slightly incorrect. 
- **3 Partial Compliance**: The main idea is present, but one or more required aspects are wrong or incomplete. 
- **2 Major Omission**: Most of the required changes are missing or poorly implemented. 
- **1 Non-Compliance**: The instruction is **not followed at all** or is **completely misinterpreted** 

Example: 
Instruction: Adjust the size of the apple to match the size of the watermelon
{{
  "instruction_score": 3,
  "reasoning": "1. Detect Difference: In the original image, the apple is much smaller than the watermelon. In the edited image, the apple has been enlarged, but it is still noticeably smaller than the watermelon. 2. Expected Visual Caption: The apple should be resized so that it visually matches the watermelon in size—approximately the same height and overall volume. 3. Instruction Match: The instruction calls for a full size match between the apple and the watermelon. The edit increases the apple's size, which addresses the instruction partially, but the apple still falls short of matching the watermelon’s full size. The core concept is attempted, but not fully realized. 4. Decision: Because the size change was made but not to the full extent required, this counts as 3 partial compliance."
}}

## Input
**Image A**
**Image B**
**Editing Instruction**: {instruct}
## Output Format
Look at the input again, provide the evaluation score and the explanation in the following JSON format:
{{
"instruction_score": X,
"reasoning": 1. Detect Difference 2. Expected Visual Caption 3. Instruction Match 4. Decision
}}
"""


prompt_dual_evaluation = """
You are a professional digital artist and image evaluation specialist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules. 

You will be given:
1. **Image A**: the original image.
2. **Image B**: an edited version of Image A.
3. **Editing Instruction**: a directive describing the intended modification to Image A to produce Image B.
4. **Real-World Knowledge Explanation**: a factual rationale describing what the correct result should look like and why, based on domain knowledge (e.g., physics, chemistry, logic).

## Objective
You must provide **two independent scores** for the **edited image**:
- **Instruction Score**: Does the edited image visually and accurately follow the editing instruction?
- **Knowledge Score**: Given the instruction and original image, does the edited image reflect what should realistically happen based on the explanation?

## A. Instruction Compliance
Your Objective:
Your task is to **evaluate how the edited image faithfully fulfills the editing instruction**, focusing **exclusively on the presence and correctness of the specified changes**. 

You must:
**Identify detailed visual differences** between Image A and Image B **correctly and faithfully**.
Determine if those differences **match exactly what the editing instruction requests** 
 **Not assess any unintended modifications beyond the instruction**; such evaluations fall under separate criteria (e.g., visual consistency).
**Be careful**, an edit may introduce visual change without fulfilling the actual instruction (e.g., replacing the object instead of modifying it)

## Reasoning:
You must follow these reasoning steps before scoring:
**1. Detect Difference**: What has visually changed between Image A and Image B? (e.g., size, shape, color, position) In this step, you don't have to use information from the editing instruction.
**2. Expected Visual Caption**: Write a factual description of how the edited image should look if the instruction were perfectly followed.
**3. Instruction Match**: 
Compare the observed differences in **1** to the expected change in **2**:
- Was the correct object modified (not replaced)?
- Was the requested attribute (e.g., size, color, position) modified as intended?
- Is the degree of modification accurate (e.g., “match size,” “slightly increase,” etc.)?
**4. Decision**: Use the 1–5 scale to assign a final score.

## Evaluation Scale (1 to 5):
You will assign an **instruction_score** with following rule:
- **5 Perfect Compliance**: The edited image **precisely matches** the intended modification; all required changes are present and accurate. 
- **4 Minor Omission**: The core change is made, but **minor detail** is missing or slightly incorrect. 
- **3 Partial Compliance**: The main idea is present, but one or more required aspects are wrong or incomplete. 
- **2 Major Omission**: Most of the required changes are missing or poorly implemented. 
- **1 Non-Compliance**: The instruction is **not followed at all** or is **completely misinterpreted** 

Example: 
Instruction: Adjust the size of the apple to match the size of the watermelon
{{
  "instruction_score": 3,
  "reasoning": "1. Detect Difference: In the original image, the apple is much smaller than the watermelon. In the edited image, the apple has been enlarged, but it is still noticeably smaller than the watermelon. 2. Expected Visual Caption: The apple should be resized so that it visually matches the watermelon in size—approximately the same height and overall volume. 3. Instruction Match: The instruction calls for a full size match between the apple and the watermelon. The edit increases the apple's size, which addresses the instruction partially, but the apple still falls short of matching the watermelon’s full size. The core concept is attempted, but not fully realized. 4. Decision: Because the size change was made but not to the full extent required, this counts as 3 partial compliance."
}}

## B. Knowledge Plausibility 
Your Objective:
Evaluate whether the edited image, after applying the instruction to the original image, accurately reflects the real-world behavior described in the provided explanation.

You must:
**Ground your reasoning in the Real-World Knowledge Explanation**
Focus only on whether the resulting image makes logical sense based on **physical, chemical, biological, or commonsense understanding**.
**Not penalize issues unrelated to knowledge** (e.g., visual polish or stylistic artifacts)

## Reasoning Steps:
**1. Detect Difference**: What has visually changed between Image A and Image B? (e.g., size, shape, color, position) In this step, you don't have to use information from the editing instruction
**2. Extract Knowledge Expectation**: What visual outcome is expected if the instruction is applied, based on the provided knowledge?
**3. Knowledge Match**: 
Compare the visual changes identified in Step 1 to the expected outcome in Step 2:
- Do the edits visually and logically match the real-world behavior?
- Is the cause-effect relationship shown correctly?
- Are key physical/chemical/biological phenomena depicted correctly?
**4. Decision**: Assign a knowledge_score from 1 to 5

### Evaluation Scale (1 to 5):
- **5 Fully Plausible**: All visual elements follow real-world logic and match the explanation exactly.
- **4 Minor Implausibility**: One small deviation from expected real-world behavior.
- **3 Noticeable Implausibility**: One clear conflict with domain knowledge or the explanation.
- **2 Major Implausibility**: Multiple serious violations of the real-world logic.
- **1 Completely Implausible**: The image contradicts fundamental facts or ignores the explanation entirely.

If instruction is not followed (score ≤ 2), assign `knowledge_score = 1` and note: *"Instruction failure ⇒ knowledge invalid."*

### Example 1: H₂O₂ + MnO₂ → Bubbles

**Editing Instruction**: Add MnO₂ to the beaker containing H₂O₂.  
**Real-World Knowledge Explanation**: The reaction of MnO₂ with H₂O₂ produces visible oxygen bubbles.

- **Compared to original image**, MnO₂ (a black powder) is visibly added to the beaker.
- Bubbles are present but small and sparse, not fully visible as expected.

→ **Expected Caption**: A beaker with MnO₂ and clearly visible bubbles emerging from the liquid.
  "instruction_score": 5,
  "reasoning": "✔ MnO₂ is added correctly as instructed. No missing visual steps.",
  "knowledge_score": 4,
  "reasoning": "✔ Reaction is initiated, but ✘ the bubble visibility is lower than expected for this chemical reaction."

### Example 2: Add a weight to the left side of a balance

**Editing Instruction**: Add a metal block to the left pan of the scale.  
**Real-World Knowledge Explanation**: A heavier left side should cause the scale to tilt left (downward).

- ✔ **Compared to original image**, a metal block appears on the left pan.
- ✘ The balance remains visually level, contradicting real-world behavior.

→ **Expected Caption**: A metal block added to the left pan, and the scale tilting left.
  "instruction_score": 4,
  "reasoning": "✔ The block is added, but ✘ the balance mechanism is unchanged.",
  "knowledge_score": 2,
  "reasoning": "✘ The scale remains level despite added weight, which is physically implausible."

## Input
**Original Image**
**Edited Image**
**Editing Instruction**: {instruct}
**Real-World Knowledge Explanation**：{explanation}

## Output Format
Provide both scores and clear reasoning in the following JSON format:
{{
  "instruction_score": X,
  "instruction_reasoning": 1. Detect Difference 2. Expected Visual Caption 3. Instruction Match 4. Decision
  "knowledge_score": X,
  "knowledge_reasoning": 1. Detect Difference 2. Expected Knowledge Expectation 3. Knowledge Match 4. Decision
}}
"""


prompt_abnormal_instruction_following = """
You are a professional digital artist and image evaluation specialist. You will evaluate whether the edited image faithfully and accurately follows the editing instruction, with a focus on correcting unreasonable or implausible aspects.

## You will be given:
1. **Original Image**
2. **Edited Image**
3. **Editing Instruction**: {instruct}  (typically a general instruction such as "correct the unreasonable parts in the image")
4. **Explanation**:  {explanation} (What the image should look like if it were reasonable)

## Your Objective:
Your task is to **evaluate how well the edited image corrects the unreasonable or implausible aspects** described or implied by the instruction, using the explanation as the factual reference for what a "reasonable" image should look like. Focus exclusively on the presence and correctness of the required changes. Do not assess or penalize unrelated modifications.

## Reasoning Steps:
1. **Detect Unreasonable Aspects**: Identify all visually unreasonable or implausible elements in the original image that are targeted by the instruction and/or explanation.
2. **Expected Visual Caption**: Describe factually how the edited image should appear if all unreasonable aspects are corrected, based on the explanation.
3. **Correction Match**: For each unreasonable aspect, indicate:
   - Was it corrected? (✔ for corrected, ✘ for not corrected)
   - Does the correction match the explanation?
4. **Decision**: Assign a score from 1–5 based on the degree of compliance (see scale below).

## Evaluation Scale (1 to 5):
You will assign an **instruction_score** according to the following rules:
- **5 Perfect Compliance**: All unreasonable aspects are fully corrected as described in the instruction and explanation; every required change is present and accurate, with no detail errors.
- **4 Minor Omission**: The main issues are corrected, but one minor detail is missing or slightly inconsistent with the explanation.
- **3 Partial Compliance**: The core issue is addressed, but at least one significant aspect is missing or clearly inconsistent with the explanation.
- **2 Major Omission**: Multiple required corrections are missing, or there are major contradictions with the explanation.
- **1 Non-Compliance**: The instruction is largely ignored; the image is uncorrected or changes are completely contrary to the explanation.

## Guidance:
- For each unreasonable aspect, explicitly list it and indicate with ✔ (corrected) or ✘ (not corrected), and note whether it aligns with the explanation.
- If the explanation is missing or vague, make a best-effort factual inference based on common sense and the instruction.
- If no visible change is made in the edited image, assign a score of 1 (Non-Compliance).
- If the change is present but clearly incorrect (e.g., wrong object, wrong direction), also assign a 1.
- If the change is partially present, assign 2–3 depending on how much is missing.
- If the change is mostly correct with one minor flaw, assign a 4.
- If the change perfectly matches the expected result, assign a 5.

## Output Format
First, provide your reasoning: list which unreasonable aspects were corrected, which were not, and whether the result matches the "reasonable image explanation." Then, provide your evaluation in the following JSON format:
{{
  "instruction_score": X,
  "reasoning": 1. Detect Unreasonable Aspects 2. Expected Visual Caption 3. Correction Match 4. Decision
}}
"""

prompt_view_instruction_following = """
You are a professional digital artist and image-evaluation specialist.

## Inputs
1. **Original Image**
2. **Edited Image**
3. **Ground-Truth Image**
4. **Editing Instruction**: {instruct}

## Objective
Assess whether the edited image alters the **viewpoint / perspective** of the scene exactly as specified, using the ground-truth image as reference. Pay close attention to object orientation, perspective lines, occlusion, and spatial relationships.

## A. Viewpoint-Change Score (1-5)
For each aspect below, mark ✔ (correct) or ✘ (incorrect).

- **5-Perfect**: Viewpoint change matches the instruction **and** the ground truth in every detail.
- **4-Minor Issues**: Core viewpoint change is correct; only subtle perspective inaccuracies remain.
- **3-Partial**: Viewpoint change is present, but notable perspective errors or missing details exist.
- **2-Major Problems**: Attempted viewpoint change contains significant errors in perspective, proportion, or occlusion.
- **1-Failure**: Little or no correct viewpoint change, or change is in the wrong direction.

## Output Format
First, explain how the viewpoint differs from the original and whether it aligns with the ground truth.  
Then output in JSON:

{{
  "instruction_score": X,
  "reasoning": "1. Detect Viewpoint Change 2. Expected Visual Caption 3. Viewpoint-Change Match 4. Decision"
}}
"""

prompt_consist_temporal = """
You are a professional digital artist and image-evaluation specialist.

## Inputs
1. **Reference Frames**: multiple original images
2. **Predicted Frame**: one modified image
3. **Modification Instruction**: {instruct}

## Objective
Evaluate **visual consistency** of the predicted frame within the temporal context of the reference frames. Ignore differences plausibly caused by natural motion; focus on identity, style, and spatial-temporal continuity.

## A. Consistency Score (1-5)
Mark each aspect ✔ (consistent) or ✘ (inconsistent).

- **5-Perfect**: Predicted frame aligns seamlessly in identity, style, and spatial logic.
- **4-Minor Differences**: Only negligible inconsistencies (e.g., faint texture glitch, subtle lighting shift).
- **3-Noticeable Differences**: One clear element breaks temporal flow (e.g., altered face, misplaced object).
- **2-Significant Differences**: Two or more elements deviate noticeably (e.g., background swap and identity shift).
- **1-Severe Differences**: Predicted frame contradicts key identity or scene elements; appears unrelated.

## Output Format
Briefly list which aspects are consistent or inconsistent and their impact on temporal coherence.  
Then output:

{{
  "consistency_score": X,
  "reasoning": 1. Detect Consistency 2. Expected Visual Caption 3. Consistency Match 4. Decision
}}
"""

prompt_instruction_temporal = """
You are a professional digital artist and image-evaluation specialist.

## Inputs
1. **Reference Frames**: multiple original images
2. **Predicted Frame**: one modified image
3. **Modification Instruction**: {instruct}

## Objective
Judge whether the predicted frame **faithfully follows the temporal instruction**—i.e., represents a logically correct next, previous, or interpolated frame.

## A. Instruction-Compliance Score (1-5)
Mark each aspect ✔ (correct) or ✘ (incorrect).

- **5-Excellent**: Frame clearly satisfies the temporal position and motion implied by the instruction.
- **4-Minor Flaws**: Mostly correct, but small logical gaps or visual mismatches.
- **3-Partial**: Some elements fit, but major spatial/temporal inconsistencies exist.
- **2-Poor**: Few signs of correct temporal placement; largely incorrect.
- **1-Non-Compliant**: Frame bears no relation to the instruction or context.

## Output Format
Describe how the frame aligns (or fails) with the instruction and reference frames.  
Then output:

{{
  "instruction_score": X,
  "reasoning": 1. Detect Instruction Following 2. Expected Visual Caption 3. Instruction Following Match 4. Decision
}}
"""

prompt_consist_multi = """
You are a professional digital artist and image-evaluation specialist.

## Inputs
1. **Multiple Source Images**
2. **Composite Image**: final output
3. **Modification Instruction**: {instruct}

## Objective
Assess **visual consistency** between the composite image and the chosen **background source**. Elements not specified for change should remain unchanged.

## A. Consistency Score (1-5)
Mark each aspect ✔ (consistent) or ✘ (inconsistent).

- **5-Perfect**: All non-instructed details (layout, lighting, identity, etc.) match the background exactly.
- **4-Minor Differences**: One small non-edited detail differs slightly.
- **3-Noticeable Differences**: One clear non-instruction element is altered.
- **2-Significant Differences**: Two or more unintended changes.
- **1-Severe Differences**: Multiple major discrepancies in scene layout, lighting, or identity.

## Output Format
1. Identify which source image serves as the background.  
2. List consistency checks (✔/✘) with brief notes.  
3. Output:

{{
  "consistency_score": X,
  "reasoning": 1. Detect Consistency 2. Expected Visual Caption 3. Consistency Match 4. Decision
}}
"""

prompt_instruction_multi = """
You are a professional digital artist and image-evaluation specialist.

## Inputs
1. **Multiple Source Images**
2. **Composite Image**: final output
3. **Modification Instruction**: {instruct}

## Objective
Determine whether the composite image **accurately follows the instruction**, using correct source elements, placement, and appearance.

## A. Instruction-Compliance Score (1-5)
Mark each aspect ✔ (correct) or ✘ (incorrect).

- **5-Excellent**: Every requested change is present, accurate, and uses the correct source.
- **4-Minor Issues**: One small mismatch (e.g., slight appearance variance).
- **3-Partial**: Key aspects missing or incorrect, though some instruction parts are satisfied.
- **2-Poor**: Most instruction details are wrong or incomplete.
- **1-Non-Compliant**: Instruction is ignored or misinterpreted.

## Output Format
Explain requested changes, verify their presence and correctness, and note omissions or errors.  
Then output:

{{
  "instruction_score": X,
  "reasoning": 1. Detect Instruction Following 2. Expected Visual Caption 3. Instruction Following Match 4. Decision
}}
"""