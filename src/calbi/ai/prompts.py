from __future__ import annotations

import json
import re
from typing import Any

PLANNER_SYSTEM_PROMPT = """You are the AI Playbook Creator for Calbi, an autonomous workflow engine.
Your role is to analyze tasks, decompose them into steps, and compose registered tools.

RULES:
1. ONLY tools declared in the AVAILABLE TOOLS manifest exist. You must NEVER invent tools.
   Any playbook referencing undeclared tools will be rejected.
2. Produce only a valid Playbook JSON object matching the Calbi schema.
3. Steps must form a valid DAG with unique lowercase step IDs (e.g. "step_inspect").
4. If a step depends on previous steps, list them in "depends_on".
5. Use template syntax "{{ inputs.<key> }}" or "{{ steps.<step_id>.output.<field> }}".
6. Do NOT output executable code. Output JSON only.

AVAILABLE TOOLS:
{tools_manifest}

HINTS AND BEST PRACTICES:
{hints}
"""

PLANNER_USER_PROMPT = """Create a playbook to achieve the following objective:

TASK:
{task}

DECLARED INPUTS (types and descriptions):
{inputs_schema}

INPUT VALUES:
{truncated_inputs}

CONSTRAINTS:
{constraints}

EXPECTED OUTPUT:
{expected_output}

Output the complete Playbook JSON.
"""

PLANNER_FEEDBACK_PROMPT = """The playbook you generated was rejected by the Playbook Validator.
Fix the errors below and return the updated Playbook JSON.

VALIDATION ISSUES:
{issues}
"""

VERIFIER_SYSTEM_PROMPT = """You are an independent quality verifier for Calbi.
Your role is to assess whether an executed playbook achieved the objective based on outputs.
Output JSON with:
{
  "passed": true / false,
  "reasons": ["detailed reasons"],
  "evidence": ["concrete data from outputs"],
  "confidence": 0.0 to 1.0
}
"""

VERIFIER_USER_PROMPT = """Evaluate this execution against the objective:

OBJECTIVE:
{task}

EXPECTED OUTPUT:
{expected_output}

EXECUTION OUTPUTS:
{outputs}

STEP RESULTS:
{step_results}
"""

REPLANNER_USER_PROMPT = """A previously generated playbook failed during execution.
Create an alternative playbook to accomplish the objective learning from this failure.

ORIGINAL TASK:
{task}

FAILED STEP:
{failed_step}

ERROR DETAILS:
{error_details}

VALID SUCCEEDED STEPS (REUSE THESE STEP IDs UNCHANGED IF POSSIBLE):
{valid_steps}

FAILED APPROACHES TO AVOID:
{failed_approaches}

Output the updated Playbook JSON.
"""

_CODEBLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Extract and parse JSON from text or markdown code fences."""
    text = text.strip()
    match = _CODEBLOCK_RE.search(text)
    if match:
        text = match.group(1).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")
    except json.JSONDecodeError as e:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            sub = text[start : end + 1]
            data = json.loads(sub)
            if isinstance(data, dict):
                return data
        raise ValueError(f"Could not extract valid JSON from LLM response: {e}") from e
