# Calbi Safety Model

Calbi enforces a strict dual-system architecture:
> **The AI proposes. The runtime disposes.**

## Non-Negotiable Guarantees

1. **No Code Execution**: The LLM generates declarative JSON/YAML Playbooks, never raw Python or shell code.
2. **Whitelisted AST Templating**: Template expressions `{{ ... }}` are evaluated using an AST whitelist. Dunder attributes, imports, arbitrary calls, and comprehensions raise `TemplateError`.
3. **Hard Policy Enforcement**: Permissions are enforced immediately before each step executes in the runtime, not solely at validation or in prompt instructions.
4. **Mandatory Approvals**: Tools marked `DESTRUCTIVE`, `EXTERNAL`, `FINANCIAL`, `DEPLOY` or with `requires_approval=True` require human approval via `policy.approver` at ALL autonomy levels (including Autonomous and Adaptive).
5. **Budgets**: `max_steps`, `max_replans`, `max_total_attempts`, and `max_runtime_s` are hard budget limits checked in code.
