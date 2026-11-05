# Calbi Playbook Composition & Concurrent For-Each

Calbi provides first-class modular workflow composition mechanisms within `calbi.core.composition`:
1. **Subplaybooks (`register_subplaybook`)**: Register complete playbooks as reusable tools inside other playbooks, complete with automatic risk inheritance, schema synthesis, and recursion depth protection.
2. **Concurrent For-Each (`register_map_tool`)**: Wrap any registered tool (synchronous or asynchronous) into a bounded, concurrent batching tool with deterministic order preservation and strict item limits.

Both mechanisms adhere to Calbi's core independence rule: `calbi.core` never imports from `calbi.ai` or `calbi.memory`.

---

## Architecture & Guarantees

- **Seamless DAG Composition**: Subplaybooks can be invoked as regular steps in parent playbooks; step inputs and outputs pass cleanly through standard templating expressions (`{{ steps.<id>.output.<key> }}`).
- **Risk & Approval Inheritance**: The registered subplaybook tool's effective risk matches the highest `Risk` among all constituent steps (`READ < WRITE < DESTRUCTIVE < EXTERNAL < FINANCIAL < DEPLOY`). If any step requires approval, the subplaybook requires approval.
- **Task-Local Recursion Guard**: Call depth is tracked via `contextvars.ContextVar[int]` across async tasks. If mutual or self-recursion exceeds `max_depth` (default `3`), execution immediately raises `SubPlaybookError` without risking call stack exhaustion.
- **Bounded Concurrency**: Concurrent item processing in `register_map_tool` is bounded by `asyncio.Semaphore(max_concurrency)`.
- **Order Preservation**: Output results in `register_map_tool` strictly match the input items list order.
- **Payload Bound Check**: The number of items processed in a single map tool invocation cannot exceed `max_items` (default `50`), preventing unbounded memory and execution spikes.

---

## Subplaybooks (`register_subplaybook`)

### Function Signature
```python
def register_subplaybook(
    registry: ToolRegistry,
    playbook: Playbook,
    runtime: Runtime,
    *,
    name: str | None = None,
    description: str | None = None,
    max_depth: int = 3,
) -> ToolSpec: ...
```

### Risk & Metadata Inheritance Rules
- **Risk**: Calculated as `max(step.risk for step in playbook.steps)`. For example, if a subplaybook contains three `Risk.READ` steps and one `Risk.DESTRUCTIVE` step, the subplaybook tool is assigned `Risk.DESTRUCTIVE`.
- **Approval**: `requires_approval=True` if any step tool requires approval.
- **Idempotency**: `idempotent=True` only if all constituent step tools are idempotent.
- **Schemas**:
  - `input_schema`: Derived from `playbook.inputs`, preserving parameter types, descriptions, defaults, and required flags.
  - `output_schema`: Derived from `playbook.outputs` as an object schema mapping output keys.

### Usage Example
```python
from calbi.core.composition import register_subplaybook
from calbi.core.policy import Policy
from calbi.core.registry import ToolRegistry
from calbi.core.runtime import Runtime
from calbi.core.schema import InputSpec, Playbook, Step

reg = ToolRegistry()


@reg.tool(name="calc_tax")
def calc_tax(amount: float) -> dict[str, float]:
    return {"tax": amount * 0.15}


# 1. Define subplaybook
sub_pb = Playbook(
    name="tax_service",
    objective="Compute standard tax on order amount",
    inputs={"amount": InputSpec(type="number", description="Order subtotal")},
    steps=[Step(id="calc", tool="calc_tax", inputs={"amount": "{{ inputs.amount }}"})],
    outputs={"tax_amount": "{{ steps.calc.output.tax }}"},
)

rt = Runtime(reg, Policy())

# 2. Register subplaybook as a reusable tool
register_subplaybook(reg, sub_pb, rt, name="tax_calculator", max_depth=3)

# 3. Use registered subplaybook inside parent playbook
parent_pb = Playbook(
    name="checkout_flow",
    inputs={"order_total": InputSpec(type="number")},
    steps=[
        Step(
            id="compute_tax",
            tool="tax_calculator",
            inputs={"amount": "{{ inputs.order_total }}"},
        )
    ],
    outputs={"final_tax": "{{ steps.compute_tax.output.tax_amount }}"},
)

# 4. Execute parent playbook
# run_state = await rt.run(parent_pb, inputs={"order_total": 200.0})
# run_state.outputs["final_tax"] == 30.0
```

---

## Concurrent For-Each (`register_map_tool`)

### Function Signature
```python
def register_map_tool(
    registry: ToolRegistry,
    base_tool_name: str,
    *,
    name: str | None = None,
    items_param: str = "items",
    item_arg: str,
    max_items: int = 50,
    max_concurrency: int = 8,
) -> ToolSpec: ...
```

### Behavior & Semantics
- **Concurrency Control**: Executes up to `max_concurrency` concurrent tasks simultaneously using `asyncio.Semaphore`.
- **Sync & Async Support**: Transparently awaits async tools on the event loop and dispatches synchronous tools to worker threads via `asyncio.to_thread`.
- **Static Arguments**: Supports passing additional static kwargs alongside the collection (e.g. prefix, timeout, flags).
- **Result Schema**: Returns a dictionary conforming to:
  ```json
  {
    "type": "object",
    "properties": {
      "results": {"type": "array", "description": "Mapped execution results in order."},
      "count": {"type": "integer", "description": "Number of items processed."}
    },
    "required": ["results", "count"]
  }
  ```

### Usage Example
```python
import asyncio
from calbi.core.composition import register_map_tool
from calbi.core.registry import ToolRegistry

reg = ToolRegistry()


@reg.tool(name="fetch_status")
async def fetch_status(service_name: str) -> dict[str, str]:
    await asyncio.sleep(0.01)
    return {"service": service_name, "status": "healthy"}


# Register mapped tool
map_spec = register_map_tool(
    reg,
    base_tool_name="fetch_status",
    name="batch_health_check",
    items_param="services",
    item_arg="service_name",
    max_items=50,
    max_concurrency=4,
)

# Invocation
# response = await map_spec.fn(services=["auth", "billing", "search"])
# response == {
#     "results": [
#         {"service": "auth", "status": "healthy"},
#         {"service": "billing", "status": "healthy"},
#         {"service": "search", "status": "healthy"},
#     ],
#     "count": 3,
# }
```

---

## Error Handling

| Scenario | Raised Exception | Resolution / Behavior |
|---|---|---|
| Recursion depth exceeds `max_depth` | `SubPlaybookError` | Execution halts immediately; context depth resets cleanly. |
| Subplaybook run status != `SUCCEEDED` | `SubPlaybookError` | Includes step ID and underlying failure message. |
| Items count > `max_items` | `CalbiError` | Call rejected before processing any items. |
| Missing `items_param` argument | `CalbiError` | Call rejected before execution. |
| Item argument not a `list` or `tuple` | `CalbiError` | Call rejected before execution. |
| Tool already registered under target name | `DuplicateToolError` | Registration rejected to avoid shadowing. |
| Base tool not found in registry | `UnknownToolError` | Registration fails during initialization. |
