# Writing Tools for Calbi

Tools are standard Python functions or coroutines registered with `@engine.tool(...)` or `engine.register(...)`.

## Guidelines

1. **Type Annotations**: Always annotate parameters and return types. Calbi automatically derives JSON Schema from type hints for LLM discovery and validation.
2. **Docstrings & Descriptions**: Provide a clear docstring or `description="..."` explaining what the tool does and when to use it.
3. **Risk Levels**: Specify the appropriate risk level (`Risk.READ`, `Risk.WRITE`, `Risk.DESTRUCTIVE`, `Risk.EXTERNAL`, `Risk.FINANCIAL`, `Risk.DEPLOY`).
4. **Idempotency**: Set `idempotent=True` if the tool can be safely retried without side effects.
5. **JSON-Serializable Outputs**: Step return values must be serializable to JSON.

## Example

```python
from calbi import Risk


@engine.tool(
    name="documents.extract_tables",
    description="Extract tables from PDF files into CSV format",
    risk=Risk.READ,
    idempotent=True,
    limitations="Only supports text-based PDFs. Scanned image PDFs will fail.",
)
async def extract_tables(pdf_path: str, max_pages: int = 10) -> dict[str, list[str]]:
    # Async or sync implementation
    return {"csv_paths": ["/tmp/table1.csv"]}
```
