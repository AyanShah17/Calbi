# Playbook Schema Reference

Playbooks are Directed Acyclic Graphs (DAGs) of tasks with typed inputs, conditional branching, retries, and output mapping.

```yaml
name: analyze_dataset
version: 1
objective: Analyze dataset and create report

inputs:
  dataset:
    type: string
    required: true
    description: Path to CSV dataset

steps:
  - id: inspect
    tool: dataset.inspect
    inputs:
      file: "{{ inputs.dataset }}"
    retry:
      max_attempts: 3
      backoff: exponential
      base_delay_s: 0.5
      max_delay_s: 30.0
    timeout_s: 60.0
    on_failure: fail

  - id: analyze
    tool: dataset.analyze
    depends_on:
      - inspect
    condition: "{{ steps.inspect.output.valid }}"
    inputs:
      file: "{{ inputs.dataset }}"
    on_failure: replan

outputs:
  report: "{{ steps.analyze.output.path }}"
```
