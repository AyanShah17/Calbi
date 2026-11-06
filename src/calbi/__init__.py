"""Calbi — Autonomous Playbook Engine."""

from __future__ import annotations

from calbi.ai.strategies import (
    StrategyCandidate,
    StrategyExplorationFailed,
    StrategyExplorationResult,
    StrategyExplorer,
    StrategyTrial,
)
from calbi.core.composition import (
    SubPlaybookError,
    register_map_tool,
    register_subplaybook,
)
from calbi.core.events import Event, EventBus
from calbi.core.executor import (
    ExecutionIsolationError,
    InProcessExecutor,
    StepExecutor,
    SubprocessExecutor,
)
from calbi.core.observability import (
    EventSink,
    JSONLEventSink,
    Redactor,
    SpanRecord,
    TraceCollector,
)
from calbi.core.policy import ApprovalRequest, Autonomy, Policy
from calbi.core.registry import (
    DuplicateToolError,
    ToolRegistry,
    ToolSpec,
    UnknownToolError,
)
from calbi.core.runner import Runner, Runtime
from calbi.core.schema import (
    CalbiError,
    FailurePolicy,
    InputSpec,
    Playbook,
    RetryPolicy,
    Risk,
    SchemaError,
    Step,
    dump_playbook,
    load_playbook,
)
from calbi.core.state import RunState, RunStatus, StepState, StepStatus
from calbi.core.validator import Issue, ValidationReport, validate
from calbi.facade import Calbi, set_delegate_handler
from calbi.memory.improvement import (
    ImprovementManager,
    ImprovementProposal,
    ProposalStatus,
)

__version__ = "0.2.0"

__all__ = [
    "ApprovalRequest",
    "Autonomy",
    "Calbi",
    "CalbiError",
    "DuplicateToolError",
    "Event",
    "EventBus",
    "EventSink",
    "ExecutionIsolationError",
    "FailurePolicy",
    "ImprovementManager",
    "ImprovementProposal",
    "InProcessExecutor",
    "InputSpec",
    "Issue",
    "JSONLEventSink",
    "Playbook",
    "Policy",
    "ProposalStatus",
    "Redactor",
    "RetryPolicy",
    "Risk",
    "RunState",
    "RunStatus",
    "Runner",
    "Runtime",
    "SchemaError",
    "SpanRecord",
    "Step",
    "StepExecutor",
    "StepState",
    "StepStatus",
    "StrategyCandidate",
    "StrategyExplorationFailed",
    "StrategyExplorationResult",
    "StrategyExplorer",
    "StrategyTrial",
    "SubPlaybookError",
    "SubprocessExecutor",
    "ToolRegistry",
    "ToolSpec",
    "TraceCollector",
    "UnknownToolError",
    "ValidationReport",
    "dump_playbook",
    "load_playbook",
    "register_map_tool",
    "register_subplaybook",
    "set_delegate_handler",
    "validate",
]
