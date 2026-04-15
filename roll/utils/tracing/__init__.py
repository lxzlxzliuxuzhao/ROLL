"""
Tracing utilities for ROLL.

This module provides standardized naming conventions, phase definitions,
and HTML export for tracing spans and metrics.
"""

from roll.utils.tracing.core import (
    NullTraceManager,
    NullTraceSpan,
    TraceContext,
    TraceManager,
    TraceSpan,
    TracingConfig,
    get_trace_manager,
)
from roll.utils.tracing.exporter import export_trace_directory, export_trace_step
from roll.utils.tracing.phases import (
    DEFAULT_STEP_PHASES,
    GENERATION_SUBPHASES,
    TRACE_PHASE_DEFINITIONS,
    TracePhaseDefinition,
)

__all__ = [
    # Core tracing
    "NullTraceManager",
    "NullTraceSpan",
    "TraceContext",
    "TraceManager",
    "TraceSpan",
    "TracingConfig",
    "get_trace_manager",
    # Phase definitions
    "TracePhaseDefinition",
    "TRACE_PHASE_DEFINITIONS",
    "DEFAULT_STEP_PHASES",
    "GENERATION_SUBPHASES",
    # Export
    "export_trace_directory",
    "export_trace_step",
]
