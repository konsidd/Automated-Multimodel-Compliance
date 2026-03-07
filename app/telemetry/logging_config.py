"""
app/telemetry/logging_config.py
───────────────────────────────
Structured logging using structlog + rich.
"""

from __future__ import annotations

import logging
import sys

import structlog
from rich.console import Console
from rich.logging import RichHandler


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structlog with rich console output."""

    level = getattr(logging, log_level.upper(), logging.INFO)

    # Standard library root logger → rich handler
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[
            RichHandler(
                console=Console(stderr=True),
                rich_tracebacks=True,
                markup=True,
            )
        ],
    )

    # structlog processors
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
