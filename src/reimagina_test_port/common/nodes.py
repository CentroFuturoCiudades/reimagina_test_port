"""Helpers every Lyra node uses: stage timing and nullable table columns."""

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from lyra.sdk import TableColumn, Unit

logger = logging.getLogger(__name__)


@contextmanager
def timed(stage: str, log: Any = logger) -> Iterator[None]:
    """Log how long a stage of a job took (to the job's logger in Lyra)."""
    start = time.perf_counter()
    yield
    log.info("%s: %.2fs", stage, time.perf_counter() - start)


def column(name: str, kind: str, unit: Unit | None, description: str) -> TableColumn:
    """Nullable column."""
    return TableColumn(
        name=name, type=kind, unit=unit, nullable=True, description=description
    )
