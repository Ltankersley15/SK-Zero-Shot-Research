from __future__ import annotations

from dataclasses import dataclass
from time import time


@dataclass(frozen=True)
class TraceEvent:
    stamp_sec: float
    name: str
    payload: dict


class Trace:
    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def add(self, name: str, **payload) -> None:
        self.events.append(TraceEvent(time(), str(name), dict(payload)))

