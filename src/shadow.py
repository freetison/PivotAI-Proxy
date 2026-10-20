"""Shadow mode: run the cascade for comparison, but answer with the final level.

Why this exists: before trusting the JEV thresholds, you want to know how often
the cascade's answer differs from the answer the client would have got going
straight to the final tier — and whether the difference mattered. Shadow mode
collects that evidence without changing what the client receives.

Two rules keep it honest:

* the client always gets the direct final-level answer, so the comparison is
  never an experiment on live traffic;
* the extra spend is measured and reported (``shadow_cost_usd``), because a
  request served in shadow mode costs strictly more than the same request in
  active mode.

Nothing is persisted unless ``SHADOW_STORE`` is enabled: these records contain
prompts and answers, so they stay opt-in and local.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("ai_cascade_proxy.shadow")

#: Where pairs are appended when the store is enabled but no path was given.
DEFAULT_SHADOW_STORE_PATH = "data/shadow_pairs.jsonl"

#: ``active`` answers with the cascade; ``shadow`` answers with the final level and
#: keeps the cascade's answer for offline comparison.
SHADOW_MODES = ("active", "shadow")
DEFAULT_SHADOW_MODE = "active"


def _answer_text(response: dict | None) -> str:
    """Best-effort readable answer, for comparison and storage only."""
    if not isinstance(response, dict):
        return ""
    try:
        content = response["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block["text"])
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return ""


@dataclass(frozen=True)
class ShadowRecord:
    """One ``(cascade answer, direct final-level answer)`` pair."""

    request_id: str
    created_at: str
    cascade_model: str
    cascade_response: str
    shadow_model: str
    shadow_response: str
    #: Levels whose provider call failed while running the cascade, if any.
    failed_levels: tuple[str, ...] = field(default_factory=tuple)

    @property
    def identical(self) -> bool:
        """Same answer text. Not a quality judgement — just a cheap first filter."""
        return self.cascade_response.strip() == self.shadow_response.strip()

    def as_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "created_at": self.created_at,
            "cascade_model": self.cascade_model,
            "cascade_response": self.cascade_response,
            "shadow_model": self.shadow_model,
            "shadow_response": self.shadow_response,
            "identical": self.identical,
            "failed_levels": list(self.failed_levels),
        }


class ShadowStore:
    """Append-only local JSONL store. One line per request, written under a lock.

    Deliberately not a database: the records are meant to be read by an offline
    script and deleted afterwards.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, record: ShadowRecord) -> None:
        line = json.dumps(record.as_dict(), ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def store_from_settings(settings) -> ShadowStore | None:
    """The configured store, or None when ``SHADOW_STORE`` is off."""
    if not getattr(settings, "shadow_store", False):
        return None
    path = getattr(settings, "shadow_store_path", "") or DEFAULT_SHADOW_STORE_PATH
    return ShadowStore(path)


def record_for(request_id: str, *, cascade_model: str, cascade_response: dict | None,
               shadow_model: str, shadow_response: dict, failed_levels: tuple[str, ...] = ()) -> ShadowRecord:
    return ShadowRecord(
        request_id=request_id,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        cascade_model=cascade_model,
        cascade_response=_answer_text(cascade_response),
        shadow_model=shadow_model,
        shadow_response=_answer_text(shadow_response),
        failed_levels=failed_levels,
    )
