"""Raw LLM prompt/response transcript recording for providers.

每次请求（含重试的每一次尝试）追加一行 JSON 到 transcript 文件，
用于事后核查生成过程（reward hacking 复查、失败归因、实验复盘）。
落盘路径由 cli 按运行输出目录解析，也可用环境变量
PYRAMIDKERNEL_TRANSCRIPT 覆盖；取值为 off/0/false/none 时关闭记录。
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

_DISABLE_VALUES = {"off", "0", "false", "none", "disable", "disabled"}


def resolve_transcript_path(output_dir: str | Path | None) -> str | None:
    raw = os.environ.get("PYRAMIDKERNEL_TRANSCRIPT", "").strip()
    if raw.lower() in _DISABLE_VALUES:
        return None
    if raw:
        return raw
    if output_dir is None:
        return None
    return str(Path(output_dir) / "transcript.jsonl")


class TranscriptRecorder:
    """Append-only JSONL writer；首次写入时才创建目录。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def record(
        self,
        *,
        provider: str,
        model: str,
        wire_api: str,
        purpose: str,
        attempt: int,
        system_prompt: str,
        user_text: str,
        response_text: str | None,
        error: str | None,
        started_at: float,
        finished_at: float,
    ) -> None:
        entry: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started_at)),
            "provider": provider,
            "model": model,
            "wire_api": wire_api,
            "purpose": purpose,
            "attempt": attempt,
            "system_prompt": system_prompt,
            "user_text": user_text,
            "response_text": response_text,
            "error": error,
            "latency_seconds": round(finished_at - started_at, 6),
        }
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
