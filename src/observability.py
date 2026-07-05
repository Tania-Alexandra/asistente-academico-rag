import json
import logging
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b(?:\+?56)?(?:9|2|6|7)\d{8}\b")
TOKEN_RE = re.compile(r"(sk-[A-Za-z0-9]{12,}|ghp_[A-Za-z0-9]{20,}|Bearer\s+[A-Za-z0-9._-]+)")
KEY_RE = re.compile(r"(api[_-]?key|token|secret)\s*[:=]\s*['\"]?([A-Za-z0-9._-]{3,})", re.IGNORECASE)


def sanitize_text(text: Optional[Any]) -> str:
    if text is None:
        return ""
    text = str(text)
    text = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = PHONE_RE.sub("[REDACTED_PHONE]", text)
    text = TOKEN_RE.sub("[REDACTED_TOKEN]", text)
    text = KEY_RE.sub(r"\1=[REDACTED_KEY]", text)
    return text


def normalize_text(text: Optional[Any]) -> str:
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFKD", str(text).lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", normalized).strip()


def evaluate_response_against_expected(response: Optional[Any], expected_keywords: Optional[List[str]]) -> float:
    if not expected_keywords:
        return 1.0
    response_text = normalize_text(response)
    if not response_text:
        return 0.0
    matches = 0
    for keyword in expected_keywords:
        if normalize_text(keyword) and normalize_text(keyword) in response_text:
            matches += 1
    return round(matches / len(expected_keywords), 2)


def should_block_query(question: Optional[Any]) -> bool:
    if question is None:
        return False
    normalized = normalize_text(question)
    blocked_patterns = [
        "ignore previous instructions",
        "system prompt",
        "developer mode",
        "bypass",
        "drop table",
        "rm -rf",
        "api key",
    ]
    return any(pattern in normalized for pattern in blocked_patterns)


class ObservabilityRecorder:
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent.parent).resolve()
        self.observability_dir = self.base_dir / "observability"
        self.observability_dir.mkdir(exist_ok=True)
        self.metric_file = self.observability_dir / "metrics.jsonl"
        self.summary_file = self.observability_dir / "metrics_summary.json"
        self.log_file = self.observability_dir / "agent_events.log"
        self._records: List[Dict[str, Any]] = []
        self._load_existing_records()
        self._setup_logger()

    def _load_existing_records(self) -> None:
        if not self.metric_file.exists():
            return
        for line in self.metric_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    self._records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def _setup_logger(self) -> None:
        self.logger = logging.getLogger("assistant_rag_observability")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if self.logger.handlers:
            self.logger.handlers.clear()
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
        self.logger.addHandler(stream_handler)

    def get_resource_usage(self) -> Dict[str, float]:
        if psutil is None:
            return {"cpu_percent": 0.0, "memory_mb": 0.0}
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            "cpu_percent": round(psutil.cpu_percent(interval=None), 2),
            "memory_mb": round(memory_info.rss / (1024 * 1024), 2),
        }

    def record_request(
        self,
        *,
        question: Optional[str],
        response: Optional[str],
        latency_ms: float,
        success: bool,
        error: Optional[str] = None,
        tokens: Optional[int] = None,
        resource_usage: Optional[Dict[str, float]] = None,
        expected_keywords: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        sanitized_question = sanitize_text(question)
        sanitized_response = sanitize_text(response)
        resource_data = resource_usage or self.get_resource_usage()
        accuracy_score = evaluate_response_against_expected(sanitized_response, expected_keywords)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": sanitized_question,
            "response": sanitized_response,
            "latency_ms": round(latency_ms, 2),
            "success": success,
            "error": sanitize_text(error) if error else None,
            "tokens": tokens or max(1, len((sanitized_response or "").split()) + len((sanitized_question or "").split())),
            "resource_usage": resource_data,
            "accuracy": accuracy_score,
        }
        self._records.append(record)
        with self.metric_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        event_message = json.dumps(
            {
                "event": "request_completed",
                "success": success,
                "latency_ms": record["latency_ms"],
                "tokens": record["tokens"],
                "error": record["error"],
            },
            ensure_ascii=False,
        )
        if success:
            self.logger.info(event_message)
        else:
            self.logger.error(event_message)

        self._write_summary()
        return record

    def get_metrics_summary(self) -> Dict[str, Any]:
        if not self._records:
            return {
                "request_count": 0,
                "success_rate": 0.0,
                "precision_score": 0.0,
                "latency_ms": 0.0,
                "consistency_score": 1.0,
                "error_rate": 0.0,
                "avg_tokens": 0,
                "avg_cpu_percent": 0.0,
                "avg_memory_mb": 0.0,
            }

        success_rate = sum(1 for item in self._records if item.get("success")) / len(self._records)
        precision_score = sum(item.get("accuracy", 0.0) for item in self._records) / len(self._records)
        latency_values = [item.get("latency_ms", 0.0) for item in self._records if item.get("latency_ms") is not None]
        token_values = [item.get("tokens", 0) for item in self._records if item.get("tokens") is not None]
        cpu_values = [item.get("resource_usage", {}).get("cpu_percent", 0.0) for item in self._records]
        memory_values = [item.get("resource_usage", {}).get("memory_mb", 0.0) for item in self._records]

        summary = {
            "request_count": len(self._records),
            "success_rate": round(success_rate, 2),
            "precision_score": round(precision_score, 2),
            "latency_ms": round(sum(latency_values) / len(latency_values), 2),
            "consistency_score": round(self._calculate_consistency(), 2),
            "error_rate": round(1 - success_rate, 2),
            "avg_tokens": round(sum(token_values) / len(token_values), 2),
            "avg_cpu_percent": round(sum(cpu_values) / len(cpu_values), 2),
            "avg_memory_mb": round(sum(memory_values) / len(memory_values), 2),
        }
        return summary

    def _calculate_consistency(self) -> float:
        if len(self._records) < 2:
            return 1.0
        scores = []
        for previous, current in zip(self._records, self._records[1:]):
            prev_text = normalize_text(previous.get("response"))
            curr_text = normalize_text(current.get("response"))
            if prev_text and curr_text:
                if prev_text == curr_text:
                    scores.append(1.0)
                else:
                    scores.append(round(1.0 - (sum(1 for a, b in zip(prev_text, curr_text) if a != b) / max(len(prev_text), len(curr_text))), 2))
        return round(sum(scores) / len(scores), 2) if scores else 1.0

    def _write_summary(self) -> None:
        summary = self.get_metrics_summary()
        self.summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    def get_recent_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.log_file.exists():
            return []
        entries: List[Dict[str, Any]] = []
        for line in self.log_file.read_text(encoding="utf-8").splitlines()[-limit:]:
            if not line.strip():
                continue
            timestamp, level, message = line.split(" | ", 2)
            try:
                entries.append({"timestamp": timestamp, "level": level, "message": message})
            except ValueError:
                continue
        return entries

    def export_dashboard_payload(self) -> Dict[str, Any]:
        return {
            "summary": self.get_metrics_summary(),
            "records": self._records[-20:],
            "logs": self.get_recent_logs(20),
        }
