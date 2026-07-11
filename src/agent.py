import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from observability import ObservabilityRecorder, should_block_query


class AcademicAgent:
    """Agente académico ligero con historial de conversación y trazabilidad."""

    def __init__(self, chain: Any, recorder: Optional[ObservabilityRecorder] = None, max_history: int = 3):
        self.chain = chain
        self.recorder = recorder or ObservabilityRecorder()
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []

    def _build_prompt_with_history(self, question: str) -> str:
        if not self.history:
            return question

        history_lines = []
        for entry in self.history[-self.max_history:]:
            history_lines.append(f"- Usuario: {entry['question']}")
            history_lines.append(f"- Asistente: {entry['response']}")

        return question

    def _append_history(self, question: str, response: str) -> None:
        self.history.append(
            {
                "question": question,
                "response": response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def answer(self, question: str, expected_keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        if should_block_query(question):
            self.recorder.record_request(
                question=question,
                response="Solicitud bloqueada por guardrail de seguridad.",
                latency_ms=0.0,
                success=False,
                error="Pregunta bloqueada por seguridad",
                tokens=0,
                expected_keywords=["seguridad"],
            )
            return {
                "response": "Solicitud bloqueada por guardrail de seguridad.",
                "success": False,
                "latency_ms": 0.0,
            }

        start = time.time()
        question_to_process = self._build_prompt_with_history(question)

        try:
            response = self.chain.invoke(question_to_process)
            latency_ms = (time.time() - start) * 1000
            record = self.recorder.record_request(
                question=question,
                response=response,
                latency_ms=latency_ms,
                success=True,
                tokens=len((question + response).split()),
                expected_keywords=expected_keywords or ["reglamento", "titulación"],
            )
            self._append_history(question, response)
            return {
                "response": response,
                "success": True,
                "latency_ms": latency_ms,
                "record": record,
            }
        except Exception as exc:
            latency_ms = (time.time() - start) * 1000
            record = self.recorder.record_request(
                question=question,
                response=str(exc),
                latency_ms=latency_ms,
                success=False,
                error=str(exc),
                tokens=0,
            )
            return {
                "response": f"❌ Error al procesar: {exc}",
                "success": False,
                "latency_ms": latency_ms,
                "record": record,
            }
