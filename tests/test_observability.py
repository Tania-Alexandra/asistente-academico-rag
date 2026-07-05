import os
import sys
import unittest
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from observability import ObservabilityRecorder, sanitize_text, evaluate_response_against_expected


class ObservabilityRecorderTests(unittest.TestCase):
    def test_sanitize_text_redacts_sensitive_data(self):
        text = "Contacto: juan@example.com, tel 987654321"
        redacted = sanitize_text(text)
        self.assertNotIn("juan@example.com", redacted)
        self.assertNotIn("987654321", redacted)
        self.assertIn("[REDACTED_EMAIL]", redacted)
        self.assertIn("[REDACTED_PHONE]", redacted)

    def test_recording_metrics_and_summary(self):
        with TemporaryDirectory() as tmpdir:
            recorder = ObservabilityRecorder(base_dir=tmpdir)
            recorder.record_request(
                question="¿Qué dice el reglamento?",
                response="El reglamento establece la evaluación.",
                latency_ms=320.0,
                success=True,
                error=None,
                tokens=120,
                resource_usage={"cpu_percent": 12.5, "memory_mb": 80.0},
            )
            recorder.record_request(
                question="¿Qué dice el reglamento?",
                response="El reglamento establece la evaluación.",
                latency_ms=350.0,
                success=True,
                error=None,
                tokens=125,
                resource_usage={"cpu_percent": 13.0, "memory_mb": 82.0},
            )

            summary = recorder.get_metrics_summary()
            self.assertEqual(summary["request_count"], 2)
            self.assertGreaterEqual(summary["latency_ms"], 320.0)
            self.assertGreaterEqual(summary["consistency_score"], 0.95)
            self.assertEqual(summary["error_rate"], 0.0)

    def test_accuracy_evaluation(self):
        score = evaluate_response_against_expected(
            "El reglamento establece el proceso de titulación.",
            ["reglamento", "titulación"],
        )
        self.assertGreaterEqual(score, 0.8)


if __name__ == "__main__":
    unittest.main()
