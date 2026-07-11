import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agent import AcademicAgent


class DummyChain:
    def __init__(self):
        self.calls = []

    def invoke(self, question):
        self.calls.append(question)
        return f"Respuesta para: {question}"


class DummyRecorder:
    def __init__(self):
        self.records = []

    def record_request(self, **kwargs):
        self.records.append(kwargs)
        return kwargs


class AgentMemoryTests(unittest.TestCase):
    def test_history_is_kept_between_turns(self):
        chain = DummyChain()
        recorder = DummyRecorder()
        agent = AcademicAgent(chain=chain, recorder=recorder, max_history=2)

        first = agent.answer("¿Qué dice el reglamento?")
        second = agent.answer("¿Y qué pasa con la titulación?")

        self.assertEqual(first["response"], "Respuesta para: ¿Qué dice el reglamento?")
        self.assertEqual(second["response"], "Respuesta para: ¿Y qué pasa con la titulación?")
        self.assertEqual(len(agent.history), 2)
        self.assertEqual(agent.history[0]["question"], "¿Qué dice el reglamento?")
        self.assertEqual(agent.history[1]["question"], "¿Y qué pasa con la titulación?")


if __name__ == "__main__":
    unittest.main()
