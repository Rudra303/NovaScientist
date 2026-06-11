from typing import Any, Dict

class MockLLM:
    """Mock LLM for testing purposes."""
    def __init__(self, responses: list[str] = None):
        self.responses = responses or ["Mock response"]
        self.call_count = 0

    async def ainvoke(self, *args, **kwargs) -> Any:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response

class MockState:
    """Mock State for testing."""
    def __init__(self, goal: str = "Test Goal"):
        self.goal = goal
        self.generated_hypotheses = []
        self.reviewed_hypotheses = []
        self.evolved_hypotheses = []
        self.tournament = None
        self.meta_reviews = []
        self.reflection_queue = []
        self.supervisor_decisions = []
        self.final_report = None

    def save(self) -> str:
        return "mock_path.pkl"
