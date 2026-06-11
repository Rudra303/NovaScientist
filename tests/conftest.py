import pytest
from tests.testing import MockLLM, MockState

@pytest.fixture
def mock_llm():
    """Returns a mock LLM instance."""
    return MockLLM()

@pytest.fixture
def mock_state():
    """Returns a mock NovaScientistState instance."""
    return MockState()
