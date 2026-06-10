import pickle
from typing import Optional

import streamlit as st

# Import the necessary types from the novascientist package
from novascientist.global_state import NovaScientistState


def load_novascientist_state(filepath: str) -> Optional[NovaScientistState]:
    """Load a NovaScientistState from a pickle file."""
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading state file: {e}")
        return None


def load_novascientist_state_by_goal(goal: str) -> Optional[NovaScientistState]:
    """Load the latest NovaScientistState for a given research goal."""
    try:
        return NovaScientistState.load_latest(goal=goal)
    except Exception as e:
        st.error(f"Error loading state for goal '{goal}': {e}")
        return None


def get_available_states() -> list[str]:
    """Get all available research goals from the goal-based directory structure."""
    try:
        # Use the NovaScientistState method to get all available goals
        goals_and_dirs = NovaScientistState.list_all_goals()
        # Return just the goal texts (first element of each tuple)
        return [goal for goal, _ in goals_and_dirs]
    except Exception as e:
        st.error(f"Error getting available states: {e}")
        return []
