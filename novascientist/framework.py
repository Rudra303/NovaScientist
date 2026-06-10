"""
The overall framework that takes a NovaScientistStateManager from global_state.py,
setups the agents, and organizes the multi-agent system. The framework will be controlled
by a supervisor agent.
"""

import logging
import math
import random
import asyncio

import numpy as np
from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from novascientist.evolution_agent import build_evolution_agent
from novascientist.final_report_agent import build_final_report_agent
from novascientist.generation_agent import (
    CollaborativeConfig,
    IndependentConfig,
    build_generation_agent,
)
from novascientist.global_state import NovaScientistStateManager
from novascientist.literature_review_agent import build_literature_review_agent
from novascientist.meta_review_agent import build_meta_review_agent
from novascientist.reasoning_types import ReasoningType
from novascientist.reflection_agent import build_deep_verification_agent
from novascientist.supervisor_agent import build_supervisor_agent
from novascientist.ml_scheduler import MultiArmedBanditScheduler

# Generally reasoning models are better suited for the scientific reasoning
# tasks entailed by the NovaScientist system.
_SMARTER_LLM_POOL = {
    "o3": ChatOpenAI(model="o3", max_tokens=50_000, max_retries=3),
    "gemini-2.5-pro": ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=1.0,
        max_retries=3,
        max_tokens=50_000,
    ),
    "claude-sonnet-4-20250514": ChatAnthropic(
        model="claude-sonnet-4-20250514", max_tokens=50_000, max_retries=3
    ),
}
_CHEAPER_LLM_POOL = {
    "o4-mini": ChatOpenAI(model="o4-mini", max_tokens=50_000, max_retries=3),
    "gemini-2.5-flash": ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=1.0,
        max_retries=3,
        max_tokens=50_000,
    ),
    # Anthropic doesn't have a good cheaper model
    "claude-sonnet-4-20250514": ChatAnthropic(
        model="claude-sonnet-4-20250514", max_tokens=50_000, max_retries=3
    ),
}


class NovaScientistConfig:
    """
    Configuration for the NovaScientist system.

    Note that the config for GPTResearcher which is used throughout the system
    is defined in `researcher_config.json`.

    Attributes
    ----------
    literature_review_agent_llm : BaseChatModel
        The language model for the literature review. This LLM decides on the research
        subtopics for GPTResearcher.
    generation_agent_llms : dict[str, BaseChatModel]
        The language models for the generation agents
    reflection_agent_llms : dict[str, BaseChatModel]
        The language models for the reflection agents
    evolution_agent_llms : dict[str, BaseChatModel]
        The language models for the evolution agents
    meta_review_agent_llm : BaseChatModel
        The language model for the meta-review. Gemini works best because of the long
        context window that isn't severely rate limited like other providers.
    proximity_agent_embedding_model : Embeddings
        The embedding model for the proximity agent
    specialist_fields : list[str]
        The fields of expertise for generation agents. This list should be expanded
        by the configuration agent.

    """

    def __init__(
        self,
        literature_review_agent_llm: BaseChatModel = _SMARTER_LLM_POOL[
            "claude-sonnet-4-20250514"
        ],
        generation_agent_llms: dict[str, BaseChatModel] = _SMARTER_LLM_POOL,
        reflection_agent_llms: dict[str, BaseChatModel] = _SMARTER_LLM_POOL,
        evolution_agent_llms: dict[str, BaseChatModel] = _SMARTER_LLM_POOL,
        meta_review_agent_llm: BaseChatModel = _CHEAPER_LLM_POOL["gemini-2.5-flash"],
        supervisor_agent_llm: BaseChatModel = _SMARTER_LLM_POOL[
            "claude-sonnet-4-20250514"
        ],
        final_report_agent_llm: BaseChatModel = _SMARTER_LLM_POOL[
            "claude-sonnet-4-20250514"
        ],
        proximity_agent_embedding_model: Embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", dimensions=256
        ),
        specialist_fields: list[str] | None = None,
    ):
        # TODO: Add functionality for overriding GPTResearcher config.
        self.literature_review_agent_llm = literature_review_agent_llm
        self.generation_agent_llms = generation_agent_llms
        self.reflection_agent_llms = reflection_agent_llms
        self.evolution_agent_llms = evolution_agent_llms
        self.meta_review_agent_llm = meta_review_agent_llm
        self.supervisor_agent_llm = supervisor_agent_llm
        self.proximity_agent_embedding_model = proximity_agent_embedding_model
        self.final_report_agent_llm = final_report_agent_llm
        if specialist_fields is None:
            self.specialist_fields = ["biology"]
        else:
            self.specialist_fields = specialist_fields


class NovaScientistFramework:
    """
    The framework that takes a NovaScientistStateManager from global_state.py,
    setups the agents, and organizes the multi-agent system. The framework will be controlled
    by a supervisor agent.
    """

    def __init__(
        self, config: NovaScientistConfig, state_manager: NovaScientistStateManager
    ):
        self.config = config
        self.state_manager = state_manager
        self.bandit_scheduler = MultiArmedBanditScheduler(self.state_manager._state._output_dir)

    def list_generation_llm_names(self) -> list[str]:
        """
        List the names of the generation agents.
        """
        return list(self.config.generation_agent_llms.keys())

    def list_generation_modes(self) -> list[str]:
        """
        List the names of the generation modes.
        """
        return ["independent", "collaborative"]

    def list_reflection_llm_names(self) -> list[str]:
        """
        List the names of the reflection agents.
        """
        return list(self.config.reflection_agent_llms.keys())

    def list_evolution_llm_names(self) -> list[str]:
        """
        List the names of the evolution agents.
        """
        return list(self.config.evolution_agent_llms.keys())

    def list_evolution_modes(self) -> list[str]:
        """
        List the names of the evolution modes.
        """
        return ["evolve_from_feedback", "out_of_the_box"]

    def list_specialist_fields(self) -> list[str]:
        """
        List the names of the specialist fields.
        """
        return self.config.specialist_fields

    def list_reasoning_types(self) -> list[str]:
        """
        List the names of the reasoning types.
        """
        return list(ReasoningType.__members__.keys())

    def get_semantic_communities(
        self, resolution: float = 1.0, min_weight: float = 0.85
    ) -> list[set[str]]:
        """
        Get the semantic communities of the hypotheses.
        """
        self.state_manager.proximity_graph.update_edges()
        return self.state_manager.proximity_graph.get_semantic_communities(
            resolution=resolution, min_weight=min_weight
        )

    async def process_reflection_queue(self) -> None:
        """
        Process all hypotheses in the reflection queue through deep verification.

        This method pops hypotheses from the reflection queue until it's empty,
        runs them through deep verification, and adds the reviewed hypotheses
        to the state manager.
        """
        tasks = []
        while not self.state_manager.reflection_queue_is_empty:
            # This pops from the reflection queue until it's empty
            initial_reflection_state = self.state_manager.next_reflection_state()
            llm_name = random.choice(self.list_reflection_llm_names())
            reflection_agent = build_deep_verification_agent(
                llm=self.config.reflection_agent_llms[llm_name],
                review_llm=self.config.meta_review_agent_llm,
                parallel=False,
                checkpointer=None,
            )
            tasks.append(reflection_agent.ainvoke(initial_reflection_state))
            
        results = await asyncio.gather(*tasks)
        for final_reflection_state in results:
            if final_reflection_state["passed_initial_filter"]:
                self.state_manager.add_reviewed_hypothesis(
                    final_reflection_state["reviewed_hypothesis"]
                )
                self.state_manager.advance_reviewed_hypothesis()

    async def _generate_new_hypothesis(self) -> dict:
        """
        Run the hypothesis generation for a given mode and config using the ML Scheduler.
        """
        mode, reasoning_type_str = self.bandit_scheduler.get_strategy(
            self.list_generation_modes(), self.list_reasoning_types()
        )
        
        if mode == "independent":
            llm_name = random.choice(self.list_generation_llm_names())
            specialist_field = random.choice(self.list_specialist_fields())
            config = IndependentConfig(
                llm=self.config.generation_agent_llms[llm_name],
                reasoning_type=getattr(ReasoningType, reasoning_type_str),
                field=specialist_field,
            )
            first_agent_name = None
        elif mode == "collaborative":
            llm_names = np.random.choice(self.list_generation_llm_names(), 2).tolist()
            specialist_fields = np.random.choice(
                self.list_specialist_fields(), 2
            ).tolist()
            # For collaborative, we use the selected reasoning type for both, or randomly select a second one
            second_rt = random.choice(self.list_reasoning_types())
            reasoning_types = [reasoning_type_str, second_rt]

            agent_names = [
                f"{llm_name}_{field}"
                for llm_name, field in zip(llm_names, specialist_fields)
            ]
            config = CollaborativeConfig(
                agent_names=agent_names,
                agent_fields=dict(zip(agent_names, specialist_fields)),
                agent_reasoning_types={
                    name: getattr(ReasoningType, reasoning_type)
                    for name, reasoning_type in zip(agent_names, reasoning_types)
                },
                llms={
                    name: self.config.generation_agent_llms[llm_name]
                    for name, llm_name in zip(agent_names, llm_names)
                },
                max_turns=10,
            )
            first_agent_name = agent_names[0]

        generation_agent = build_generation_agent(mode, config)
        initial_generation_state = self.state_manager.next_generation_state(
            mode, first_agent_name
        )
        return await generation_agent.ainvoke(initial_generation_state)

    async def start(self, n_hypotheses: int = 8) -> None:
        """
        Starts the NovaScientist system with a fixed number of initial
        hypotheses.
        """
        assert n_hypotheses >= 2, "Must generate at least two hypotheses to start"
        if self.state_manager.is_started:
            raise ValueError(
                "NovaScientist system has already been started. "
                f"Use one of {self.available_actions()} instead!"
            )

        # Perform the initial literature review.
        if not self.state_manager.has_literature_review:
            literature_review_agent = build_literature_review_agent(
                self.config.literature_review_agent_llm
            )
            initial_lit_review_state = self.state_manager.next_literature_review_state(
                # TODO: Make this configurable
                max_subtopics=5
            )
            final_lit_review_state = await literature_review_agent.ainvoke(
                initial_lit_review_state
            )
            self.state_manager.update_literature_review(final_lit_review_state)

        # TODO: Make this async
        _ = await self.generate_new_hypotheses(
            n_hypotheses=max(0, n_hypotheses - self.state_manager.total_hypotheses)
        )

        # Run the EloTournament
        # The top k for the bracket should the nearest power of
        # 2 less than the number of hypotheses and no more than 16.
        k_bracket = min(16, 2 ** math.floor(math.log2(n_hypotheses)))
        # TODO: Figure out the right LLM for this job; should it be different from meta-review?
        # Feels like it should be fixed for the sake of consistency though
        _ = await self.run_tournament(k_bracket=k_bracket)
        _ = await self.run_meta_review(k_bracket=k_bracket)

    async def generate_new_hypotheses(self, n_hypotheses: int = 2) -> None:
        """
        Generate new hypotheses.
        """
        tasks = [self._generate_new_hypothesis() for _ in range(n_hypotheses)]
        results = await asyncio.gather(*tasks)
        
        for final_generation_state in results:
            self.state_manager.add_generated_hypothesis(
                final_generation_state["hypothesis"]
            )
            self.state_manager.advance_hypothesis(kind="generated")

        # Now run through the review queue and perform deep verification
        await self.process_reflection_queue()
        self.state_manager.update_proximity_graph_edges()

    async def evolve_hypotheses(self, n_hypotheses: int = 4) -> None:
        """
        Takes the top (n_hypotheses // 2) hypotheses and evolves them. Also
        randomly selects (n_hypotheses // 2) hypotheses to evolve.
        """
        assert n_hypotheses >= 2, "Must evolve at least two hypotheses"
        assert self.state_manager.is_started, "NovaScientist system must be started first"
        evolution_candidate_uids = (
            self.state_manager.get_tournament_hypotheses_for_evolution()
        )
        if len(evolution_candidate_uids) < n_hypotheses:
            logging.warning(
                f"Only {len(evolution_candidate_uids)} hypotheses are qualified for evolution. "
                f"Evolving {len(evolution_candidate_uids)} hypotheses."
            )
            n_hypotheses = len(evolution_candidate_uids)

        # The first uids are the top ranked hypotheses
        top_ranked_uids = evolution_candidate_uids[: (n_hypotheses // 2)]
        # The rest are randomly selected
        random_uids = np.random.choice(
            evolution_candidate_uids[(n_hypotheses // 2) :],
            size=n_hypotheses // 2,
            replace=False,
        ).tolist()

        # Evolve the top ranked and random hypotheses based on feedback
        tasks = []
        for uid in top_ranked_uids + random_uids:
            initial_evolution_state = self.state_manager.next_evolution_state(
                mode="evolve_from_feedback", uid_to_evolve=uid
            )
            llm_name = random.choice(self.list_evolution_llm_names())
            evolution_agent = build_evolution_agent(
                mode="evolve_from_feedback",
                llm=self.config.evolution_agent_llms[llm_name],
            )
            tasks.append(evolution_agent.ainvoke(initial_evolution_state))
            
        feedback_results = await asyncio.gather(*tasks)
        for final_evolution_state in feedback_results:
            self.state_manager.add_evolved_hypothesis(
                final_evolution_state["evolved_hypothesis"]
            )
            self.state_manager.advance_hypothesis(kind="evolved")

        # Run one round instance of evolving the top ranked hypotheses
        # into something new
        out_of_box_initial_state = self.state_manager.next_evolution_state(
            mode="out_of_the_box",
            top_k=n_hypotheses // 2,
        )
        llm_name = random.choice(self.list_evolution_llm_names())
        evolution_agent = build_evolution_agent(
            mode="out_of_the_box", llm=self.config.evolution_agent_llms[llm_name]
        )
        out_of_box_state = await evolution_agent.ainvoke(out_of_box_initial_state)
        self.state_manager.add_evolved_hypothesis(
            out_of_box_state["evolved_hypothesis"]
        )

        # Move the evolved hypotheses to the reflection queue
        self.state_manager.advance_hypothesis(kind="evolved")

        # TODO: Do we have to worry about reflecting on hypotheses that are
        # already in the reflection queue but weren't advanced yet?
        # Do we always want to run reflection immediately after a hypothesis
        # is generated?
        await self.process_reflection_queue()

        # Move the reviewed hypothesis to the EloTournament.
        self.state_manager.update_proximity_graph_edges()

    async def expand_literature_review(self) -> None:
        """
        Expands the literature review by adding more subtopics.
        """
        initial_lit_review_state = self.state_manager.next_literature_review_state(
            # TODO: Make this configurable
            max_subtopics=5
        )
        literature_review_agent = build_literature_review_agent(
            self.config.literature_review_agent_llm
        )
        final_lit_review_state = await literature_review_agent.ainvoke(
            initial_lit_review_state
        )
        self.state_manager.update_literature_review(final_lit_review_state)

    async def run_tournament(self, k_bracket: int = 8) -> None:
        k_bracket = min(
            k_bracket,
            2 ** math.floor(math.log2(self.state_manager.num_tournament_hypotheses)),
        )
        self.state_manager.run_tournament(
            llm=self.config.meta_review_agent_llm, k_bracket=k_bracket
        )

    async def run_meta_review(self, k_bracket: int = 8) -> None:
        initial_meta_review_state = self.state_manager.next_meta_review_state(
            top_k=k_bracket
        )
        meta_review_agent = build_meta_review_agent(self.config.meta_review_agent_llm)
        final_meta_review_state = await meta_review_agent.ainvoke(initial_meta_review_state)
        self.state_manager.update_meta_review(final_meta_review_state)

    async def finish(self) -> None:
        initial_final_report_state = self.state_manager.next_final_report_state(top_k=3)
        final_report_agent = build_final_report_agent(
            self.config.final_report_agent_llm
        )
        final_report_state = await final_report_agent.ainvoke(initial_final_report_state)
        self.state_manager.update_final_report(final_report_state)

    @classmethod
    def available_actions(self) -> list[str]:
        """
        List the available actions for the NovaScientist system.
        """
        return [
            "generate_new_hypotheses",
            "evolve_hypotheses",
            "expand_literature_review",
            "run_tournament",
            "run_meta_review",
            "finish",
        ]

    async def run(self) -> tuple[str, str]:
        """
        Runs the novascientist system until it is finished.
        """
        # Start off with 4 hypotheses
        if not self.state_manager.is_started:
            _ = await self.start(n_hypotheses=4)

        supervisor_agent = build_supervisor_agent(self.config.supervisor_agent_llm)

        current_action = None
        while not self.state_manager.is_finished:
            initial_supervisor_state = self.state_manager.next_supervisor_state()
            final_supervisor_state = await supervisor_agent.ainvoke(initial_supervisor_state)
            current_action = final_supervisor_state["action"]
            assert (
                current_action in self.available_actions()
            ), f"Invalid action: {current_action}. Available actions: {self.available_actions()}"
            self.state_manager.update_supervisor_decision(final_supervisor_state)
            self.state_manager.add_action(current_action)
            _ = await getattr(self, current_action)()

        return self.state_manager.final_report, self.state_manager.meta_reviews[-1]
