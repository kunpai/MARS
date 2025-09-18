"""
LangGraph-based Multi-Agent Discussion System for Paper Review

This module implements a sophisticated multi-agent system where 3 reviewer agents
discuss among themselves to reach a consensus on paper sections.
"""

import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass
import operator
import json

try:
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_core.language_models.base import BaseLanguageModel
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.llms import Ollama
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolExecutor
    from langgraph.checkpoint.memory import MemorySaver
except ImportError as e:
    logging.error(f"LangChain/LangGraph not installed: {e}")
    logging.error("Please install with: pip install langchain langchain-community langgraph")
    raise

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ReviewerProfile:
    """Profile for a reviewer agent"""
    name: str
    experience_level: str
    expertise_area: str
    review_style: str
    personality: str

# Define reviewer profiles
REVIEWER_PROFILES = [
    ReviewerProfile(
        name="Dr. Sarah Chen",
        experience_level="Senior Researcher",
        expertise_area="Machine Learning and AI",
        review_style="thorough and methodical",
        personality="analytical, asks probing questions, focuses on technical rigor"
    ),
    ReviewerProfile(
        name="Prof. Marcus Rivera",
        experience_level="Associate Professor", 
        expertise_area="Natural Language Processing",
        review_style="balanced and constructive",
        personality="collaborative, seeks consensus, good at synthesis"
    ),
    ReviewerProfile(
        name="Dr. Aisha Patel",
        experience_level="Principal Scientist",
        expertise_area="Computer Vision and Deep Learning",
        review_style="critical but fair",
        personality="direct, efficiency-focused, highlights practical implications"
    )
]

class ReviewState(TypedDict):
    """State of the review discussion"""
    section_text: str
    section_name: str
    messages: Annotated[List[str], operator.add]
    current_speaker: str
    round_number: int
    consensus_reached: bool
    final_decision: Optional[str]
    individual_reviews: Dict[str, str]
    discussion_summary: str

class LangGraphReviewSystem:
    """LangGraph-based multi-agent review system"""
    
    def __init__(self, model_name: str = "llama3.2"):
        """Initialize the review system
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.llm = self._create_llm()
        self.graph = self._create_graph()
        self.memory = MemorySaver()
        
    def _create_llm(self) -> BaseLanguageModel:
        """Create the language model"""
        try:
            return Ollama(model=self.model_name, temperature=0.7)
        except Exception as e:
            logger.error(f"Failed to create Ollama model {self.model_name}: {e}")
            raise
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow"""
        graph = StateGraph(ReviewState)
        
        # Add nodes for each reviewer
        for profile in REVIEWER_PROFILES:
            graph.add_node(profile.name, self._create_reviewer_node(profile))
        
        # Add facilitator node for managing discussion
        graph.add_node("facilitator", self._facilitator_node)
        
        # Add consensus checker node
        graph.add_node("consensus_checker", self._consensus_checker_node)
        
        # Add final decision node
        graph.add_node("final_decision", self._final_decision_node)
        
        # Define the flow
        graph.set_entry_point("facilitator")
        
        # From facilitator, go to first reviewer
        graph.add_edge("facilitator", REVIEWER_PROFILES[0].name)
        
        # Chain reviewers in discussion rounds
        for i in range(len(REVIEWER_PROFILES)):
            current_reviewer = REVIEWER_PROFILES[i].name
            next_reviewer = REVIEWER_PROFILES[(i + 1) % len(REVIEWER_PROFILES)].name
            
            # Each reviewer can go to next reviewer or consensus checker
            graph.add_conditional_edges(
                current_reviewer,
                self._should_continue_discussion,
                {
                    "continue": next_reviewer,
                    "check_consensus": "consensus_checker"
                }
            )
        
        # From consensus checker, either continue discussion or make final decision
        graph.add_conditional_edges(
            "consensus_checker",
            self._consensus_reached,
            {
                "consensus": "final_decision",
                "continue": REVIEWER_PROFILES[0].name
            }
        )
        
        # Final decision ends the process
        graph.add_edge("final_decision", END)
        
        return graph.compile(checkpointer=self.memory)
    
    def _create_reviewer_node(self, profile: ReviewerProfile):
        """Create a reviewer node function"""
        def reviewer_node(state: ReviewState) -> ReviewState:
            """Process review for a specific reviewer"""
            
            # Create system prompt for this reviewer
            system_prompt = f"""
            You are {profile.name}, a {profile.experience_level} with expertise in {profile.expertise_area}.
            Your review style is {profile.review_style} and your personality is {profile.personality}.
            
            You are participating in a collaborative review discussion with other experts.
            Your goal is to provide thoughtful feedback and work toward a consensus decision.
            
            Current discussion context:
            - Section being reviewed: {state['section_name']}
            - Discussion round: {state['round_number']}
            - Previous messages in this discussion: {len(state['messages'])}
            
            Guidelines:
            1. Be true to your expertise area and personality
            2. Build on previous comments constructively
            3. Ask clarifying questions when needed
            4. Propose specific improvements
            5. State your current leaning (Accept/Reject/Revise) with reasoning
            6. Keep responses focused and under 200 words
            """
            
            # Get recent discussion context
            recent_messages = state['messages'][-6:] if len(state['messages']) > 6 else state['messages']
            context = "\n".join(recent_messages) if recent_messages else "Starting discussion..."
            
            user_prompt = f"""
            Paper section to review:
            ---
            {state['section_text'][:1000]}{'...' if len(state['section_text']) > 1000 else ''}
            ---
            
            Recent discussion context:
            {context}
            
            Please provide your review input for this round of discussion.
            """
            
            try:
                # Generate response using the LLM
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", user_prompt)
                ])
                
                chain = prompt | self.llm
                response = chain.invoke({})
                
                # Update state
                new_message = f"{profile.name}: {response}"
                state['messages'].append(new_message)
                state['current_speaker'] = profile.name
                state['individual_reviews'][profile.name] = response
                
                logger.info(f"{profile.name} contributed to discussion")
                return state
                
            except Exception as e:
                logger.error(f"Error in reviewer node for {profile.name}: {e}")
                error_message = f"{profile.name}: Error generating review - {str(e)}"
                state['messages'].append(error_message)
                return state
        
        return reviewer_node
    
    def _facilitator_node(self, state: ReviewState) -> ReviewState:
        """Facilitator node to manage the discussion"""
        if state['round_number'] == 0:
            intro_message = f"Starting review discussion for section: {state['section_name']}"
            state['messages'].append(f"Facilitator: {intro_message}")
            logger.info(f"Starting review discussion for {state['section_name']}")
        
        state['round_number'] += 1
        return state
    
    def _consensus_checker_node(self, state: ReviewState) -> ReviewState:
        """Check if consensus has been reached"""
        
        # Analyze the discussion to determine consensus
        system_prompt = """
        You are a neutral facilitator analyzing a review discussion.
        Determine if the reviewers have reached sufficient consensus to make a decision.
        
        Look for:
        1. Agreement on major points
        2. Convergence toward Accept/Reject/Revise
        3. Resolution of significant concerns
        4. Sufficient discussion depth
        
        Respond with either:
        - "CONSENSUS_REACHED" if ready for final decision
        - "CONTINUE_DISCUSSION" if more discussion needed
        
        Provide brief reasoning for your decision.
        """
        
        recent_discussion = "\n".join(state['messages'][-9:])  # Last 3 rounds
        
        user_prompt = f"""
        Review discussion so far:
        {recent_discussion}
        
        Has consensus been reached? Should we make a final decision or continue discussion?
        """
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", user_prompt)
            ])
            
            chain = prompt | self.llm
            response = chain.invoke({})
            
            # Determine consensus
            if "CONSENSUS_REACHED" in response.upper():
                state['consensus_reached'] = True
                logger.info("Consensus reached, moving to final decision")
            else:
                state['consensus_reached'] = False
                logger.info("Continuing discussion - no consensus yet")
            
            state['messages'].append(f"Facilitator: {response}")
            return state
            
        except Exception as e:
            logger.error(f"Error in consensus checker: {e}")
            # Default to continue if error
            state['consensus_reached'] = False
            return state
    
    def _final_decision_node(self, state: ReviewState) -> ReviewState:
        """Make the final decision based on discussion"""
        
        system_prompt = """
        You are synthesizing a collaborative review discussion into a final decision.
        
        Based on the entire discussion, provide:
        1. Final decision: Accept, Reject, or Major Revisions
        2. Key reasoning (2-3 main points)
        3. Specific suggestions for improvement (if applicable)
        4. Summary of reviewer consensus
        
        Format your response as:
        DECISION: [Accept/Reject/Major Revisions]
        REASONING: [key points]
        SUGGESTIONS: [specific improvements]
        CONSENSUS: [level of agreement]
        """
        
        full_discussion = "\n".join(state['messages'])
        
        user_prompt = f"""
        Complete review discussion:
        {full_discussion}
        
        Please synthesize this into a final decision and summary.
        """
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", user_prompt)
            ])
            
            chain = prompt | self.llm
            response = chain.invoke({})
            
            state['final_decision'] = response
            state['discussion_summary'] = response
            
            logger.info("Final decision reached")
            return state
            
        except Exception as e:
            logger.error(f"Error in final decision: {e}")
            state['final_decision'] = f"Error generating final decision: {e}"
            return state
    
    def _should_continue_discussion(self, state: ReviewState) -> str:
        """Decide whether to continue discussion or check consensus"""
        
        # Simple logic: after each reviewer in a round, check consensus
        # Max 3 rounds to prevent infinite loops
        if state['round_number'] >= 3:
            return "check_consensus"
        
        # Check if current speaker is the last reviewer in the round
        current_speaker_idx = None
        for i, profile in enumerate(REVIEWER_PROFILES):
            if profile.name == state['current_speaker']:
                current_speaker_idx = i
                break
        
        if current_speaker_idx == len(REVIEWER_PROFILES) - 1:
            return "check_consensus"
        else:
            return "continue"
    
    def _consensus_reached(self, state: ReviewState) -> str:
        """Check if consensus was reached"""
        return "consensus" if state['consensus_reached'] else "continue"
    
    def review_section(self, section_text: str, section_name: str) -> Dict[str, Any]:
        """Review a paper section using multi-agent discussion
        
        Args:
            section_text: The text content to review
            section_name: Name of the section being reviewed
            
        Returns:
            Dictionary containing the review results
        """
        
        # Initialize state
        initial_state = ReviewState(
            section_text=section_text,
            section_name=section_name,
            messages=[],
            current_speaker="",
            round_number=0,
            consensus_reached=False,
            final_decision=None,
            individual_reviews={},
            discussion_summary=""
        )
        
        try:
            # Run the graph
            config = {"configurable": {"thread_id": f"review_{section_name}"}}
            final_state = self.graph.invoke(initial_state, config)
            
            # Format results
            result = {
                "section_name": section_name,
                "final_decision": final_state.get('final_decision', 'No decision reached'),
                "discussion_summary": final_state.get('discussion_summary', ''),
                "individual_reviews": final_state.get('individual_reviews', {}),
                "full_discussion": final_state.get('messages', []),
                "rounds_completed": final_state.get('round_number', 0),
                "consensus_reached": final_state.get('consensus_reached', False)
            }
            
            logger.info(f"Completed review for {section_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error in review process: {e}")
            return {
                "section_name": section_name,
                "final_decision": f"Error: {str(e)}",
                "discussion_summary": "Review process failed",
                "individual_reviews": {},
                "full_discussion": [],
                "rounds_completed": 0,
                "consensus_reached": False
            }

def create_review_system(model_name: str = "llama3.2") -> LangGraphReviewSystem:
    """Create a new review system instance
    
    Args:
        model_name: Name of the Ollama model to use
        
    Returns:
        Configured LangGraphReviewSystem instance
    """
    return LangGraphReviewSystem(model_name)