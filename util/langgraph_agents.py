"""
LangGraph-based Multi-Agent Discussion System for Paper Review

This module implements a sophisticated multi-agent system where 3 reviewer agents
dynamically generated based on paper content discuss among themselves to reach consensus.
"""

import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass
import operator
import json
import random

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

def generate_reviewer_profiles(paper_text: str, model_name: str = "llama3.2") -> List[ReviewerProfile]:
    """Dynamically generate reviewer profiles based on paper content.
    
    Args:
        paper_text: The text content of the paper to analyze
        model_name: Name of the Ollama model to use for generation
        
    Returns:
        List of 3 dynamically generated reviewer profiles
    """
    logger.info("Generating reviewer profiles based on paper content...")
    
    try:
        # Initialize LLM for profile generation
        llm = Ollama(model=model_name, temperature=0.8)
        
        # Analyze paper content to determine research areas and expertise needed
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert in academic peer review who analyzes papers to determine what types of reviewers would be most appropriate.
            
            Based on the paper content, identify:
            1. Primary research domain (e.g., Machine Learning, Computer Vision, NLP, etc.)
            2. Secondary research areas that are relevant
            3. Methodological approaches used
            4. Application domains
            5. Technical complexity level
            
            Respond with a JSON object containing these analysis results.
            """),
            ("human", f"""
            Analyze this paper excerpt to determine appropriate reviewer expertise areas:
            
            {paper_text[:2000]}{'...' if len(paper_text) > 2000 else ''}
            
            Provide analysis in JSON format with fields: primary_domain, secondary_areas, methods, applications, complexity_level
            """)
        ])
        
        chain = analysis_prompt | llm
        analysis_result = chain.invoke({})
        
        # Parse the analysis (with fallback if JSON parsing fails)
        try:
            import re
            json_match = re.search(r'\{.*\}', analysis_result, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found")
        except (json.JSONDecodeError, ValueError):
            # Fallback analysis if JSON parsing fails
            analysis = {
                "primary_domain": "Computer Science",
                "secondary_areas": ["Machine Learning", "Data Science"],
                "methods": ["Statistical Analysis", "Experimental Design"],
                "applications": ["General Applications"],
                "complexity_level": "Medium"
            }
            logger.warning("Failed to parse analysis JSON, using fallback")
        
        # Generate three diverse reviewer profiles
        profile_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are creating reviewer profiles for a peer review panel. Create diverse, realistic academic reviewer profiles.
            
            Generate exactly 3 different reviewer profiles with these characteristics:
            - Different experience levels (mix of senior/junior)
            - Different expertise areas relevant to the paper
            - Different review styles and personalities
            - Realistic academic names and titles
            
            Make each reviewer unique and complementary to the others.
            Respond with a JSON array of 3 reviewer objects, each with: name, experience_level, expertise_area, review_style, personality
            """),
            ("human", f"""
            Based on this paper analysis, create 3 reviewer profiles:
            
            Primary Domain: {analysis.get('primary_domain', 'Computer Science')}
            Secondary Areas: {analysis.get('secondary_areas', [])}
            Methods: {analysis.get('methods', [])}
            Applications: {analysis.get('applications', [])}
            Complexity: {analysis.get('complexity_level', 'Medium')}
            
            Generate 3 diverse reviewer profiles as JSON array.
            """)
        ])
        
        chain = profile_prompt | llm
        profiles_result = chain.invoke({})
        
        # Parse generated profiles
        try:
            import re
            json_match = re.search(r'\[.*\]', profiles_result, re.DOTALL)
            if json_match:
                profiles_data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON array found")
                
            if len(profiles_data) != 3:
                raise ValueError(f"Expected 3 profiles, got {len(profiles_data)}")
                
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse generated profiles: {e}, using fallback")
            # Fallback to domain-appropriate default profiles
            primary_domain = analysis.get('primary_domain', 'Computer Science')
            profiles_data = generate_fallback_profiles(primary_domain)
        
        # Convert to ReviewerProfile objects
        reviewer_profiles = []
        for i, profile_data in enumerate(profiles_data):
            try:
                profile = ReviewerProfile(
                    name=profile_data.get('name', f'Reviewer {i+1}'),
                    experience_level=profile_data.get('experience_level', 'Associate Professor'),
                    expertise_area=profile_data.get('expertise_area', analysis.get('primary_domain', 'Computer Science')),
                    review_style=profile_data.get('review_style', 'thorough and balanced'),
                    personality=profile_data.get('personality', 'analytical and constructive')
                )
                reviewer_profiles.append(profile)
            except Exception as e:
                logger.error(f"Error creating profile {i}: {e}")
                # Create a basic profile as fallback
                profile = ReviewerProfile(
                    name=f"Dr. Reviewer {i+1}",
                    experience_level="Associate Professor",
                    expertise_area=analysis.get('primary_domain', 'Computer Science'),
                    review_style="balanced and thorough",
                    personality="analytical and fair"
                )
                reviewer_profiles.append(profile)
        
        logger.info(f"Generated {len(reviewer_profiles)} reviewer profiles")
        for profile in reviewer_profiles:
            logger.info(f"  - {profile.name}: {profile.expertise_area} ({profile.experience_level})")
        
        return reviewer_profiles
        
    except Exception as e:
        logger.error(f"Error in profile generation: {e}")
        # Use fallback profiles if generation fails completely
        return generate_fallback_profiles("Computer Science")

def generate_fallback_profiles(domain: str) -> List[Dict[str, str]]:
    """Generate fallback reviewer profiles for a given domain.
    
    Args:
        domain: Primary research domain
        
    Returns:
        List of 3 fallback profile dictionaries
    """
    # Define some variety in names and characteristics
    first_names = ["Dr. Alex", "Prof. Sam", "Dr. Jordan", "Prof. Taylor", "Dr. Casey", "Prof. Morgan"]
    last_names = ["Chen", "Rivera", "Patel", "Johnson", "Williams", "Garcia", "Martinez", "Anderson"]
    
    experience_levels = ["Senior Researcher", "Associate Professor", "Principal Scientist", "Full Professor", "Research Scientist"]
    review_styles = ["thorough and methodical", "balanced and constructive", "critical but fair", "detailed and analytical", "practical and focused"]
    personalities = [
        "analytical, asks probing questions",
        "collaborative, seeks consensus", 
        "direct, efficiency-focused",
        "thorough, detail-oriented",
        "innovative, future-thinking"
    ]
    
    # Create domain-specific expertise areas
    if "Machine Learning" in domain or "AI" in domain:
        expertise_areas = ["Machine Learning", "Deep Learning", "Neural Networks"]
    elif "Computer Vision" in domain or "Vision" in domain:
        expertise_areas = ["Computer Vision", "Image Processing", "Visual Recognition"]
    elif "Natural Language" in domain or "NLP" in domain:
        expertise_areas = ["Natural Language Processing", "Computational Linguistics", "Text Mining"]
    else:
        expertise_areas = [domain, f"{domain} Applications", f"Computational {domain}"]
    
    profiles = []
    used_names = set()
    
    for i in range(3):
        # Generate unique names
        max_attempts = 10
        for attempt in range(max_attempts):
            name = f"{random.choice(first_names)} {random.choice(last_names)}"
            if name not in used_names:
                used_names.add(name)
                break
        else:
            # Fallback if all combinations are somehow used
            name = f"{first_names[i % len(first_names)]} {last_names[i % len(last_names)]}"
        
        profiles.append({
            "name": name,
            "experience_level": experience_levels[i % len(experience_levels)],
            "expertise_area": expertise_areas[i % len(expertise_areas)],
            "review_style": review_styles[i % len(review_styles)],
            "personality": personalities[i % len(personalities)]
        })
    
    return profiles

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
    reviewer_profiles: List[ReviewerProfile]

class LangGraphReviewSystem:
    """LangGraph-based multi-agent review system with dynamic reviewer profiles"""
    
    def __init__(self, model_name: str = "llama3.2", reviewer_profiles: Optional[List[ReviewerProfile]] = None):
        """Initialize the review system
        
        Args:
            model_name: Name of the Ollama model to use
            reviewer_profiles: Pre-generated reviewer profiles (if None, will be generated dynamically)
        """
        self.model_name = model_name
        self.llm = self._create_llm()
        self.reviewer_profiles = reviewer_profiles  # Will be set dynamically per review
        self.memory = MemorySaver()
        
    def _create_llm(self) -> BaseLanguageModel:
        """Create the language model"""
        try:
            return Ollama(model=self.model_name, temperature=0.7)
        except Exception as e:
            logger.error(f"Failed to create Ollama model {self.model_name}: {e}")
            raise
    
    def _create_graph(self, reviewer_profiles: List[ReviewerProfile]) -> StateGraph:
        """Create the LangGraph workflow with dynamic reviewer profiles
        
        Args:
            reviewer_profiles: List of reviewer profiles to use for this review
        """
        graph = StateGraph(ReviewState)
        
        # Add nodes for each reviewer
        for profile in reviewer_profiles:
            graph.add_node(profile.name, self._create_reviewer_node(profile))
        
        # Add facilitator node for managing discussion
        graph.add_node("facilitator", self._facilitator_node)
        
        # Add consensus checker node
        graph.add_node("consensus_checker", self._consensus_checker_node)
        
        # Add final decision node
        graph.add_node("final_decision", self._final_decision_node)
        
        # Define the flow
        graph.set_entry_point("facilitator")
        
        # From facilitator, go to first reviewer (dynamically determined)
        graph.add_conditional_edges(
            "facilitator",
            self._route_from_facilitator,
            {profile.name: profile.name for profile in reviewer_profiles}
        )
        
        # Chain reviewers in discussion rounds
        for i in range(len(reviewer_profiles)):
            current_reviewer = reviewer_profiles[i].name
            next_reviewer = reviewer_profiles[(i + 1) % len(reviewer_profiles)].name
            
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
            self._consensus_reached_with_routing,
            {
                "consensus": "final_decision",
                "continue_discussion": "facilitator"  # Will route to first reviewer through facilitator
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
    
    def _route_from_facilitator(self, state: ReviewState) -> str:
        """Route from facilitator to the first reviewer"""
        reviewer_profiles = state.get('reviewer_profiles', [])
        if reviewer_profiles:
            return reviewer_profiles[0].name
        else:
            logger.error("No reviewer profiles available for routing")
            return "final_decision"  # Fallback
    
    def _facilitator_node(self, state: ReviewState) -> ReviewState:
        """Facilitator node to manage the discussion"""
        if state['round_number'] == 0:
            intro_message = f"Starting review discussion for section: {state['section_name']}"
            state['messages'].append(f"Facilitator: {intro_message}")
            logger.info(f"Starting review discussion for {state['section_name']}")
        else:
            # Continuing discussion after consensus check
            continuation_message = f"Continuing discussion - Round {state['round_number'] + 1}"
            state['messages'].append(f"Facilitator: {continuation_message}")
            logger.info(f"Continuing discussion for round {state['round_number'] + 1}")
        
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
        
        # Get reviewer profiles from state
        reviewer_profiles = state.get('reviewer_profiles', [])
        if not reviewer_profiles:
            return "check_consensus"
        
        # Check if current speaker is the last reviewer in the round
        current_speaker_idx = None
        for i, profile in enumerate(reviewer_profiles):
            if profile.name == state['current_speaker']:
                current_speaker_idx = i
                break
        
        if current_speaker_idx == len(reviewer_profiles) - 1:
            return "check_consensus"
        else:
            return "continue"
    
    def _consensus_reached(self, state: ReviewState) -> str:
        """Check if consensus was reached"""
        return "consensus" if state['consensus_reached'] else "continue"
    
    def _consensus_reached_with_routing(self, state: ReviewState) -> str:
        """Check if consensus was reached and determine routing"""
        if state.get('consensus_reached', False):
            return "consensus"
        else:
            return "continue_discussion"
    
    def review_section(self, section_text: str, section_name: str) -> Dict[str, Any]:
        """Review a paper section using dynamically generated multi-agent discussion
        
        Args:
            section_text: The text content to review
            section_name: Name of the section being reviewed
            
        Returns:
            Dictionary containing the review results
        """
        
        try:
            # Generate reviewer profiles based on paper content
            logger.info(f"Generating reviewers for section: {section_name}")
            reviewer_profiles = generate_reviewer_profiles(section_text, self.model_name)
            
            # Create the graph with the generated profiles
            graph = self._create_graph(reviewer_profiles)
            
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
                discussion_summary="",
                reviewer_profiles=reviewer_profiles
            )
            
            # Run the graph
            config = {"configurable": {"thread_id": f"review_{section_name}"}}
            final_state = graph.invoke(initial_state, config)
            
            # Format results
            result = {
                "section_name": section_name,
                "final_decision": final_state.get('final_decision', 'No decision reached'),
                "discussion_summary": final_state.get('discussion_summary', ''),
                "individual_reviews": final_state.get('individual_reviews', {}),
                "full_discussion": final_state.get('messages', []),
                "rounds_completed": final_state.get('round_number', 0),
                "consensus_reached": final_state.get('consensus_reached', False),
                "generated_reviewers": [
                    {
                        "name": profile.name,
                        "expertise": profile.expertise_area,
                        "experience": profile.experience_level,
                        "style": profile.review_style
                    }
                    for profile in reviewer_profiles
                ]
            }
            
            logger.info(f"Completed review for {section_name} with generated reviewers")
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
                "consensus_reached": False,
                "generated_reviewers": []
            }

def create_review_system(model_name: str = "llama3.2") -> LangGraphReviewSystem:
    """Create a new review system instance
    
    Args:
        model_name: Name of the Ollama model to use
        
    Returns:
        Configured LangGraphReviewSystem instance
    """
    return LangGraphReviewSystem(model_name)