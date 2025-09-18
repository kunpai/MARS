import logging
from typing import List, Dict, Any, Optional, Tuple
import ollama
from ollama import chat, ChatResponse
import requests
import re
from bs4 import BeautifulSoup

# Configure logging
logger = logging.getLogger(__name__)

def isModelLoaded(model: str) -> bool:
    """Check if a model is loaded in Ollama.
    
    Args:
        model: Model name to check
        
    Returns:
        True if model is loaded, False otherwise
    """
    try:
        loaded_models = [model.model for model in ollama.list().models]
        return model in loaded_models or f"{model}:latest" in loaded_models
    except Exception as e:
        logger.error(f"Error checking if model {model} is loaded: {e}")
        return False

def consultWiki(question: str) -> str:
    """Search Wikipedia for information and return summary.
    
    Args:
        question: Search query for Wikipedia
        
    Returns:
        Summary of Wikipedia article or error message
    """
    logger.info(f"Searching Wikipedia for: {question}")
    
    search_url = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": question,
        "srlimit": 1,
    }

    try:
        response = requests.get(search_url, params=search_params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        search_results = data.get("query", {}).get("search", [])

        if search_results:
            top_result = search_results[0]["title"]
            page_url = f"https://en.wikipedia.org/wiki/{top_result.replace(' ', '_')}"
            logger.info(f"Fetching content from: {page_url}")

            # Fetch the full page HTML
            html_url = f"https://en.wikipedia.org/api/rest_v1/page/html/{top_result.replace(' ', '_')}"
            html_response = requests.get(html_url, timeout=10)
            html_response.raise_for_status()

            soup = BeautifulSoup(html_response.text, "html.parser")

            # Extract all paragraphs from the page
            paragraphs = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]
            if paragraphs:
                full_text = " ".join(paragraphs)
                # Summarize (basic extractive approach)
                summary = " ".join(full_text.split(". ")[:5])  # First 5 sentences
                return f"**{top_result}**\n{summary}...\n[Read more]({page_url})"
    
    except requests.RequestException as e:
        logger.error(f"Error fetching from Wikipedia: {e}")
        return f"Error fetching from Wikipedia: {e}"
    except Exception as e:
        logger.error(f"Unexpected error in Wikipedia search: {e}")
        return f"Unexpected error in Wikipedia search: {e}"
    
    return "No results found on Wikipedia. Try using simpler keywords."
    
def consultAgent(agent: str, question: str) -> Optional[str]:
    """Consult an AI agent with a question.
    
    Args:
        agent: Name of the agent/model to consult
        question: Question to ask the agent
        
    Returns:
        Response from the agent or None if error
    """
    if not isModelLoaded(agent):
        logger.error(f"Model {agent} not found")
        return None
        
    try:
        response: ChatResponse = chat(model=agent, messages=[
            {
                'role': 'user',
                'content': question,
            },
        ])
        return response.message.content
    except Exception as e:
        logger.error(f"Error consulting agent {agent}: {e}")
        return None

def consultDeskReviewer(abstract: str) -> Tuple[bool, str]:
    """Consult desk reviewer for paper acceptance decision.
    
    Args:
        abstract: Paper abstract to review
        
    Returns:
        Tuple of (accept_decision, review_text)
    """
    desk_review = consultAgent('deskreviewer', abstract)
    if desk_review is None:
        return False, "Error: Could not get desk review"
        
    logger.info(f"Desk review: {desk_review}")
    return 'accept' in desk_review.lower(), desk_review

def consultReviewer1(abstract: str) -> str:
    """Consult reviewer 1 for paper review.
    
    Args:
        abstract: Paper abstract to review
        
    Returns:
        Review decision (first word of response)
    """
    review = consultAgent('reviewer1', abstract)
    if review is None:
        return "Error"
    logger.info(f"Reviewer 1: {review}")
    return review.split(' ')[0]

def consultReviewer2(abstract: str) -> str:
    """Consult reviewer 2 for paper review.
    
    Args:
        abstract: Paper abstract to review
        
    Returns:
        Review decision (first word of response)
    """
    review = consultAgent('reviewer2', abstract)
    if review is None:
        return "Error"
    logger.info(f"Reviewer 2: {review}")
    return review.split(' ')[0]

def consultReviewer3(abstract: str) -> str:
    """Consult reviewer 3 for paper review.
    
    Args:
        abstract: Paper abstract to review
        
    Returns:
        Review decision (first word of response)
    """
    review = consultAgent('reviewer3', abstract)
    if review is None:
        return "Error"
    logger.info(f"Reviewer 3: {review}")
    return review.split(' ')[0]

def consultPaperSpecificModels(model, question):
    return consultAgent(model, question)

def consultQuestioner(text):
    return consultAgent('questioner', text)

def consultGrammar(text):
    return consultAgent('grammar', text)

def consultTest(text):
    return consultAgent('test', text)

def consultNovelty(text):
    return consultAgent('novelty', text)

def consultFactChecker(text):
    tool_config = {
        "name": "consultWiki",
        "type": "function",
        "function": {
            "name": "consultWiki",
            "description": "Consult Wikipedia to check facts",
            "parameters": {
                "type": "object",
                "required": ["question"],
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask Wikipedia"
                    }
                }
            }
        }
    }
    
    retries = 3  # Set a max retry limit
    query = text

    response = chat(model='factchecker', messages=[{'role': 'user', 'content': "Do you need more facts? Only say yes or no. \n " + query}])
    if 'yes' in response.message.content.lower():
        
        for attempt in range(retries):
            response = chat(model='factchecker', messages=[{'role': 'user', 'content': query}], tools=[tool_config])
            
            print("Attempt number", attempt + 1)

            if response.message.tool_calls:
                for tool in response.message.tool_calls:
                    if function_to_call := available_functions.get(tool.function.name):
                        try:
                            print("Asking " + tool.function.name + ": " + tool.function.arguments['question'])
                        except:
                            pass
                        output = function_to_call(**tool.function.arguments)
                        
                        if output and output != "No results found on Wikipedia. Try using simpler keywords.":
                            # new_query = "Question: \n" + query + " " + "Answer: \n" + output
                            # print("new_query", new_query)
                            # consultFactChecker(new_query)
                            return output
                        
                        print("Refining query...")
                        query = " ".join(re.findall(r'\b[A-Za-z0-9-]+\b', text)[:10]) # more specific query

        print("Could not retrieve relevant information from Wikipedia after multiple attempts.")
        return None
    else:
        response = chat(model='factchecker', messages=[{'role': 'user', 'content': "Do you accept the claims? Say 'Accept' if yes and 'Reject' if no. \n " + query}])
        return response.message.content

# Initialize available models safely
try:
    available_models = [model.model for model in ollama.list().models]
    logger.info(f"Loaded {len(available_models)} available models")
except Exception as e:
    logger.warning(f"Could not load available models (Ollama may not be running): {e}")
    available_models = []

available_functions = {
    'consultWiki': consultWiki,
    'consultDeskReviewer': consultDeskReviewer,
    'consultReviewer1': consultReviewer1,
    'consultReviewer2': consultReviewer2,
    'consultReviewer3': consultReviewer3,
    'consultQuestioner': consultQuestioner,
    'consultGrammar': consultGrammar,
    'consultTest': consultTest,
    'consultNovelty': consultNovelty,
    'consultFactChecker': consultFactChecker,
}

