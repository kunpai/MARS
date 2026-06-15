import litellm
import requests
import re
import json
from bs4 import BeautifulSoup
from util.rag import search_relevant_context

# We'll expect base_models to be passed in or accessible, but for this file's functions
# we can just use litellm.completion with the appropriate system prompt if we have it,
# or for simplicity, let's keep a global dict of system prompts that we can update.
system_prompts = {}

def set_system_prompts(prompts):
    global system_prompts
    system_prompts = prompts

def consultWiki(question):
    print(f"Searching Wikipedia for: {question}")
    
    search_url = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": question,
        "srlimit": 1,
    }

    try:
        response = requests.get(search_url, params=search_params)
        if response.status_code == 200:
            data = response.json()
            search_results = data.get("query", {}).get("search", [])

            if search_results:
                top_result = search_results[0]["title"]
                page_url = f"https://en.wikipedia.org/wiki/{top_result.replace(' ', '_')}"
                print(f"Fetching full content from: {page_url}")

                html_url = f"https://en.wikipedia.org/api/rest_v1/page/html/{top_result.replace(' ', '_')}"
                html_response = requests.get(html_url)

                if html_response.status_code == 200:
                    soup = BeautifulSoup(html_response.text, "html.parser")
                    paragraphs = [p.get_text() for p in soup.find_all("p") if p.get_text()]
                    full_text = " ".join(paragraphs)

                    # Instead of basic extractive, let's use RAG to find the most relevant chunk
                    summary = search_relevant_context(question, full_text, top_k=1)

                    return f"**{top_result}**\n{summary}\n[Read more]({page_url})"
    except Exception as e:
        print(f"Wikipedia search error: {e}")
    
    return "No results found on Wikipedia. Try using simpler keywords."
    
def consultAgent(agent_role, question, model="nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"):
    system_prompt = system_prompts.get(agent_role, "")
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    try:
        response = litellm.completion(model=model, messages=messages)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error consulting agent {agent_role}: {e}")
        return f"Error: {e}"

def consultDeskReviewer(abstract, model="nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"):
    desk_review = consultAgent('deskreviewer', abstract, model=model)
    print(desk_review)
    return 'accept' in desk_review.lower(), desk_review

def consultQuestioner(text, model="nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"):
    return consultAgent('questioner', text, model=model)

def consultGrammar(text, model="nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"):
    return consultAgent('grammar', text, model=model)

def consultTest(text, model="nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"):
    return consultAgent('test', text, model=model)

def consultNovelty(text, full_paper_text="", model="nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"):
    if full_paper_text:
        # Use RAG to get relevant context from the paper for novelty
        context = search_relevant_context(text, full_paper_text, top_k=2)
        query = f"Context: {context}\n\nSection to evaluate: {text}"
    else:
        query = text
    return consultAgent('novelty', query, model=model)

def consultFactChecker(text, full_paper_text="", model="nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"):
    tool_config = {
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
    
    system_prompt = system_prompts.get('factchecker', "You are a fact checker.")

    # Check if more facts are needed
    try:
        need_facts_response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Do you need more facts to evaluate this? Only say yes or no.\n" + text}
            ]
        )
        
        if 'yes' in need_facts_response.choices[0].message.content.lower():
            retries = 3
            query = text
            for attempt in range(retries):
                print("Attempt number", attempt + 1)

                response = litellm.completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ],
                    tools=[tool_config]
                )

                message = response.choices[0].message
                if message.tool_calls:
                    for tool in message.tool_calls:
                        if tool.function.name == 'consultWiki':
                            try:
                                args = json.loads(tool.function.arguments)
                                print("Asking Wiki:", args['question'])
                                output = consultWiki(args['question'])
                                if output and output != "No results found on Wikipedia. Try using simpler keywords.":
                                    return output
                            except Exception as e:
                                print(f"Tool execution error: {e}")

                    print("Refining query...")
                    query = " ".join(re.findall(r'\b[A-Za-z0-9-]+\b', text)[:10])

            print("Could not retrieve relevant information from Wikipedia after multiple attempts.")
            return None
        else:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Do you accept the claims? Say 'Accept' if yes and 'Reject' if no.\n" + text}
                ]
            )
            return response.choices[0].message.content
    except Exception as e:
        print(f"Fact checking error: {e}")
        return None

available_functions = {
    'consultWiki': consultWiki,
    'consultDeskReviewer': consultDeskReviewer,
    'consultQuestioner': consultQuestioner,
    'consultGrammar': consultGrammar,
    'consultTest': consultTest,
    'consultNovelty': consultNovelty,
    'consultFactChecker': consultFactChecker,
}
