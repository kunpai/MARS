import litellm
import requests
import re
import json
from bs4 import BeautifulSoup
from util.rag import search_relevant_context
from pydantic import BaseModel, Field
from typing import List
from util.scholar import search_arxiv_papers

def consultArxiv(query):
    print(f"Searching arXiv for: {query}")
    try:
        import time
        time.sleep(3)  # Wait to respect arXiv rate limits (HTTP 429)
        papers = search_arxiv_papers(query, max_results=3)
        if not papers:
            return "No matching papers found on arXiv."
        results = []
        for paper in papers:
            results.append(f"Title: {paper['title']}\nAuthors: {', '.join(paper['authors'])}\nSummary: {paper['summary']}\nURL: {paper['pdf_url']}")
        return "\n\n".join(results)
    except Exception as e:
        print(f"arXiv search error: {e}. Falling back to web search...")
        web_fallback = consultWeb(query)
        return f"arXiv search failed due to rate limits or API error ({e}). Web search fallback results:\n{web_fallback}"

def consultWeb(query):
    print(f"Searching web (DuckDuckGo) for: {query}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            snippets = []
            for a in soup.find_all("a", class_="result__snippet")[:3]:
                snippets.append(a.get_text().strip())
            if snippets:
                return "\n".join(snippets)
    except Exception as e:
        print(f"Web search error: {e}")
    return "No web search results found."

class DeskReviewResponse(BaseModel):
    accept: bool = Field(..., description="Whether the paper is relevant to the conference topics")
    explanation: str = Field(..., description="Explanation for the desk review decision")

class QuestionerResponse(BaseModel):
    questions: List[str] = Field(..., description="List of open-ended, non-leading questions about the paper section")

class GrammarResponse(BaseModel):
    accept: bool = Field(..., description="Whether the grammar is correct")
    corrections: str = Field(..., description="Specific grammar corrections or notes")

class NoveltyResponse(BaseModel):
    novel: bool = Field(..., description="Whether the paper is novel")
    reasoning: str = Field(..., description="Detailed explanation of novelty evaluation")

class FactCheckResponse(BaseModel):
    accept: bool = Field(..., description="Whether the facts are correct and verified")
    corrections: str = Field(..., description="Specific corrections if inaccuracies were found, or verification notes")

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
    
def parse_json_markdown(text):
    text = text.strip()
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end+1]
    return json.loads(text)

def consultAgent(agent_role, question, model="nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", response_schema_instructions=""):
    system_prompt = system_prompts.get(agent_role, "")
    if response_schema_instructions:
        system_prompt += f"\n\nCRITICAL: You MUST respond ONLY with a JSON object matching this schema or format:\n{response_schema_instructions}\nDo not include any preambles, markdown formatting outside of a JSON code block, or notes outside the JSON itself."
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
    schema_inst = '{"accept": boolean, "explanation": "string"}'
    desk_review_raw = consultAgent('deskreviewer', abstract, model=model, response_schema_instructions=schema_inst)
    print(desk_review_raw)
    try:
        data = parse_json_markdown(desk_review_raw)
        return data.get("accept", False), data
    except Exception:
        return 'accept' in desk_review_raw.lower(), desk_review_raw

def consultQuestioner(text, model="nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"):
    schema_inst = '{"questions": ["question 1", "question 2", ...]}'
    raw = consultAgent('questioner', text, model=model, response_schema_instructions=schema_inst)
    try:
        return parse_json_markdown(raw)
    except Exception:
        return raw

def consultGrammar(text, model="nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"):
    schema_inst = '{"accept": boolean, "corrections": "string"}'
    raw = consultAgent('grammar', text, model=model, response_schema_instructions=schema_inst)
    try:
        return parse_json_markdown(raw)
    except Exception:
        return raw

def consultTest(text, model="nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"):
    return consultAgent('test', text, model=model)

def consultNovelty(text, full_paper_text="", model="nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"):
    if full_paper_text:
        # Use RAG to get relevant context from the paper for novelty
        context = search_relevant_context(text, full_paper_text, top_k=2)
        query = f"Context: {context}\n\nSection to evaluate: {text}"
    else:
        query = text
    schema_inst = '{"novel": boolean, "reasoning": "string"}'
    raw = consultAgent('novelty', query, model=model, response_schema_instructions=schema_inst)
    try:
        return parse_json_markdown(raw)
    except Exception:
        return raw

def parse_bibliography(paper_text):
    entries = {}
    bib_matches = re.finditer(r'@(\w+)\s*\{\s*([^,\s]+),', paper_text)
    for match in bib_matches:
        key = match.group(2).strip()
        start_idx = match.end()
        brace_count = 1
        block_text = ""
        for i in range(start_idx, len(paper_text)):
            if i >= len(paper_text):
                break
            char = paper_text[i]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            if brace_count == 0:
                block_text = paper_text[start_idx:i]
                break
        
        # Parse title and author
        title_match = re.search(r'title\s*=\s*[\{"\']*(.+?)[\}"\']*\s*(?:,|$)', block_text, re.IGNORECASE)
        author_match = re.search(r'author\s*=\s*[\{"\']*(.+?)[\}"\']*\s*(?:,|$)', block_text, re.IGNORECASE)
        
        entry_details = []
        if title_match:
            title_val = title_match.group(1).strip().strip('{}""\'\'')
            entry_details.append(title_val)
        if author_match:
            author_val = author_match.group(1).strip().strip('{}""\'\'')
            entry_details.append(author_val)
            
        if entry_details:
            entries[key] = " ".join(entry_details)
    return entries

def resolve_query(query, bib_mapping):
    clean_key = query.strip()
    cite_match = re.search(r'\\cite\{([^}]+)\}', clean_key)
    if cite_match:
        clean_key = cite_match.group(1)
    
    # Check if direct match
    for key in bib_mapping:
        if key.lower() == clean_key.lower() or clean_key.lower() in key.lower():
            resolved = bib_mapping[key]
            print(f"Resolving citation key '{clean_key}' to paper title/author: '{resolved}'")
            return resolved
            
    # Comma-separated list handling
    keys = [k.strip() for k in clean_key.split(',')]
    resolved_parts = []
    for k in keys:
        for key in bib_mapping:
            if key.lower() == k.lower():
                resolved_parts.append(bib_mapping[key])
                break
    if resolved_parts:
        resolved = " OR ".join(resolved_parts)
        print(f"Resolving citation keys '{clean_key}' to: '{resolved}'")
        return resolved
        
    return query

def consultFactChecker(text, full_paper_text="", model="nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "searchArxiv",
                "description": "Search arXiv academic papers for claims or citations",
                "parameters": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Keywords to search on arXiv"
                        }
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "searchWeb",
                "description": "Search the general web via DuckDuckGo for tools, terms, or facts",
                "parameters": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Keywords or question to search on the web"
                        }
                    }
                }
            }
        }
    ]

    if full_paper_text:
        tools.append({
            "type": "function",
            "function": {
                "name": "searchInternalPaper",
                "description": "Search and cross-reference other sections of this paper using RAG",
                "parameters": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The claim or query to search internally within the paper"
                        }
                    }
                }
            }
        })

    # Parse bibliography from full paper text
    bib_mapping = parse_bibliography(full_paper_text) if full_paper_text else {}

    # Filter text to only keep sentences containing \cite
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cite_sentences = []
    for s in sentences:
        if r'\cite' in s:
            cite_sentences.append(s.strip())

    if not cite_sentences:
        print("No citation-supported claims found in the text. Skipping fact checking API call.")
        return {"accept": True, "corrections": "No citation-supported claims found in this section to fact check."}

    filtered_text = " ".join(cite_sentences)

    system_prompt = system_prompts.get('factchecker', "You are a fact checker.")
    system_prompt += "\nEvaluate the claims in the text. Focus ONLY on validating factual claims supported by citations (e.g. \\cite{...}). IGNORE other draft placeholders, formatting outlines, or custom macros (such as \\kunal{...}) that do not represent cited claims. If you need information to verify a cited claim, call a tool (arXiv, Web, or Internal RAG). Once you have enough context, return the final evaluation."
    system_prompt += '\n\nCRITICAL: When returning your final evaluation, you MUST respond ONLY with a JSON object matching this schema:\n{"accept": boolean, "corrections": "string"}\nDo not include any preambles or notes outside the JSON.'

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Text to evaluate:\n{filtered_text}"}
    ]

    try:
        raw = ""
        for turn in range(5):
            response = litellm.completion(
                model=model,
                messages=messages,
                tools=tools if tools else None
            )
            
            message = response.choices[0].message
            if not message.tool_calls:
                raw = message.content or ""
                break
                
            messages.append(message)
            for tool in message.tool_calls:
                tool_name = tool.function.name
                tool_args = json.loads(tool.function.arguments)
                query = tool_args.get("query", "")

                print(f"Tool Call (Turn {turn + 1}): {tool_name} with query: {query}")
                
                if tool_name == "searchArxiv":
                    resolved_query = resolve_query(query, bib_mapping)
                    output = consultArxiv(resolved_query)
                elif tool_name == "searchWeb":
                    resolved_query = resolve_query(query, bib_mapping)
                    output = consultWeb(resolved_query)
                elif tool_name == "searchInternalPaper":
                    if full_paper_text:
                        output = search_relevant_context(query, full_paper_text, top_k=2)
                    else:
                        output = "No internal paper text available for cross-referencing."
                else:
                    output = "Unknown tool."

                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "tool_call_id": tool.id,
                    "content": output
                })
        else:
            print("Reached maximum tool calling turns. Requesting final decision.")
            messages.append({
                "role": "user",
                "content": "You have reached the limit of search turns. Do not call any more tools. Based on the information you have gathered so far, provide your final evaluation now. Remember you MUST respond ONLY with a JSON object matching this schema:\n{\"accept\": boolean, \"corrections\": \"string\"}\nDo not include any preambles or notes outside the JSON."
            })
            final_response = litellm.completion(
                model=model,
                messages=messages
            )
            raw = final_response.choices[0].message.content

        try:
            return parse_json_markdown(raw)
        except Exception:
            return {"accept": False, "corrections": f"Failed to parse JSON. Raw output: {raw}"}
    except Exception as e:
        print(f"Fact checking error: {e}")
        return {"accept": False, "corrections": f"Fact checking error: {e}"}

available_functions = {
    'consultWiki': consultWiki,
    'consultArxiv': consultArxiv,
    'consultWeb': consultWeb,
    'consultDeskReviewer': consultDeskReviewer,
    'consultQuestioner': consultQuestioner,
    'consultGrammar': consultGrammar,
    'consultTest': consultTest,
    'consultNovelty': consultNovelty,
    'consultFactChecker': consultFactChecker,
}
