import os
import json
import re
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import requests
import ollama
from bs4 import BeautifulSoup
from ollama import chat, ChatResponse
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from util.review_collab import (
    parse_pdf_to_text, clean_text, extract_section,
    split_text_into_sections, summarizer
)
from util.build_models import generate_base_models, generate_paper_models
from util.multiagent import (
    consultGrammar as consult_grammar,
    consultNovelty as consult_novelty,
    consultFactChecker as fact_checker,
    consultQuestioner as consult_question,
    consultTest as consult_test,
    consultDeskReviewer as consult_desk_reviewer
)
from util.reviewer import assigned_reviewers
from util.build_models import isModelLoaded
from util.langgraph_agents import create_review_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mars.log'),
        logging.StreamHandler()
    ]
)

# Constants
MODELS: List[str] = ["mistral", "llama3.2", "qwen2.5", "deepseek-r1"]
CHECKPOINT_FILE: str = "feedback_collab.json"
ANSWER_FILE: str = "feedback_collab_with_answers.json"
MODEL_LIST_FILE: str = "paper_specific_models.txt"

# Argument Parser
parser = argparse.ArgumentParser(description="MultiAgent Paper Review with Optional Q&A")
parser.add_argument("url", type=str, help="Path to the Conference CFP")
parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
parser.add_argument("section_name", type=str, nargs='?', default='', help="Optional: specific paper section for review")
parser.add_argument("--answer-questions", action="store_true", help="Enable answering questions in the second stage")
args = parser.parse_args()

# ---- Stage 1: Review Paper Sections ----

def load_document_sections(file_path: str) -> List[Tuple[str, str]]:
    """Load document sections from PDF or JSON file.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        List of (section_name, section_content) tuples
    """
    file_path_obj = Path(file_path)
    
    if not file_path_obj.exists():
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path_obj.suffix.lower() == ".pdf":
        logging.info(f"Processing PDF file: {file_path}")
        pdf_text = parse_pdf_to_text(file_path)
        if pdf_text.startswith("Error"):
            logging.error(f"Failed to parse PDF: {pdf_text}")
            return []
        cleaned_text = clean_text(pdf_text)
        return split_text_into_sections(cleaned_text)
    
    elif file_path_obj.suffix.lower() == ".json":
        logging.info(f"Processing JSON file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "input" in data and "sections" in data["input"]:
            return [(s["heading"], s["text"]) for s in data["input"]["sections"]]
        else:
            logging.error("Invalid JSON format - expected 'input.sections' structure")
            return []
    
    else:
        logging.error(f"Unsupported file format: {file_path_obj.suffix}")
        raise ValueError(f"Unsupported file format: {file_path_obj.suffix}")

# Parse and clean the document
try:
    sections = load_document_sections(args.pdf_path)
except (FileNotFoundError, ValueError) as e:
    logging.error(f"Error loading document: {e}")
    print(f"Error: {e}")
    exit(1)

print("\nAvailable Sections in the Paper:")
for section in sections:
    print(f"- {section[0]}")

# Load checkpoint if available
checkpoint_path = Path(CHECKPOINT_FILE)
if checkpoint_path.exists():
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
            all_section_reviews = checkpoint_data.get("Section Reviews", {})
        processed_sections = set(all_section_reviews.keys())
        logging.info(f"Loaded checkpoint with {len(processed_sections)} processed sections")
        print(f"\nFound checkpoint. Processed sections: {', '.join(processed_sections) if processed_sections else 'None'}")
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Error loading checkpoint: {e}")
        all_section_reviews = {}
        processed_sections = set()
else:
    all_section_reviews = {}
    processed_sections = set()
    processed_sections = set()

# Determine sections to process
if args.section_name:
    if args.section_name in processed_sections:
        print(f"\nSection '{args.section_name}' is already processed. Exiting.")
        exit(0)
    sections_to_process = [args.section_name]
else:
    sections_to_process = [s[0] for s in sections if s[0] not in processed_sections]

if not sections_to_process:
    print("\nNo new sections to process. Exiting.")
    exit(0)

def checkpoint_progress() -> None:
    """Save current progress to checkpoint file.
    
    Saves the current state of section reviews to allow resuming processing.
    """
    try:
        feedback = {
            "Available Sections": [s[0] for s in sections],
            "Section Reviews": all_section_reviews,
            "timestamp": time.time(),
            "processed_count": len(all_section_reviews)
        }
        
        checkpoint_path = Path(CHECKPOINT_FILE)
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(feedback, f, indent=4, ensure_ascii=False)
        
        logging.info(f"Checkpoint saved to {checkpoint_path}")
        print(f"\nCheckpoint saved to {CHECKPOINT_FILE}.")
        
    except (IOError, json.JSONEncodeError) as e:
        logging.error(f"Error saving checkpoint: {e}")
        print(f"Warning: Could not save checkpoint: {e}")

# Generate paper-specific models
paper_specific_models = generate_paper_models(sections)

# Initialize the LangGraph-based review system
print("\n🤖 Initializing LangGraph Multi-Agent Review System...")
try:
    review_system = create_review_system(model_name="llama3.2")
    logging.info("LangGraph review system initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize LangGraph review system: {e}")
    print(f"❌ Error: {e}")
    print("Falling back to basic review mode...")
    review_system = None

start_time = time.time()
for section_name in sections_to_process:
    print(f"\n\n{'='*60}")
    print(f"🔍 Processing section: {section_name}")
    print(f"{'='*60}")
    
    section_text = extract_section(args.pdf_path, section_name) if args.pdf_path.endswith(".pdf") else next(s[1] for s in sections if s[0] == section_name)
    
    print("\n📄 **Extracted Section Preview:**")
    print(section_text[:500] + "..." if len(section_text) > 500 else section_text)

    similar_paper_data = generate_base_models(args.url, section_text)
   
    # Use LangGraph multi-agent discussion if available
    if review_system:
        print(f"\n🗣️ **Multi-Agent Discussion for {section_name}:**")
        print("Three expert reviewers will now discuss this section to reach consensus...")
        
        try:
            # Run the multi-agent discussion
            review_result = review_system.review_section(section_text, section_name)
            
            # Display information about the generated reviewers
            print(f"\n🤖 **Dynamically Generated Reviewers:**")
            for reviewer in review_result.get('generated_reviewers', []):
                print(f"  • {reviewer['name']} ({reviewer['experience']})")
                print(f"    Expertise: {reviewer['expertise']}")
                print(f"    Style: {reviewer['style']}")
            
            # Display the discussion
            print(f"\n💬 **Discussion Summary:**")
            print(f"Rounds completed: {review_result['rounds_completed']}")
            print(f"Consensus reached: {review_result['consensus_reached']}")
            
            print(f"\n📋 **Individual Reviewer Contributions:**")
            for reviewer_name, review_content in review_result['individual_reviews'].items():
                print(f"\n{reviewer_name}:")
                print(f"  {review_content[:200]}...")
            
            print(f"\n🎯 **Final Decision:**")
            print(review_result['final_decision'])
            
            # Structure the output for compatibility
            review_outputs = {
                "langgraph_discussion": review_result['discussion_summary'],
                "final_decision": review_result['final_decision'],
                "individual_reviews": review_result['individual_reviews'],
                "consensus_reached": review_result['consensus_reached'],
                "generated_reviewers": review_result.get('generated_reviewers', [])
            }
            
            # Use the final decision as the aggregated review
            aggregated_review = review_result['final_decision']
            
        except Exception as e:
            logging.error(f"Error in LangGraph review: {e}")
            print(f"❌ Error in multi-agent discussion: {e}")
            print("📝 Falling back to individual model reviews...")
            
            # Fallback to original approach
            review_outputs = {}
            for model in MODELS:
                if isModelLoaded(model):
                    try:
                        response = ollama.chat(
                            model=model, 
                            messages=[{
                                "role": "user", 
                                "content": f"Review this paper section and provide Accept/Reject decision with reasoning:\n\n{section_text[:1000]}"
                            }]
                        )
                        review_outputs[model] = response['message']['content']
                    except Exception as model_error:
                        logging.error(f"Error with model {model}: {model_error}")
                        review_outputs[model] = f"Error: Could not get review from {model}"
            
            aggregated_review = "Multiple individual reviews completed due to discussion system error."
    
    else:
        # Fallback to original sequential approach if LangGraph failed to initialize
        print(f"\n📝 **Individual Model Reviews for {section_name}:**")
        review_outputs = {}
        for model in MODELS:
            if isModelLoaded(model):
                try:
                    response = ollama.chat(
                        model=model, 
                        messages=[{
                            "role": "user", 
                            "content": f"Review this paper section and provide Accept/Reject decision with reasoning:\n\n{section_text[:1000]}"
                        }]
                    )
                    review_outputs[model] = response['message']['content']
                    print(f"\n{model}: {response['message']['content'][:200]}...")
                except Exception as model_error:
                    logging.error(f"Error with model {model}: {model_error}")
                    review_outputs[model] = f"Error: Could not get review from {model}"
        
        aggregated_review = "Individual model reviews completed."

    # Generate final summary using the aggregated review
    final_summary = summarizer(section_text, aggregated_review)

    if "DeskReviewer" not in all_section_reviews:
        desk_review = consult_desk_reviewer(sections[0][1])
        all_section_reviews["DeskReviewer"] = {"Review": desk_review[1], "Accept": desk_review[0]}

    all_section_reviews[section_name] = {
        "Test": consult_test(section_text),
        "Multi-Agent Discussion": review_outputs,  # Updated to reflect new approach
        "Grammar Check": consult_grammar(section_text),
        "Novelty Check": consult_novelty(section_text),
        "Fact Check": fact_checker(section_text),
        "Questioner": consult_question(section_text),
        "Final Summary": aggregated_review + "\n" + final_summary
    }

    checkpoint_progress()

print("\nAll new sections processed. Final checkpoint saved.")
print(f"\nTotal time taken: {time.time() - start_time:.2f} seconds")

# Save paper-specific models list
try:
    model_list_path = Path(MODEL_LIST_FILE)
    with open(model_list_path, "w", encoding="utf-8") as f:
        for key in paper_specific_models:
            f.write(f"{key}\n")
    logging.info(f"Saved {len(paper_specific_models)} paper-specific models to {model_list_path}")
except (IOError, NameError) as e:
    logging.warning(f"Could not save paper-specific models: {e}")

# ---- Stage 2: Answering Questions (Optional) ----

if args.answer_questions:
    print("\nStarting Question-Answering Stage...")

    with open(MODEL_LIST_FILE, "r") as f:
        paper_specific_models = [line.strip() for line in f if line.strip()]

    with open(CHECKPOINT_FILE, "r") as f:
        feedback = json.load(f)

    if "Answers" not in feedback:
        feedback["Answers"] = {}

    start_time = time.time()
    for section_name, section_data in feedback["Section Reviews"].items():
        if section_name in feedback["Answers"]:
            print(f"\nSkipping already processed section: {section_name}")
            continue

        print(f"\nProcessing section: {section_name}")
        questions = section_data.get("Questioner", "").split("?")
        feedback["Answers"][section_name] = {}

        for question in questions:
            question = question.strip() + "?"
            if question == "?":
                continue

            print(f"Processing question: {question}")
            feedback["Answers"][section_name][question] = {}

            for model in paper_specific_models:
                if section_name == model:
                    continue
                answer = chat(model=model, messages=[{"role": "user", "content": question}]).message.content.strip()
                feedback["Answers"][section_name][question][model] = answer

        with open(ANSWER_FILE, "w") as f:
            json.dump(feedback, f, indent=4)

    print(f"\nAll questions answered in {time.time() - start_time:.2f} seconds")
    print(f"Final answers saved to {ANSWER_FILE}")
