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
    split_text_into_sections, reviewer_agent, summarizer
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

start_time = time.time()
for section_name in sections_to_process:
    print(f"\n\nProcessing section: {section_name}")
    
    section_text = extract_section(args.pdf_path, section_name) if args.pdf_path.endswith(".pdf") else next(s[1] for s in sections if s[0] == section_name)
    
    print("\n🔍 **Extracted Section:**")
    print(section_text[:1000])

    similar_paper_data = generate_base_models(args.url, section_text)
   
    print(f"\n📢 **Reviewers Begin Discussion for {section_name}:**\n")
    review_outputs = {model: reviewer_agent(assigned_reviewers[0], section_text, model) for model in MODELS}

def modern_aggregate_reviews(review_list: List[str]) -> str:
    """Aggregate reviews using modern transformer-based sentiment analysis and summarization.
    
    Args:
        review_list: List of review texts to aggregate
        
    Returns:
        Aggregated and summarized review text
    """
    if not review_list:
        logging.warning("Empty review list provided for aggregation")
        return "No reviews to aggregate"
        
    try:
        # Use modern sentiment analysis from transformers
        sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            return_all_scores=True
        )
        
        # Analyze sentiment for each review
        sentiments = []
        for review in review_list:
            # Truncate review if too long for the model
            truncated_review = review[:512] if len(review) > 512 else review
            try:
                sentiment_scores = sentiment_pipeline(truncated_review)
                # Get confidence score (sum of all scores)
                confidence = sum(score['score'] for score in sentiment_scores[0])
                sentiments.append(confidence)
            except Exception as e:
                logging.warning(f"Error analyzing sentiment for review: {e}")
                sentiments.append(0.5)  # Default neutral score
        
        # Calculate weights based on sentiment confidence
        total_confidence = sum(sentiments) + 1e-6
        normalized_weights = [s / total_confidence for s in sentiments]
        
        # Create weighted combination of reviews
        weighted_reviews = []
        for review, weight in zip(review_list, normalized_weights):
            # Repeat review content based on weight (capped at 3 repetitions)
            repetitions = max(1, min(3, int(weight * 10)))
            weighted_reviews.extend([review] * repetitions)
        
        combined_text = " ".join(weighted_reviews)
        
        # Use summarization with better error handling
        try:
            summarizer_model = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn",
                max_length=150,
                min_length=40,
                do_sample=False
            )
            
            # Truncate if text is too long
            if len(combined_text) > 1024:
                combined_text = combined_text[:1024]
                
            summary = summarizer_model(combined_text)
            return summary[0]['summary_text']
            
        except Exception as e:
            logging.error(f"Error in summarization: {e}")
            # Fallback to simple truncation
            return combined_text[:300] + "..." if len(combined_text) > 300 else combined_text
            
    except Exception as e:
        logging.error(f"Error in modern review aggregation: {e}")
        # Fallback to simple concatenation
        return " ".join(review_list[:3])  # Take first 3 reviews as fallback

    aggregated_review = modern_aggregate_reviews(list(review_outputs.values()))
    final_summary = summarizer(section_text, aggregated_review)

    if "DeskReviewer" not in all_section_reviews:
        desk_review = consult_desk_reviewer(sections[0][1])
        all_section_reviews["DeskReviewer"] = {"Review": desk_review[1], "Accept": desk_review[0]}

    all_section_reviews[section_name] = {
        "Test": consult_test(section_text),
        "Reviewers": review_outputs,
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
