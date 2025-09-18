import os
import json
import re
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import requests
import ollama
from bs4 import BeautifulSoup
from ollama import chat, ChatResponse
from transformers import pipeline
from loguru import logger
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text

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

# Initialize console for rich output
console = Console()

# Configure logging
logger.add(
    "logs/mars_{time}.log",
    rotation="1 day",
    retention="7 days",
    level="INFO",
    format="{time} | {level} | {message}"
)

# Constants
MODELS: List[str] = ["mistral", "llama3.2", "qwen2.5", "deepseek-r1"]
CHECKPOINT_FILE: str = "feedback_collab.json"
ANSWER_FILE: str = "feedback_collab_with_answers.json"
MODEL_LIST_FILE: str = "paper_specific_models.txt"

def setup_logging() -> None:
    """Setup logging directory if it doesn't exist."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="MARS: Multi-Agent Review System for Academic Papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python MARS.py https://example.com/cfp paper.pdf
  python MARS.py https://example.com/cfp paper.json --answer-questions
  python MARS.py https://example.com/cfp paper.pdf "Introduction"
        """
    )
    parser.add_argument(
        "url", 
        type=str, 
        help="URL to the Conference Call for Papers (CFP)"
    )
    parser.add_argument(
        "pdf_path", 
        type=str, 
        help="Path to the PDF file or JSON file containing paper content"
    )
    parser.add_argument(
        "section_name", 
        type=str, 
        nargs='?', 
        default='', 
        help="Optional: specific paper section for review"
    )
    parser.add_argument(
        "--answer-questions", 
        action="store_true", 
        help="Enable answering questions in the second stage"
    )
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true", 
        help="Enable verbose logging output"
    )
    return parser

def load_paper_sections(pdf_path: str) -> List[Tuple[str, str]]:
    """Load and parse paper sections from PDF or JSON file."""
    try:
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")
        
        if pdf_path.endswith(".pdf"):
            logger.info(f"Processing PDF file: {pdf_path}")
            console.print(f"📄 Processing PDF: [bold blue]{pdf_path}[/bold blue]")
            
            pdf_text = parse_pdf_to_text(pdf_path)
            if pdf_text.startswith("Error"):
                raise ValueError(pdf_text)
            
            cleaned_text = clean_text(pdf_text)
            sections = split_text_into_sections(cleaned_text)
            
        else:
            logger.info(f"Processing JSON file: {pdf_path}")
            console.print(f"📋 Processing JSON: [bold blue]{pdf_path}[/bold blue]")
            
            with open(pdf_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if "input" not in data or "sections" not in data["input"]:
                raise ValueError("Invalid JSON format: missing 'input.sections'")
                
            sections = [(s["heading"], s["text"]) for s in data["input"]["sections"]]
        
        if not sections:
            raise ValueError("No sections found in the document")
            
        logger.info(f"Successfully loaded {len(sections)} sections")
        return sections
        
    except Exception as e:
        logger.error(f"Error loading paper sections: {e}")
        console.print(f"[red]❌ Error loading paper: {e}[/red]")
        raise

def load_checkpoint() -> Tuple[Dict[str, Any], set]:
    """Load existing checkpoint data if available."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            all_section_reviews = checkpoint_data.get("Section Reviews", {})
            processed_sections = set(all_section_reviews.keys())
            
            logger.info(f"Loaded checkpoint with {len(processed_sections)} processed sections")
            console.print(f"💾 Found checkpoint with {len(processed_sections)} processed sections")
            
            return all_section_reviews, processed_sections
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            console.print(f"[yellow]⚠️ Could not load checkpoint: {e}[/yellow]")
            
    return {}, set()

def fancy_aggregate_reviews(review_list: List[str]) -> str:
    """Aggregate multiple reviews using modern NLP techniques."""
    if not review_list:
        return "No reviews to aggregate"
    
    try:
        # Use a more modern approach for review aggregation
        from textblob import TextBlob
        
        # Calculate sentiment and confidence for each review
        reviews_with_sentiment = []
        for review in review_list:
            blob = TextBlob(review)
            sentiment = blob.sentiment
            # Use polarity and subjectivity to determine weight
            weight = abs(sentiment.polarity) * (1 - sentiment.subjectivity)
            reviews_with_sentiment.append((review, weight))
        
        # Weight reviews by their sentiment confidence
        total_weight = sum(weight for _, weight in reviews_with_sentiment) + 1e-6
        normalized_weights = [weight / total_weight for _, weight in reviews_with_sentiment]
        
        # Create weighted summary text
        weighted_text = " ".join([
            review * max(1, int(weight * 10)) 
            for (review, _), weight in zip(reviews_with_sentiment, normalized_weights)
        ])
        
        # Use modern summarization
        summarizer_model = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn",
            device_map="auto" if hasattr(pipeline, "device_map") else None
        )
        
        # Truncate if too long for the model
        max_length = min(1024, len(weighted_text))
        if len(weighted_text) > max_length:
            weighted_text = weighted_text[:max_length]
        
        summary = summarizer_model(
            weighted_text, 
            max_length=150, 
            min_length=40, 
            do_sample=False
        )
        
        return summary[0]['summary_text']
        
    except Exception as e:
        logger.error(f"Error in review aggregation: {e}")
        # Fallback to simple concatenation
        return " | ".join(review_list[:3])  # Take first 3 reviews

def checkpoint_progress(sections: List[Tuple[str, str]], all_section_reviews: Dict[str, Any]) -> None:
    """Save progress to checkpoint file."""
    try:
        feedback = {
            "Available Sections": [s[0] for s in sections],
            "Section Reviews": all_section_reviews
        }
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(feedback, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Checkpoint saved to {CHECKPOINT_FILE}")
        console.print(f"💾 Checkpoint saved")
        
    except Exception as e:
        logger.error(f"Error saving checkpoint: {e}")
        console.print(f"[red]❌ Error saving checkpoint: {e}[/red]")

def process_sections(
    sections_to_process: List[str], 
    sections: List[Tuple[str, str]], 
    all_section_reviews: Dict[str, Any],
    args: argparse.Namespace
) -> List[str]:
    """Process paper sections with multi-agent review."""
    
    # Generate paper-specific models
    console.print("🔧 Generating paper-specific models...")
    paper_specific_models = generate_paper_models(sections)
    
    start_time = time.time()
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("Processing sections...", total=len(sections_to_process))
        
        for section_name in sections_to_process:
            progress.update(task, description=f"Processing: {section_name}")
            
            logger.info(f"Processing section: {section_name}")
            console.print(Panel(f"Processing section: [bold cyan]{section_name}[/bold cyan]"))
            
            # Extract section text
            if args.pdf_path.endswith(".pdf"):
                section_text = extract_section(args.pdf_path, section_name)
            else:
                section_text = next((s[1] for s in sections if s[0] == section_name), "")
            
            if not section_text or section_text.startswith("Error"):
                logger.error(f"Could not extract section: {section_name}")
                console.print(f"[red]❌ Could not extract section: {section_name}[/red]")
                continue
            
            console.print(f"📄 Extracted section ({len(section_text)} chars)")
            logger.info(f"Extracted section text: {len(section_text)} characters")
            
            # Generate base models for this section
            similar_paper_data = generate_base_models(args.url, section_text)
            
            # Get reviews from different models
            console.print("🤖 Getting reviews from AI models...")
            review_outputs = {}
            for model in MODELS:
                try:
                    review = reviewer_agent(assigned_reviewers[0], section_text, model)
                    review_outputs[model] = review
                    console.print(f"  ✅ {model}")
                except Exception as e:
                    logger.error(f"Error getting review from {model}: {e}")
                    console.print(f"  ❌ {model}: {e}")
                    review_outputs[model] = f"Error: {e}"
            
            # Aggregate reviews
            aggregated_review = fancy_aggregate_reviews(list(review_outputs.values()))
            final_summary = summarizer(section_text, aggregated_review)
            
            # Run additional checks
            console.print("🔍 Running additional checks...")
            
            # Desk reviewer check (only once)
            if "DeskReviewer" not in all_section_reviews and sections:
                try:
                    desk_review = consult_desk_reviewer(sections[0][1])
                    all_section_reviews["DeskReviewer"] = {
                        "Review": desk_review[1], 
                        "Accept": desk_review[0]
                    }
                except Exception as e:
                    logger.error(f"Error in desk review: {e}")
            
            # Compile all section reviews
            all_section_reviews[section_name] = {
                "Test": consult_test(section_text),
                "Reviewers": review_outputs,
                "Grammar Check": consult_grammar(section_text),
                "Novelty Check": consult_novelty(section_text),
                "Fact Check": fact_checker(section_text),
                "Questioner": consult_question(section_text),
                "Final Summary": aggregated_review + "\n" + final_summary
            }
            
            # Save progress
            checkpoint_progress(sections, all_section_reviews)
            progress.advance(task)
    
    elapsed_time = time.time() - start_time
    console.print(f"✅ All sections processed in {elapsed_time:.2f} seconds")
    logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
    
    # Save model list
    try:
        with open(MODEL_LIST_FILE, "w") as f:
            for key in paper_specific_models:
                f.write(f"{key}\n")
        logger.info(f"Model list saved to {MODEL_LIST_FILE}")
    except Exception as e:
        logger.error(f"Error saving model list: {e}")
    
    return paper_specific_models

def process_questions(paper_specific_models: List[str]) -> None:
    """Process questions from the review stage."""
    console.print("\n🤔 Starting Question-Answering Stage...")
    
    try:
        # Load model list
        if os.path.exists(MODEL_LIST_FILE):
            with open(MODEL_LIST_FILE, "r") as f:
                paper_specific_models = [line.strip() for line in f if line.strip()]
        
        # Load feedback data
        if not os.path.exists(CHECKPOINT_FILE):
            console.print("[red]❌ No checkpoint file found for question answering[/red]")
            return
            
        with open(CHECKPOINT_FILE, "r") as f:
            feedback = json.load(f)
        
        if "Answers" not in feedback:
            feedback["Answers"] = {}
        
        start_time = time.time()
        sections_to_process = [
            section_name for section_name in feedback["Section Reviews"].keys() 
            if section_name not in feedback["Answers"]
        ]
        
        if not sections_to_process:
            console.print("✅ All questions already answered")
            return
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Answering questions...", total=len(sections_to_process))
            
            for section_name in sections_to_process:
                progress.update(task, description=f"Processing: {section_name}")
                
                section_data = feedback["Section Reviews"][section_name]
                questions_text = section_data.get("Questioner", "")
                
                if not questions_text:
                    progress.advance(task)
                    continue
                
                # Parse questions
                questions = [q.strip() + "?" for q in questions_text.split("?") if q.strip()]
                feedback["Answers"][section_name] = {}
                
                for question in questions:
                    if question == "?":
                        continue
                    
                    logger.info(f"Processing question: {question}")
                    feedback["Answers"][section_name][question] = {}
                    
                    # Get answers from paper-specific models
                    for model in paper_specific_models:
                        if section_name == model:
                            continue
                        
                        try:
                            response = chat(
                                model=model, 
                                messages=[{"role": "user", "content": question}]
                            )
                            answer = response.message.content.strip()
                            feedback["Answers"][section_name][question][model] = answer
                        except Exception as e:
                            logger.error(f"Error getting answer from {model}: {e}")
                            feedback["Answers"][section_name][question][model] = f"Error: {e}"
                
                # Save intermediate progress
                with open(ANSWER_FILE, "w") as f:
                    json.dump(feedback, f, indent=4)
                
                progress.advance(task)
        
        elapsed_time = time.time() - start_time
        console.print(f"✅ Question answering completed in {elapsed_time:.2f} seconds")
        console.print(f"📁 Final answers saved to [bold green]{ANSWER_FILE}[/bold green]")
        
    except Exception as e:
        logger.error(f"Error in question processing: {e}")
        console.print(f"[red]❌ Error in question processing: {e}[/red]")

def main() -> None:
    """Main function to orchestrate the paper review process."""
    setup_logging()
    
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logger.configure(handlers=[{"sink": lambda msg: console.print(f"[dim]{msg}[/dim]", markup=False)}])
    
    try:
        console.print(Panel.fit(
            "🚀 [bold blue]MARS: Multi-Agent Review System[/bold blue] 🚀\n"
            "Academic Paper Review with AI Agents",
            border_style="blue"
        ))
        
        # Load paper sections
        sections = load_paper_sections(args.pdf_path)
        
        # Display available sections
        console.print("\n📋 Available Sections in the Paper:")
        for i, (section_name, _) in enumerate(sections, 1):
            console.print(f"  {i}. [cyan]{section_name}[/cyan]")
        
        # Load checkpoint
        all_section_reviews, processed_sections = load_checkpoint()
        
        # Determine sections to process
        if args.section_name:
            if args.section_name in processed_sections:
                console.print(f"[yellow]⚠️ Section '{args.section_name}' already processed[/yellow]")
                return
            sections_to_process = [args.section_name]
        else:
            sections_to_process = [s[0] for s in sections if s[0] not in processed_sections]
        
        if not sections_to_process:
            console.print("[yellow]✅ No new sections to process[/yellow]")
            if args.answer_questions:
                process_questions([])
            return
        
        console.print(f"\n🎯 Sections to process: {len(sections_to_process)}")
        for section in sections_to_process:
            console.print(f"  • [green]{section}[/green]")
        
        # Process sections
        paper_specific_models = process_sections(
            sections_to_process, sections, all_section_reviews, args
        )
        
        # Process questions if requested
        if args.answer_questions:
            process_questions(paper_specific_models)
        
        console.print("\n🎉 [bold green]MARS processing completed successfully![/bold green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Process interrupted by user[/yellow]")
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        console.print(f"[red]💥 Fatal error: {e}[/red]")
        raise

if __name__ == "__main__":
    main()
