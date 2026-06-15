import os
import json
import time
import argparse
import litellm

from util.review_collab import (
    parse_pdf_to_text, clean_text, split_text_into_sections,
    board_room_review, summarizer
)
from util.build_models import generate_base_models, generate_paper_models
from util.multiagent import (
    consultGrammar as consult_grammar,
    consultNovelty as consult_novelty,
    consultFactChecker as fact_checker,
    consultQuestioner as consult_question,
    consultTest as consult_test,
    consultDeskReviewer as consult_desk_reviewer,
    set_system_prompts
)
from util.reviewer import assigned_reviewers
from util.rag import search_relevant_context

# Constants
MODELS = ["ollama/mistral", "ollama/llama3.2", "ollama/qwen2.5"]
CHECKPOINT_FILE = "feedback_collab.json"
ANSWER_FILE = "feedback_collab_with_answers.json"

# Argument Parser
parser = argparse.ArgumentParser(description="MultiAgent Paper Review with Optional Q&A")
parser.add_argument("url", type=str, help="Path to the Conference CFP")
parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
parser.add_argument("section_name", type=str, nargs='?', default='', help="Optional: specific paper section for review")
parser.add_argument("--answer-questions", action="store_true", help="Enable answering questions in the second stage")
args = parser.parse_args()

# ---- Stage 1: Review Paper Sections ----

# Parse and clean the PDF text
if args.pdf_path.endswith(".pdf"):
    pdf_text = parse_pdf_to_text(args.pdf_path)
    cleaned_text = clean_text(pdf_text)
    sections = split_text_into_sections(cleaned_text)
else:
    sections = []
    with open(args.pdf_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "input" in data and "sections" in data["input"]:
        sections = [(s["heading"], s["text"]) for s in data["input"]["sections"]]

print("\nAvailable Sections in the Paper:")
for section in sections:
    print(f"- {section[0]}")

full_paper_text = " ".join([s[1] for s in sections])

if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        checkpoint_data = json.load(f)
        all_section_reviews = checkpoint_data.get("Section Reviews", {})
    processed_sections = set(all_section_reviews.keys())
    print(f"\nFound checkpoint. Processed sections: {', '.join(processed_sections) if processed_sections else 'None'}")
else:
    all_section_reviews = {}
    processed_sections = set()

if args.section_name:
    if args.section_name in processed_sections:
        print(f"\nSection '{args.section_name}' is already processed. Exiting.")
        exit(0)
    sections_to_process = [args.section_name]
else:
    sections_to_process = [s[0] for s in sections if s[0] not in processed_sections]

if not sections_to_process:
    print("\nNo new sections to process.")

def checkpoint_progress():
    feedback = {
        "Available Sections": [s[0] for s in sections],
        "Section Reviews": all_section_reviews
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(feedback, f, indent=4, ensure_ascii=False)
    print(f"\nCheckpoint saved to {CHECKPOINT_FILE}.")

if sections_to_process:
    # Setup global prompts for the current run
    base_prompts = generate_base_models(args.url, full_paper_text)
    set_system_prompts(base_prompts)

    start_time = time.time()
    for section_name in sections_to_process:
        print(f"\n\nProcessing section: {section_name}")

        section_text = next((s[1] for s in sections if s[0] == section_name), "")
        if not section_text:
            continue

        print("\n🔍 **Extracted Section:**")
        print(section_text[:1000])

        print(f"\n📢 **Reviewers Begin Discussion in Board Room for {section_name}:**\n")

        # Use first 3 reviewers for board room
        reviewers = assigned_reviewers[:3]
        initial_reviews, final_summary = board_room_review(reviewers, section_text, MODELS)

        if "DeskReviewer" not in all_section_reviews and sections:
            desk_review = consult_desk_reviewer(sections[0][1], model=MODELS[0])
            all_section_reviews["DeskReviewer"] = {"Review": desk_review[1], "Accept": desk_review[0]}

        all_section_reviews[section_name] = {
            "Test": consult_test(section_text, model=MODELS[0]),
            "Reviewers": initial_reviews,
            "Grammar Check": consult_grammar(section_text, model=MODELS[0]),
            "Novelty Check": consult_novelty(section_text, full_paper_text=full_paper_text, model=MODELS[0]),
            "Fact Check": fact_checker(section_text, full_paper_text=full_paper_text, model=MODELS[0]),
            "Questioner": consult_question(section_text, model=MODELS[0]),
            "Final Summary": final_summary
        }

        checkpoint_progress()

    print("\nAll new sections processed. Final checkpoint saved.")
    print(f"\nTotal time taken: {time.time() - start_time:.2f} seconds")

# ---- Stage 2: Answering Questions (Optional) ----

if args.answer_questions:
    print("\nStarting Question-Answering Stage using RAG...")

    if not os.path.exists(CHECKPOINT_FILE):
        print("Checkpoint file not found. Cannot answer questions.")
        exit(1)

    with open(CHECKPOINT_FILE, "r") as f:
        feedback = json.load(f)

    if "Answers" not in feedback:
        feedback["Answers"] = {}

    start_time = time.time()
    for section_name, section_data in feedback["Section Reviews"].items():
        if section_name == "DeskReviewer":
            continue

        if section_name in feedback["Answers"]:
            print(f"\nSkipping already processed section: {section_name}")
            continue

        print(f"\nProcessing section questions: {section_name}")
        questions_raw = section_data.get("Questioner", "")
        if not questions_raw:
            continue

        questions = [q.strip() + "?" for q in questions_raw.split("?") if q.strip()]
        feedback["Answers"][section_name] = {}

        for question in questions:
            print(f"Processing question: {question}")
            feedback["Answers"][section_name][question] = {}

            # Using RAG to get the most relevant context from the section (or full paper)
            context = search_relevant_context(question, section_text if section_text else full_paper_text, top_k=2)
            prompt = f"Use the following context to answer the question.\n\nContext: {context}\n\nQuestion: {question}"

            for model in MODELS:
                try:
                    response = litellm.completion(model=model, messages=[{"role": "user", "content": prompt}])
                    answer = response.choices[0].message.content.strip()
                except Exception as e:
                    answer = f"Error generating answer: {e}"
                feedback["Answers"][section_name][question][model] = answer

        with open(ANSWER_FILE, "w") as f:
            json.dump(feedback, f, indent=4)

    print(f"\nAll questions answered in {time.time() - start_time:.2f} seconds")
    print(f"Final answers saved to {ANSWER_FILE}")