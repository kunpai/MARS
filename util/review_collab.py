import argparse
from PyPDF2 import PdfReader
import re
import json
import litellm
from pydantic import BaseModel, Field
from util.reviewer import assigned_reviewers, reviewer_messages

class ReviewScores(BaseModel):
    quality: int = Field(..., description="Quality score from 1 to 10")
    novelty: int = Field(..., description="Novelty score from 1 to 10")
    soundness: int = Field(..., description="Soundness score from 1 to 10")

class ReviewResponse(BaseModel):
    decision: str = Field(..., description="Final decision: Accept, Reject, WeakAccept, or WeakReject")
    scores: ReviewScores
    review: str = Field(..., description="Detailed feedback explaining the decision and scores")

def parse_pdf_to_text(pdf_path):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        return f"Error parsing PDF: {e}"

def clean_text(text):
    """Cleans extracted text by fixing spacing issues and formatting errors."""
    text = re.sub(r'([A-Z])\s+([A-Z][a-z])', r'\1\2', text)  
    text = re.sub(r'([^I-Z])((?:IX|IV|V?I{1,3}|I[XV]|X{1,3}|VI{1,3})\.)', r'\1\n\2', text)  
    text = re.sub(r'([IVX]+)\.\s*([A-Z])', r'\1. \2', text)  
    return text

def split_text_into_sections(text):
    """Splits text into sections based on research paper headers."""
    section_pattern = r"""
        (?:^|\n)                   
        (?:
            (?:[IVX]+\.)\s*       
            [A-Z][A-Za-z\s]*      
            |
            (?:Abstract|ABSTRACT|ACKNOWLEDGMENTS|REFERENCES)  
        )
    """
    headers = []
    for match in re.finditer(section_pattern, text, re.VERBOSE | re.MULTILINE):
        header_text = match.group().strip()
        if "et al." in header_text.lower() or re.search(r'\[\d+\]', header_text) or "TABLE" in header_text:
            continue
        headers.append((match.start(), header_text))

    headers.append((len(text), "END"))
    sections = []

    for i in range(len(headers) - 1):
        start_pos, header = headers[i]
        end_pos = headers[i + 1][0]
        content = text[start_pos:end_pos].strip()
        if content and not content.isspace():
            sections.append((header, content))

    return sections

def extract_section(pdf_path, section_name):
    """Extracts a section based on approximate name matching."""
    pdf_text = parse_pdf_to_text(pdf_path)
    if pdf_text.startswith("Error"):
        return pdf_text

    pdf_text = clean_text(pdf_text)
    sections = split_text_into_sections(pdf_text)

    for header, content in sections:
        if section_name.lower() in header.lower():
            return content

    return f"Section '{section_name}' not found. Try a different section."

def reviewer_agent(reviewer, section_text, model, previous_feedback=None):
    """LLM agent that reviews a section based on assigned reviewer attributes and provides a decision."""
    prompt = f"""
    {reviewer_messages[assigned_reviewers.index(reviewer)]}
    
    The section for review:
    "{section_text}"
    
    {f"Previous discussion so far: {previous_feedback}" if previous_feedback else ""}
    """
    try:
        response = litellm.completion(
            model=model, 
            messages=[{"role": "user", "content": prompt}], 
            response_format=ReviewResponse
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response received from the model")
        return content
    except Exception as e:
        return json.dumps({
            "decision": "Error",
            "scores": {"quality": 0, "novelty": 0, "soundness": 0},
            "review": f"Error during review: {e}"
        })

def board_room_review(reviewers, section_text, models):
    """
    Simulates a board room where multiple reviewers independently review a section,
    then their reviews are shared to a meta reviewer to form a consensus.
    """
    initial_reviews = {}
    for i, reviewer in enumerate(reviewers):
        model = models[i % len(models)]
        initial_reviews[reviewer.name] = reviewer_agent(reviewer, section_text, model)

    combined_reviews = ""
    for name, review_json in initial_reviews.items():
        try:
            r = json.loads(review_json)
            combined_reviews += f"Reviewer {name}: Decision={r.get('decision')}, Scores={r.get('scores')}, Review={r.get('review')}\n"
        except json.JSONDecodeError:
            combined_reviews += f"Reviewer {name} raw output: {review_json}\n"

    # Meta reviewer phase
    summary_prompt = f"""Summarize the board room discussion among reviewers about the following research paper section.
    
    Section: "{section_text}"

    Reviews:
    {combined_reviews}

    Format the summary as if recording minutes of a meeting. Highlight agreements, disagreements, and key takeaways.
    
    🔹 **At the end, determine the final decision based on the majority vote (Accept, Reject, WeakAccept, WeakReject).**
    """
    try:
        response = litellm.completion(model=models[0], messages=[{"role": "user", "content": summary_prompt}])
        summary = response.choices[0].message.content
    except Exception as e:
        summary = f"Error generating summary: {e}"

    return initial_reviews, summary

def summarizer(section_text, aggregated_review):
    # This keeps compatibility with older code if needed
    prompt = f"""Summarize the discussion about the following research paper section.

    Section: "{section_text}"

    Aggregated Review:
    {aggregated_review}

    Format the summary as if recording minutes of a meeting. Highlight agreements, disagreements, and key takeaways.
    """
    try:
        response = litellm.completion(model="nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def main():
    parser = argparse.ArgumentParser(description="Extract and discuss a specific section of a research paper.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("section_name", type=str, help="Name of the section to extract and discuss")

    args = parser.parse_args()
    section_text = extract_section(args.pdf_path, args.section_name)
    if "not found" in section_text.lower():
        print(section_text)
        return

    print("\n🔍 **Extracted Section:**")
    print(section_text[:1000]) 

    print("\n📢 **Reviewers Begin Discussion in the Board Room:**\n")
    
    # Example using first 3 reviewers and dummy models for test
    reviewers = assigned_reviewers[:3]
    models = ["nvidia_nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"] * 3
    initial_reviews, final_summary = board_room_review(reviewers, section_text, models)
    
    for name, r in initial_reviews.items():
        print(f"🗣️ **{name}**: {r}\n")

    print("\n**Final Summary and Decision:**")
    print(final_summary)

if __name__ == "__main__":
    main()
