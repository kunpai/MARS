import ollama
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from PyPDF2 import PdfReader
import re
from util.reviewer import assigned_reviewers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  

def parse_pdf_to_text(pdf_path: str | Path) -> str:
    """Extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from the PDF or error message
    """
    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logging.error(f"PDF file not found: {pdf_path}")
            return f"Error: PDF file not found: {pdf_path}"
            
        if not pdf_path.suffix.lower() == '.pdf':
            logging.error(f"File is not a PDF: {pdf_path}")
            return f"Error: File is not a PDF: {pdf_path}"
            
        reader = PdfReader(pdf_path)
        text_parts = []
        
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
            else:
                logging.warning(f"No text found on page {page_num}")
                
        if not text_parts:
            logging.warning("No text extracted from PDF")
            return "Warning: No text could be extracted from the PDF"
            
        return "\n".join(text_parts)
        
    except Exception as e:
        logging.error(f"Error parsing PDF {pdf_path}: {e}")
        return f"Error parsing PDF: {e}"

def clean_text(text: str) -> str:
    """Cleans extracted text by fixing spacing issues and formatting errors.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text with better formatting
    """
    if not text:
        return ""
        
    # Fix spacing between words
    text = re.sub(r'([A-Z])\s+([A-Z][a-z])', r'\1\2', text)  
    # Handle Roman numerals section headers
    text = re.sub(r'([^I-Z])((?:IX|IV|V?I{1,3}|I[XV]|X{1,3}|VI{1,3})\.)', r'\1\n\2', text)  
    # Clean up Roman numeral formatting
    text = re.sub(r'([IVX]+)\.\s*([A-Z])', r'\1. \2', text)  
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Clean up line breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()

def split_text_into_sections(text: str) -> List[Tuple[str, str]]:
    """Splits text into sections based on research paper headers.
    
    Args:
        text: Input text to split into sections
        
    Returns:
        List of tuples containing (header, content) pairs
    """
    if not text:
        logging.warning("Empty text provided for section splitting")
        return []
        
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
        # Skip false positives
        if ("et al." in header_text.lower() or 
            re.search(r'\[\d+\]', header_text) or 
            "TABLE" in header_text):
            continue
        headers.append((match.start(), header_text))

    if not headers:
        logging.warning("No section headers found in text")
        return [("Full Text", text)]

    headers.append((len(text), "END"))
    sections = []

    for i in range(len(headers) - 1):
        start_pos, header = headers[i]
        end_pos = headers[i + 1][0]
        content = text[start_pos:end_pos].strip()
        if content and not content.isspace():
            sections.append((header, content))

    logging.info(f"Found {len(sections)} sections in text")
    return sections

def extract_section(pdf_path: str | Path, section_name: str) -> str:
    """Extracts a section based on approximate name matching.
    
    Args:
        pdf_path: Path to the PDF file
        section_name: Name of the section to extract
        
    Returns:
        Content of the requested section or error message
    """
    pdf_text = parse_pdf_to_text(pdf_path)
    if pdf_text.startswith("Error") or pdf_text.startswith("Warning"):
        return pdf_text

    pdf_text = clean_text(pdf_text)
    sections = split_text_into_sections(pdf_text)

    # Normalize section name for better matching
    section_name_lower = section_name.lower().strip()
    
    for header, content in sections:
        if section_name_lower in header.lower():
            logging.info(f"Found section '{section_name}' matching header '{header}'")
            return content

    # If exact match not found, try partial matching
    for header, content in sections:
        if any(word in header.lower() for word in section_name_lower.split()):
            logging.info(f"Found partial match for '{section_name}' in header '{header}'")
            return content

    available_sections = [header for header, _ in sections]
    logging.warning(f"Section '{section_name}' not found. Available sections: {available_sections}")
    return f"Section '{section_name}' not found. Available sections: {', '.join(available_sections)}"

# Generate reviewer messages with type safety
reviewer_messages: List[str] = []
for reviewer in assigned_reviewers:
    message = f"""
    You are {reviewer.name}, assigned to review this paper. 
    You are {reviewer.experience_level} reviewer with {reviewer.knowledge_level} expertise.
    Your feedback tone is {reviewer.tone}.
    You have no conflict of interest.
    Your decisions may include: [Accept, Reject].
    At the end of your review, provide a final decision based on your critique.
    """
    reviewer_messages.append(message.strip())

def reviewer_agent(reviewer: Any, section_text: str, model: str, previous_feedback: Optional[str] = None) -> str:
    """LLM agent that reviews a section based on assigned reviewer attributes and provides a decision.
    
    Args:
        reviewer: Reviewer object with attributes
        section_text: Text content to review
        model: Model name to use for the review
        previous_feedback: Optional previous feedback for context
        
    Returns:
        Review content from the model
    """
    try:
        reviewer_index = assigned_reviewers.index(reviewer)
        reviewer_message = reviewer_messages[reviewer_index]
    except (ValueError, IndexError) as e:
        logging.error(f"Error finding reviewer message: {e}")
        return f"Error: Could not find reviewer configuration for {getattr(reviewer, 'name', 'unknown')}"
    
    prompt = f"""
    {reviewer_message}
    
    The section for review:
    "{section_text}"
    
    {f"Previous discussion so far: {previous_feedback}" if previous_feedback else ""}
    
    Respond in a conversational manner, directly addressing previous comments if any.
    If you agree with a previous reviewer, elaborate on why.
    If you disagree, provide justification and alternative suggestions.
    
    🔹 **At the end of your review, explicitly state your final decision (Accept, Reject).**
    """
    
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response['message']['content']
    except Exception as e:
        logging.error(f"Error calling ollama model {model}: {e}")
        return f"Error: Could not get review from model {model}: {e}"

def summarizer(section_text: str, reviews: str, model: str = "mistral") -> str:
    """Summarizes the discussion into a structured summary with a final decision.
    
    Args:
        section_text: The section content that was reviewed
        reviews: Combined review content
        model: Model to use for summarization
        
    Returns:
        Summary of the review discussion
    """
    if not section_text or not reviews:
        logging.warning("Empty section text or reviews provided for summarization")
        return "Error: Cannot summarize empty content"
        
    prompt = f"""Summarize the discussion among reviewers about the following research paper section.
    
    Section: "{section_text[:500]}{'...' if len(section_text) > 500 else ''}"

    Reviews:
    {reviews}

    Format the summary as if recording minutes of a meeting. Highlight agreements, disagreements, and key takeaways.
    
    🔹 **At the end, determine the final decision based on the majority vote (Accept, Reject).**
    """
    
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response['message']['content']
    except Exception as e:
        logging.error(f"Error calling ollama model {model} for summarization: {e}")
        return f"Error: Could not generate summary using model {model}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Extract and discuss a specific section of a research paper.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("section_name", type=str, help="Name of the section to extract and discuss")

    args = parser.parse_args()
    section_text = extract_section(args.pdf_path, args.section_name)
    if "no" in section_text:
        print(section_text)
        return

    print("\n🔍 **Extracted Section:**")
    print(section_text[:1000]) 

    print("\n📢 **Reviewers Begin Discussion:**\n")
    
    review1_output = reviewer_agent(assigned_reviewers[0], section_text)
    print(f"🗣️ **{assigned_reviewers[0].name} ({assigned_reviewers[0].experience_level} - {assigned_reviewers[0].knowledge_level})**: {review1_output}\n")
    
    review2_output = reviewer_agent(assigned_reviewers[1], section_text, review1_output)
    print(f"🗣️ **{assigned_reviewers[1].name} ({assigned_reviewers[1].experience_level} - {assigned_reviewers[1].knowledge_level})**: {review2_output}\n")
    
    review3_output = reviewer_agent(assigned_reviewers[2], section_text, f"{review1_output}\n{review2_output}")
    print(f"🗣️ **{assigned_reviewers[2].name} ({assigned_reviewers[2].experience_level} - {assigned_reviewers[2].knowledge_level})**: {review3_output}\n")

    final_summary = summarizer(section_text, f"{review1_output}\n{review2_output}\n{review3_output}")
    
    print("\n**Final Summary and Decision:**")
    print(final_summary)

if __name__ == "__main__":
    main()
