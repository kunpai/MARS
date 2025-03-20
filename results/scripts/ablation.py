import os
import json
import glob
import re
import csv
from collections import defaultdict
import argparse
import copy

# Default parameters that can be ablated, now with an "active_reviewers" parameter.
DEFAULT_PARAMS = {
    "decision_scores": {"Accept": 100, "Reject": 0},
    "section_weight": 1,
    "accept_threshold": 50,
    "fallback_to_reject": True,
    "use_weighted_avg": True,
    # Components to include/exclude from review evaluation
    "include_components": {
        "Test": True,             # Include summary and test descriptions
        "Reviewers": True,        # Include reviewer comments (now with further control via active_reviewers)
        "Grammar Check": True,    # Include grammar assessment
        "Novelty Check": True,    # Include novelty assessment
        "Fact Check": True,       # Include fact-checking assessment
        "Questioner": False,      # Don't include questions by default
        "Final Summary": True     # Include final summary
    },
    # Specify which reviewer models to include by default.
    "active_reviewers": ["mistral", "llama3.2", "qwen2.5", "deepseek-r1"]
}

def extract_decision(review_obj, fallback_to_reject=True):
    """
    Extracts 'Accept' or 'Reject' decision from a review object.
    If "Accept" is **not found**, defaults to "Reject" if fallback_to_reject is True.
    """
    if isinstance(review_obj, dict):
        if "Accept" in review_obj:
            value = review_obj["Accept"]
            if isinstance(value, bool):
                return "Accept" if value else "Reject"

        for key, text in review_obj.items():
            if isinstance(text, str):
                match = re.search(r'\b(accept|reject)\b', text, re.IGNORECASE)
                if match:
                    return match.group(1).capitalize()

    elif isinstance(review_obj, str):
        match = re.search(r'\b(accept|reject)\b', review_obj, re.IGNORECASE)
        if match:
            return match.group(1).capitalize()

    return "Reject" if fallback_to_reject else None

def extract_score(review_obj, decision_scores=None, fallback_to_reject=True, active_reviewers=None):
    """
    Extracts a numerical score (0-100) based on Accept/Reject or numeric values.
    If active_reviewers is provided, only include reviews from those reviewer keys.
    """
    if decision_scores is None:
        decision_scores = DEFAULT_PARAMS["decision_scores"]

    if isinstance(review_obj, dict) and "Reviewers" in review_obj:
        reviewer_scores = []
        
        for reviewer, review in review_obj["Reviewers"].items():
            # If an active_reviewers list is provided, skip keys not in it
            if active_reviewers is not None and reviewer not in active_reviewers:
                continue
            decision = extract_decision(review, fallback_to_reject)
            if decision:
                reviewer_scores.append(decision_scores.get(decision, 0))
        
        if reviewer_scores:
            return sum(reviewer_scores) / len(reviewer_scores)

    if isinstance(review_obj, dict):
        for key, value in review_obj.items():
            if isinstance(value, (int, float)) and 0 <= value <= 100:
                return value

    decision = extract_decision(review_obj, fallback_to_reject)
    return decision_scores.get(decision, 0) if decision else 0

def determine_verdict(score, accept_threshold=50):
    """Determines the final verdict based on average section score."""
    if score >= accept_threshold:
        return "Accept"
    else:
        return "Reject"

def filter_review_components(section_data, include_components):
    """Filter out components from section data based on include_components dictionary."""
    if not isinstance(section_data, dict):
        return section_data
    
    filtered_data = {}
    
    for key, value in section_data.items():
        # If key is in include_components and set to True, include it
        if key in include_components:
            if include_components[key]:
                filtered_data[key] = value
        # For any other keys not explicitly mentioned, include them
        else:
            filtered_data[key] = value
            
    return filtered_data

def process_json_file(file_path, paper_decisions, params):
    """Processes a JSON file, extracts section scores, and computes final average-based verdict."""
    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error parsing {file_path}: {e}")
            return

    if "Section Reviews" not in data:
        return

    # Make a copy of the original section reviews
    section_reviews = copy.deepcopy(data["Section Reviews"])
    
    # Filter out components based on include_components parameter
    for section, review_data in section_reviews.items():
        section_reviews[section] = filter_review_components(review_data, params["include_components"])
    
    section_scores = {}
    weighted_total = 0
    total_weight = 0
    num_sections = 0
    calculations = []

    for section, review_data in section_reviews.items():
        score = extract_score(
            review_data,
            decision_scores=params["decision_scores"],
            fallback_to_reject=params["fallback_to_reject"],
            active_reviewers=params.get("active_reviewers")  # Pass the active_reviewers parameter here
        )
        section_scores[section] = score

        if params["use_weighted_avg"]:
            weight = params["section_weight"]
            weighted_total += score * weight
            total_weight += weight
            calculations.append(f"{score} × {weight} = {score * weight:.2f}")
        else:
            weighted_total += score
            num_sections += 1
            calculations.append(f"{score}")

    # Calculate final score
    if params["use_weighted_avg"]:
        final_score = weighted_total / total_weight if total_weight > 0 else 0
    else:
        final_score = weighted_total / num_sections if num_sections > 0 else 0

    final_paper_verdict = determine_verdict(final_score, params["accept_threshold"])
    
    if params.get("verbose"):
        print(f"\nDEBUG: Processing file: {file_path}")
        for section, score in section_scores.items():
            print(f"  Section '{section}': Score = {score:.2f}")
        
        if params["use_weighted_avg"]:
            print(f"  Weighted Contributions: {' + '.join(calculations)}")
        else:
            print(f"  Unweighted Scores: {', '.join(calculations)}")
            
        print(f"  **Final Score: {final_score:.2f}**")
        print(f"  **Final Verdict: {final_paper_verdict}**\n")

    paper_decisions[file_path] = {
        "Section Scores": section_scores,
        "Final Score": final_score,
        "Final Decision": final_paper_verdict
    }

def run_ablation_study(directory, ablation_configs):
    """Run ablation study with different parameter combinations"""
    results = {}
    
    for config_name, params in ablation_configs.items():
        print(f"\n--- Running configuration: {config_name} ---")
        
        # Use default parameters and update with the specific ablation config
        run_params = copy.deepcopy(DEFAULT_PARAMS)
        for key, value in params.items():
            if key == "include_components" and isinstance(value, dict):
                # Merge include_components dict instead of overwriting
                run_params["include_components"].update(value)
            else:
                run_params[key] = value
        
        run_params["verbose"] = True  # Add verbosity flag
        
        paper_decisions = {}
        for file in glob.glob(os.path.join(directory, "*.json")):
            process_json_file(file, paper_decisions, run_params)
        
        # Store results for this configuration
        results[config_name] = {
            "params": run_params,
            "decisions": paper_decisions
        }
    
    # Generate summary of ablation study
    print("\n=== ABLATION STUDY RESULTS ===")
    
    # Create a table for comparison
    paper_files = []
    for config_results in results.values():
        paper_files.extend(list(config_results["decisions"].keys()))
    paper_files = list(set(paper_files))  # Get unique files
    paper_names = [os.path.basename(f) for f in paper_files]
    
    # Write results to CSV
    with open("ablation_results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ["Paper"] + list(ablation_configs.keys())
        writer.writerow(header)
        
        # Write decision for each paper under each configuration
        for paper_file, paper_name in zip(paper_files, paper_names):
            row = [paper_name]
            for config in ablation_configs.keys():
                if paper_file in results[config]["decisions"]:
                    row.append(results[config]["decisions"][paper_file]["Final Decision"])
                else:
                    row.append("N/A")
            writer.writerow(row)
            
        # Add a row for summary statistics
        writer.writerow([])
        accept_counts = []
        for config in ablation_configs.keys():
            count = sum(1 for paper_file in paper_files 
                      if paper_file in results[config]["decisions"] 
                      and results[config]["decisions"][paper_file]["Final Decision"] == "Accept")
            accept_counts.append(count)
        
        total_papers = len(paper_files)
        writer.writerow(["Accept Count"] + accept_counts)
        writer.writerow(["Accept %"] + [f"{count / total_papers * 100:.1f}%" for count in accept_counts])
    
    print(f"Results saved to ablation_results.csv")
    return results

def main():
    parser = argparse.ArgumentParser(description="Process paper reviews with ablation study")
    parser.add_argument("directory", help="Directory containing JSON review files")
    parser.add_argument("--baseline-only", action="store_true", help="Run only baseline configuration without ablation")
    parser.add_argument("--verbose", action="store_true", help="Print detailed processing information")
    args = parser.parse_args()
    
    # Define ablation configurations, including ones for reviewer models
    ablation_configs = {
        "baseline": {},  # Uses all default parameters
        
        "higher_threshold": {
            "accept_threshold": 70  # Stricter acceptance threshold
        },
        
        "lower_threshold": {
            "accept_threshold": 30  # More lenient acceptance threshold
        },
        
        "reviewers_only": {
            "include_components": {
                "Test": False,
                "Grammar Check": False,
                "Novelty Check": False,
                "Fact Check": False,
                "Final Summary": False
            }
        },
        
        "no_reviewers": {
            "include_components": {
                "Reviewers": False
            }
        },
        
        "test_and_summary_only": {
            "include_components": {
                "Reviewers": False,
                "Grammar Check": False,
                "Novelty Check": False,
                "Fact Check": False
            }
        },
        
        "fact_and_novelty_only": {
            "include_components": {
                "Test": False,
                "Reviewers": False,
                "Grammar Check": False,
                "Final Summary": False
            }
        },
        
        "grammar_only": {
            "include_components": {
                "Test": False, 
                "Reviewers": False,
                "Novelty Check": False,
                "Fact Check": False,
                "Final Summary": False
            }
        },
        
        "include_questioner": {
            "include_components": {
                "Questioner": True  # Usually excluded by default
            }
        },
        
        "different_scores": {
            "decision_scores": {"Accept": 80, "Reject": 20}  # Less extreme scoring
        },
        
        "no_fallback": {
            "fallback_to_reject": False  # Don't default to Reject
        },
        
        "unweighted_avg": {
            "use_weighted_avg": False  # Use simple average instead of weighted
        },
        # --- New ablation configurations for reviewer models ---
        "all_reviewers": {
            "include_components": {
                "Test": False,
                "Grammar Check": False,
                "Novelty Check": False,
                "Fact Check": False,
                "Final Summary": False
            },
            "active_reviewers": ["mistral", "llama3.2", "qwen2.5", "deepseek-r1"]
        },
        "no_mistral": {
            "include_components": {
                "Test": False,
                "Grammar Check": False,
                "Novelty Check": False,
                "Fact Check": False,
                "Final Summary": False
            },
            "active_reviewers": ["llama3.2", "qwen2.5", "deepseek-r1"]
        },
        "only_mistral": {
            "include_components": {
                "Test": False,
                "Grammar Check": False,
                "Novelty Check": False,
                "Fact Check": False,
                "Final Summary": False
            },
            "active_reviewers": ["mistral"]
        },
        "only_llama3.2": {
            "include_components": {
                "Test": False,
                "Grammar Check": False,
                "Novelty Check": False,
                "Fact Check": False,
                "Final Summary": False
            },
            "active_reviewers": ["llama3.2"]
        },
        "only_qwen2.5": {
            "include_components": {
                "Test": False,
                "Grammar Check": False,
                "Novelty Check": False,
                "Fact Check": False,
                "Final Summary": False
            },
            "active_reviewers": ["qwen2.5"]
        },
        "only_deepseek-r1": {
            "include_components": {
                "Test": False,
                "Grammar Check": False,
                "Novelty Check": False,
                "Fact Check": False,
                "Final Summary": False
            },
            "active_reviewers": ["deepseek-r1"]
        },
        "only_fact_check": {
            "include_components": {
                "Test": False,
                "Reviewers": False,
                "Grammar Check": False,
                "Novelty Check": False,
                "Final Summary": False
            }
        },
        "only_novelty_check": {
            "include_components": {
                "Test": False,
                "Reviewers": False,
                "Grammar Check": False,
                "Fact Check": False,
                "Final Summary": False
            }
        },
        "only_summary": {
            "include_components": {
                "Test": False,
                "Reviewers": False,
                "Grammar Check": False,
                "Novelty Check": False,
                "Fact Check": False
            }
        }
    }
    
    if args.baseline_only:
        ablation_configs = {"baseline": {}}
    
    # Add verbosity to all configs if needed
    for config in ablation_configs.values():
        config["verbose"] = args.verbose
    
    # Run the ablation study
    results = run_ablation_study(args.directory, ablation_configs)
    
    # Print overall summary
    print("\nSummary of Final Decisions by Configuration:")
    for config_name, config_results in results.items():
        accept_count = sum(1 for decision in config_results["decisions"].values() 
                         if decision["Final Decision"] == "Accept")
        total_papers = len(config_results["decisions"])
        print(f"{config_name}: {accept_count}/{total_papers} accepted ({accept_count/total_papers*100:.1f}%)")

if __name__ == "__main__":
    main()
