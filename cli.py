#!/usr/bin/env python3
"""
MARS CLI - Command Line Interface for Multi-Agent Review System
Modern CLI with click and rich for better user experience
"""

import click
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from MARS import main as mars_main, create_argument_parser

console = Console()

@click.command()
@click.argument('cfp_url', type=str, help='URL to the Conference Call for Papers (CFP)')
@click.argument('paper_path', type=click.Path(exists=True), help='Path to the PDF file or JSON file')
@click.argument('section_name', required=False, default='', help='Optional: specific paper section for review')
@click.option('--answer-questions', '-q', is_flag=True, help='Enable answering questions in the second stage')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging output')
@click.option('--output', '-o', type=click.Path(), help='Output directory for results')
@click.version_option(version='1.0.0', prog_name='MARS')
def cli(cfp_url: str, paper_path: str, section_name: str, answer_questions: bool, verbose: bool, output: str):
    """
    🚀 MARS: Multi-Agent Review System for Academic Papers
    
    Analyze academic papers using multiple AI agents for comprehensive review.
    
    Examples:
    
        mars https://example.com/cfp paper.pdf
        
        mars https://example.com/cfp paper.pdf --answer-questions
        
        mars https://example.com/cfp paper.json "Introduction" --verbose
    """
    
    # Show welcome banner
    console.print(Panel.fit(
        "🚀 [bold blue]MARS: Multi-Agent Review System[/bold blue] 🚀\n"
        "Academic Paper Review with AI Agents\n"
        f"Version 1.0.0",
        border_style="blue"
    ))
    
    # Validate inputs
    paper_path_obj = Path(paper_path)
    if not paper_path_obj.exists():
        console.print(f"[red]❌ Error: File not found: {paper_path}[/red]")
        raise click.Abort()
    
    if not paper_path.endswith(('.pdf', '.json')):
        console.print(f"[yellow]⚠️ Warning: Expected .pdf or .json file, got: {paper_path}[/yellow]")
    
    # Set up output directory
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(output_dir)
        console.print(f"📁 Output directory: {output_dir.absolute()}")
    
    # Prepare arguments for MARS main function
    import argparse
    args = argparse.Namespace(
        url=cfp_url,
        pdf_path=str(paper_path_obj.absolute()),
        section_name=section_name,
        answer_questions=answer_questions,
        verbose=verbose
    )
    
    # Temporarily replace sys.argv to work with existing argument parser
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            'MARS.py',
            cfp_url,
            str(paper_path_obj.absolute())
        ]
        if section_name:
            sys.argv.append(section_name)
        if answer_questions:
            sys.argv.append('--answer-questions')
        if verbose:
            sys.argv.append('--verbose')
        
        # Run MARS
        mars_main()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Process interrupted by user[/yellow]")
        raise click.Abort()
    except Exception as e:
        console.print(f"[red]💥 Error: {e}[/red]")
        if verbose:
            import traceback
            console.print(f"[red]{traceback.format_exc()}[/red]")
        raise click.Abort()
    finally:
        sys.argv = original_argv

@click.group()
def main():
    """MARS CLI - Multi-Agent Review System"""
    pass

@main.command()
def setup():
    """Run the setup process for MARS"""
    console.print("🔧 Running MARS setup...")
    import subprocess
    
    setup_script = Path(__file__).parent / "setup.sh"
    if setup_script.exists():
        result = subprocess.run([str(setup_script)], capture_output=True, text=True)
        console.print(result.stdout)
        if result.stderr:
            console.print(f"[yellow]{result.stderr}[/yellow]")
        if result.returncode != 0:
            console.print(f"[red]❌ Setup failed with exit code {result.returncode}[/red]")
            raise click.Abort()
    else:
        console.print(f"[red]❌ Setup script not found: {setup_script}[/red]")
        raise click.Abort()

@main.command()
@click.option('--format', type=click.Choice(['json', 'yaml', 'txt']), default='json', help='Output format')
def status():
    """Check MARS system status"""
    console.print("🔍 Checking MARS system status...")
    
    # Check Python version
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    console.print(f"🐍 Python: {python_version}")
    
    # Check dependencies
    deps_status = {}
    required_deps = ['ollama', 'transformers', 'pypdf', 'rich', 'loguru']
    
    for dep in required_deps:
        try:
            __import__(dep)
            deps_status[dep] = "✅ OK"
        except ImportError:
            deps_status[dep] = "❌ Missing"
    
    console.print("📦 Dependencies:")
    for dep, status in deps_status.items():
        console.print(f"  {dep}: {status}")
    
    # Check Ollama
    try:
        import ollama
        models = ollama.list()
        console.print(f"🤖 Ollama: ✅ OK ({len(models.models)} models available)")
    except Exception as e:
        console.print(f"🤖 Ollama: ❌ Error - {e}")

# Make CLI the default when run as script
if __name__ == '__main__':
    main()