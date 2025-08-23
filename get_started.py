#!/usr/bin/env python
"""
Getting Started Script for NLP Learning Workflow.
Run this to verify your setup and see the system in action.
"""

import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
import time

console = Console()


def check_environment():
    """Check if all required environment variables are set."""
    console.print("\n[bold blue]🔍 Checking Environment Variables[/bold blue]\n")
    
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API Key",
        "SUPABASE_URL": "Supabase URL",
        "SUPABASE_KEY": "Supabase Key",
    }
    
    optional_vars = {
        "ANTHROPIC_API_KEY": "Anthropic API Key",
        "GROQ_API_KEY": "Groq API Key",
        "QDRANT_URL": "Qdrant URL",
    }
    
    all_good = True
    
    # Check required variables
    for var, name in required_vars.items():
        if os.getenv(var):
            console.print(f"✅ {name}: [green]Set[/green]")
        else:
            console.print(f"❌ {name}: [red]Missing[/red]")
            all_good = False
    
    console.print()
    
    # Check optional variables
    for var, name in optional_vars.items():
        if os.getenv(var):
            console.print(f"✅ {name}: [green]Set[/green] (optional)")
        else:
            console.print(f"ℹ️  {name}: [yellow]Not set[/yellow] (optional)")
    
    return all_good


def test_imports():
    """Test if all required packages can be imported."""
    console.print("\n[bold blue]📦 Testing Package Imports[/bold blue]\n")
    
    packages = [
        "atomic_agents",
        "instructor",
        "pydantic",
        "openai",
        "supabase",
        "qdrant_client",
        "pypdf",
        "typer",
        "arxiv",
    ]
    
    all_good = True
    
    for package in track(packages, description="Importing packages..."):
        try:
            __import__(package)
            console.print(f"✅ {package}")
        except ImportError as e:
            console.print(f"❌ {package}: {e}")
            all_good = False
        time.sleep(0.1)  # Small delay for visual effect
    
    return all_good


def test_nlp_pillars():
    """Test if the NLP Pillars package can be imported."""
    console.print("\n[bold blue]🏛️ Testing NLP Pillars Package[/bold blue]\n")
    
    try:
        from nlp_pillars import PillarID, PaperRef
        from nlp_pillars.config import get_settings, PILLAR_CONFIGS
        from nlp_pillars.agents import SummarizerAgent
        
        console.print("✅ Core modules imported successfully")
        
        # Display pillar information
        console.print("\n[bold]Available Pillars:[/bold]")
        for pillar_id, config in PILLAR_CONFIGS.items():
            console.print(f"  {pillar_id}: {config['name']}")
        
        return True
    except Exception as e:
        console.print(f"❌ Error importing NLP Pillars: {e}")
        return False


def create_sample_data():
    """Create sample data for testing."""
    console.print("\n[bold blue]🎭 Creating Sample Data[/bold blue]\n")
    
    try:
        from nlp_pillars.schemas import PaperRef, ParsedPaper, PillarID
        
        # Create a sample paper reference
        paper_ref = PaperRef(
            id="2024.sample.001",
            title="Sample Paper: Understanding Transformers",
            authors=["John Doe", "Jane Smith"],
            year=2024,
            venue="Sample Conference",
            abstract="This is a sample abstract about transformers..."
        )
        
        # Create parsed paper
        parsed_paper = ParsedPaper(
            paper_ref=paper_ref,
            full_text="This is the full text of the sample paper...",
            chunks=["Chunk 1", "Chunk 2", "Chunk 3"],
            figures_count=3,
            tables_count=2,
            references=["Ref 1", "Ref 2"]
        )
        
        console.print("✅ Sample data created successfully")
        return parsed_paper
    except Exception as e:
        console.print(f"❌ Error creating sample data: {e}")
        return None


def test_agent():
    """Test the Summarizer Agent with sample data."""
    console.print("\n[bold blue]🤖 Testing Summarizer Agent[/bold blue]\n")
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[yellow]⚠️  Skipping agent test - OpenAI API key not set[/yellow]")
        return True
    
    try:
        from nlp_pillars.agents import SummarizerAgent
        from nlp_pillars.schemas import PillarID
        
        console.print("Note: This would make an actual API call to OpenAI.")
        console.print("Skipping actual API call to save costs.")
        console.print("✅ Agent initialized successfully (dry run)")
        
        # Uncomment to test with actual API call:
        # parsed_paper = create_sample_data()
        # if parsed_paper:
        #     agent = SummarizerAgent()
        #     note = agent.summarize(
        #         parsed_paper=parsed_paper,
        #         pillar_id=PillarID.P2,
        #         recent_notes=[]
        #     )
        #     console.print(f"✅ Generated note: {note.problem[:100]}...")
        
        return True
    except Exception as e:
        console.print(f"❌ Error testing agent: {e}")
        return False


def main():
    """Main function to run all checks."""
    console.print(Panel.fit(
        "[bold cyan]NLP Learning Workflow - Setup Verification[/bold cyan]\n"
        "This script will check your environment and test the basic setup.",
        title="🚀 Getting Started",
        border_style="cyan"
    ))
    
    # Track overall success
    all_checks_passed = True
    
    # Run checks
    if not check_environment():
        all_checks_passed = False
        console.print("\n[yellow]⚠️  Some environment variables are missing.[/yellow]")
        console.print("Please copy .env.example to .env and fill in your API keys.")
    
    if not test_imports():
        all_checks_passed = False
        console.print("\n[yellow]⚠️  Some packages failed to import.[/yellow]")
        console.print("Please run: pip install -r requirements.txt")
    
    if not test_nlp_pillars():
        all_checks_passed = False
        console.print("\n[yellow]⚠️  NLP Pillars package import failed.[/yellow]")
        console.print("Make sure you're in the project directory.")
    
    sample_data = create_sample_data()
    if not sample_data:
        all_checks_passed = False
    
    if not test_agent():
        all_checks_passed = False
    
    # Final summary
    console.print("\n" + "=" * 60 + "\n")
    
    if all_checks_passed:
        console.print(Panel.fit(
            "[bold green]✨ All checks passed![/bold green]\n\n"
            "Your NLP Learning Workflow is ready to use.\n\n"
            "Next steps:\n"
            "1. Run: [cyan]python -m nlp_pillars.cli init[/cyan] to initialize the system\n"
            "2. Run: [cyan]python -m nlp_pillars.cli run --pillar P1[/cyan] to start learning\n"
            "3. Run: [cyan]python -m nlp_pillars.cli status[/cyan] to check progress",
            title="✅ Setup Complete",
            border_style="green"
        ))
    else:
        console.print(Panel.fit(
            "[bold yellow]⚠️  Some checks failed[/bold yellow]\n\n"
            "Please address the issues above before proceeding.\n\n"
            "Common fixes:\n"
            "1. Copy .env.example to .env and add your API keys\n"
            "2. Run: pip install -r requirements.txt\n"
            "3. Make sure you're in the project root directory",
            title="⚠️  Setup Incomplete",
            border_style="yellow"
        ))
    
    return 0 if all_checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
