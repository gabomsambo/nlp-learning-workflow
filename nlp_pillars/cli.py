"""
Typer-based CLI for NLP Learning Workflow.

Provides commands for running the pipeline, checking status, and managing pillars.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from .config import get_settings, env_loaded_path, PILLAR_CONFIGS
from .schemas import PillarID
from .orchestrator import Orchestrator
from . import db

# Initialize Typer app and Rich console
app = typer.Typer(
    name="nlp_pillars",
    help="NLP Learning Workflow CLI - Manage your personalized learning pipeline",
    add_completion=False
)
console = Console()

# Valid pillar choices for CLI
VALID_PILLARS = ["P1", "P2", "P3", "P4", "P5"]


def _log_env_path():
    """Log the effective .env path if loaded."""
    env_path = env_loaded_path()
    if env_path:
        console.print(f"[dim]Loaded config from: {env_path}[/dim]")
    else:
        console.print("[dim]No .env file found, using environment variables[/dim]")


def _validate_pillar(pillar: str) -> PillarID:
    """Validate and convert pillar string to PillarID."""
    if pillar not in VALID_PILLARS:
        console.print(f"[red]Error: Invalid pillar '{pillar}'. Must be one of: {', '.join(VALID_PILLARS)}[/red]")
        raise typer.Exit(1)
    
    return PillarID(pillar)


def _setup_logging():
    """Setup logging based on configuration."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@app.command()
def run(
    pillar: str = typer.Option(..., "--pillar", "-p", help="Pillar to run (P1-P5)"),
    papers: int = typer.Option(1, "--papers", "-n", help="Number of papers to process", min=1)
):
    """
    Run the daily learning pipeline for a specific pillar.
    
    Executes the full workflow: discovery, search, ingest, summarize, synthesize, and quiz generation.
    """
    _log_env_path()
    _setup_logging()
    
    try:
        # Validate pillar
        pillar_id = _validate_pillar(pillar)
        
        console.print(f"\n[bold blue]Running pipeline for {pillar} - {PILLAR_CONFIGS[pillar]['name']}[/bold blue]")
        console.print(f"Target papers: {papers}")
        
        # Initialize and run orchestrator
        with console.status("[bold green]Processing papers..."):
            orchestrator = Orchestrator(enable_quiz=True)
            result = orchestrator.run_daily(pillar_id, papers_limit=papers)
        
        # Create summary panel
        summary_lines = [
            f"[bold]Pillar:[/bold] {pillar} - {PILLAR_CONFIGS[pillar]['name']}",
            f"[bold]Papers processed:[/bold] {len(result.papers_processed)}",
            f"[bold]Lessons created:[/bold] {len(result.lessons_created)}",
            f"[bold]Quiz cards generated:[/bold] {len(result.quizzes_generated)}",
            f"[bold]Errors:[/bold] {len(result.errors)}",
            f"[bold]Total time:[/bold] {result.total_time_seconds:.2f}s"
        ]
        
        # Add error details if any
        if result.errors:
            summary_lines.append("\n[bold red]Errors:[/bold red]")
            for error in result.errors[:3]:  # Show first 3 errors
                summary_lines.append(f"  • {error.get('paper_id', 'unknown')}: {error.get('message', 'Unknown error')}")
            if len(result.errors) > 3:
                summary_lines.append(f"  ... and {len(result.errors) - 3} more")
        
        panel = Panel(
            "\n".join(summary_lines),
            title="🎯 Pipeline Results",
            border_style="green" if result.success else "red"
        )
        
        console.print("\n")
        console.print(panel)
        
        # Exit with appropriate code
        if result.success:
            console.print("\n[bold green]✅ Pipeline completed successfully![/bold green]")
            raise typer.Exit(0)
        else:
            console.print("\n[bold red]❌ Pipeline failed or no papers processed[/bold red]")
            raise typer.Exit(1)
            
    except typer.Exit:
        raise  # Re-raise typer.Exit without logging
    except Exception as e:
        console.print(f"\n[bold red]❌ Pipeline error: {str(e)}[/bold red]")
        logging.error(f"Pipeline error: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def status(
    pillar: str = typer.Option(..., "--pillar", "-p", help="Pillar to check (P1-P5)")
):
    """
    Show status for a specific pillar: recent lessons and queue size.
    
    Displays the last 3 lessons and the number of papers in the processing queue.
    """
    _log_env_path()
    _setup_logging()
    
    try:
        # Validate pillar
        pillar_id = _validate_pillar(pillar)
        
        console.print(f"\n[bold blue]Status for {pillar} - {PILLAR_CONFIGS[pillar]['name']}[/bold blue]")
        
        # Get recent lessons
        try:
            recent_lessons = db.get_recent_notes(pillar_id, limit=3)
        except Exception as e:
            console.print(f"[red]Error fetching lessons: {e}[/red]")
            raise typer.Exit(1)
        
        # Get queue size
        try:
            # Count unprocessed papers in queue for this pillar
            import os
            supabase_client = db.get_client()
            if supabase_client:
                result = supabase_client.table("paper_queue").select("id", count="exact").eq("pillar_id", pillar_id.value).eq("processed", False).execute()
                queue_size = result.count or 0
            else:
                queue_size = 0
        except Exception as e:
            console.print(f"[red]Error fetching queue size: {e}[/red]")
            queue_size = 0
        
        # Create lessons table
        if recent_lessons:
            table = Table(title="Recent Lessons", show_header=True, header_style="bold magenta")
            table.add_column("Created", style="dim", width=19)
            table.add_column("Paper ID", style="cyan", width=20)
            table.add_column("TL;DR", style="white")
            
            for lesson in recent_lessons:
                # Get lesson from recent notes (PaperNote objects)
                created_str = datetime.now().strftime("%Y-%m-%d %H:%M")  # Placeholder since PaperNote doesn't have created_at
                paper_id = lesson.paper_id[:20] + "..." if len(lesson.paper_id) > 20 else lesson.paper_id
                
                # Create a summary from the problem/method fields
                tl_dr = f"{lesson.problem[:60]}..." if len(lesson.problem) > 60 else lesson.problem
                
                table.add_row(created_str, paper_id, tl_dr)
            
            console.print("\n")
            console.print(table)
        else:
            console.print("\n[dim]No recent lessons found[/dim]")
        
        # Show queue size
        console.print(f"\n[bold]Queue:[/bold] {queue_size} papers pending")
        
        raise typer.Exit(0)
        
    except typer.Exit:
        raise  # Re-raise typer.Exit without logging
    except Exception as e:
        console.print(f"\n[bold red]❌ Status error: {str(e)}[/bold red]")
        logging.error(f"Status error: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def review(
    pillar: str = typer.Option(..., "--pillar", "-p", help="Pillar to review (P1-P5)")
):
    """
    Show review status for a specific pillar: due quiz cards count.
    
    Displays the number of quiz cards that are due for review today.
    """
    _log_env_path()
    _setup_logging()
    
    try:
        # Validate pillar
        pillar_id = _validate_pillar(pillar)
        
        console.print(f"\n[bold blue]Review for {pillar} - {PILLAR_CONFIGS[pillar]['name']}[/bold blue]")
        
        # Get due quiz cards count
        try:
            # Count quiz cards due today
            supabase_client = db.get_client()
            if supabase_client:
                today = datetime.now().date().isoformat()
                result = supabase_client.table("quiz_cards").select("id", count="exact").eq("pillar_id", pillar_id.value).lte("due_date", today).execute()
                due_count = result.count or 0
            else:
                due_count = 0
        except Exception as e:
            console.print(f"[red]Error fetching due quiz cards: {e}[/red]")
            raise typer.Exit(1)
        
        # Display result
        if due_count > 0:
            console.print(f"\n[bold yellow]Due today: {due_count}[/bold yellow] 📚")
        else:
            console.print(f"\n[bold green]Due today: {due_count}[/bold green] ✨")
        
        raise typer.Exit(0)
        
    except typer.Exit:
        raise  # Re-raise typer.Exit without logging
    except Exception as e:
        console.print(f"\n[bold red]❌ Review error: {str(e)}[/bold red]")
        logging.error(f"Review error: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command("pillars")
def pillars_list():
    """
    List all available learning pillars.
    
    Shows the five core pillars with their names and goals.
    """
    _log_env_path()
    
    console.print("\n[bold blue]Available Learning Pillars[/bold blue]")
    
    # Create pillars table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", width=4)
    table.add_column("Name", style="white")
    table.add_column("Goal", style="dim")
    
    for pillar_id, config in PILLAR_CONFIGS.items():
        table.add_row(
            pillar_id,
            config["name"],
            config["goal"][:80] + "..." if len(config["goal"]) > 80 else config["goal"]
        )
    
    console.print("\n")
    console.print(table)
    console.print("\n[dim]Use --pillar <ID> with other commands to target a specific pillar[/dim]")
    
    raise typer.Exit(0)


if __name__ == "__main__":
    app()