"""Typer-based CLI for NLP Learning Workflow.

Provides commands for running the pipeline, checking status, and managing pillars.
"""

import logging
from datetime import datetime
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import db
from .config import PILLAR_CONFIGS, env_loaded_path, get_pillar_config, get_settings
from .orchestrator import Orchestrator

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize Typer app and Rich console
app = typer.Typer(
    name="nlp_pillars",
    help="NLP Learning Workflow CLI - Manage your personalized learning pipeline",
    add_completion=False
)
console = Console()

# Get valid pillars from database or fallback to static config
def get_valid_pillars():
    """Get list of valid pillar IDs from database."""
    try:
        pillars = db.get_pillars(limit=50)
        return [p.id for p in pillars]
    except Exception:
        # Fallback to static config keys
        return list(PILLAR_CONFIGS.keys())


def _log_env_path():
    """Log the effective .env path if loaded."""
    env_path = env_loaded_path()
    if env_path:
        console.print(f"[dim]Loaded config from: {env_path}[/dim]")
    else:
        console.print("[dim]No .env file found, using environment variables[/dim]")


def _validate_pillar(pillar: str) -> str:
    """Validate pillar ID exists in database."""
    valid_pillars = get_valid_pillars()
    if pillar not in valid_pillars:
        console.print(f"[red]Error: Invalid pillar '{pillar}'.[/red]")
        console.print(f"[yellow]Valid pillars: {', '.join(valid_pillars)}[/yellow]")
        raise typer.Exit(1)

    return pillar


def _setup_logging():
    """Setup logging based on configuration."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@app.command()
def run(
    pillar: str = typer.Option(..., "--pillar", "-p", help="Pillar to run (P1-P5)"),
    papers: int = typer.Option(1, "--papers", "-n", help="Number of papers to process", min=1)
):
    """Run the daily learning pipeline for a specific pillar.
    
    Executes the full workflow: discovery, search, ingest, summarize, synthesize, and quiz generation.
    """
    _log_env_path()
    _setup_logging()

    try:
        # Validate pillar
        pillar_id = _validate_pillar(pillar)
        pillar_config = get_pillar_config(pillar_id)

        console.print(f"\n[bold blue]Running pipeline for {pillar_id} - {pillar_config['name']}[/bold blue]")
        console.print(f"Target papers: {papers}")

        # Initialize and run orchestrator
        with console.status("[bold green]Processing papers..."):
            orchestrator = Orchestrator(enable_quiz=True)
            result = orchestrator.run_daily(pillar_id, papers_limit=papers)

        # Create summary panel
        summary_lines = [
            f"[bold]Pillar:[/bold] {pillar_id} - {pillar_config['name']}",
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
    pillar: Optional[str] = typer.Option(None, "--pillar", "-p", help="Pillar to check (P1-P5). If not provided, shows all pillars")
):
    """Show status: recent activity and queue sizes.
    
    Without --pillar: Shows overview of all pillars with queue sizes and recent activity.
    With --pillar: Shows detailed status for that specific pillar.
    """
    _log_env_path()
    _setup_logging()

    try:
        if pillar:
            # Show detailed status for specific pillar
            pillar_id = _validate_pillar(pillar)
            _show_pillar_status(pillar_id)
        else:
            # Show overview of all pillars
            _show_global_status()

        raise typer.Exit(0)

    except typer.Exit:
        raise  # Re-raise typer.Exit without logging
    except Exception as e:
        console.print(f"\n[bold red]❌ Status error: {str(e)}[/bold red]")
        logging.error(f"Status error: {e}", exc_info=True)
        raise typer.Exit(1)


def _show_global_status():
    """Show overview status for all pillars."""
    console.print("\n[bold blue]📊 NLP Learning Workflow Status[/bold blue]")

    # Create status table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Pillar", style="cyan", width=6)
    table.add_column("Name", style="white", width=25)
    table.add_column("Queue", style="yellow", width=8)
    table.add_column("Recent Papers", style="green", width=12)
    table.add_column("Recent Lessons", style="blue", width=12)
    table.add_column("Due Quizzes", style="red", width=10)

    total_queue = 0
    total_papers = 0
    total_lessons = 0
    total_quizzes = 0

    for pillar_name, config in PILLAR_CONFIGS.items():
        pillar_id = _validate_pillar(pillar_name)

        try:
            # Get queue size
            queue_size = db.get_queue_size(pillar_id)
            total_queue += queue_size

            # Get recent papers count (last 7 days)
            papers = db.get_papers(pillar_id, limit=50)  # Get more to count recent ones
            recent_papers = len([p for p in papers if True])  # For now, count all as we don't have date filtering
            total_papers += min(recent_papers, 10)  # Cap display at 10

            # Get recent lessons count
            lessons = db.get_lessons(pillar_id, limit=10)
            recent_lessons = len(lessons)
            total_lessons += recent_lessons

            # Get due quiz cards count
            quiz_cards = db.get_quiz_cards_for_review(pillar_id, limit=100)
            due_quizzes = len(quiz_cards)
            total_quizzes += due_quizzes

            # Format queue size with color coding
            queue_str = str(queue_size)
            if queue_size > 10:
                queue_str = f"[red]{queue_size}[/red]"
            elif queue_size > 5:
                queue_str = f"[yellow]{queue_size}[/yellow]"
            else:
                queue_str = f"[green]{queue_size}[/green]"

            # Add row to table
            table.add_row(
                pillar_name,
                config["name"][:25],
                queue_str,
                str(min(recent_papers, 10)),
                str(recent_lessons),
                str(due_quizzes)
            )

        except Exception as e:
            logger.error(f"Error getting status for pillar {pillar_name}: {e}")
            table.add_row(
                pillar_name,
                config["name"][:25],
                "[red]Error[/red]",
                "[red]Error[/red]",
                "[red]Error[/red]",
                "[red]Error[/red]"
            )

    console.print("\n")
    console.print(table)

    # Show totals
    console.print(f"\n[bold]📈 Totals:[/bold] {total_queue} queued • {total_papers} recent papers • {total_lessons} lessons • {total_quizzes} due quizzes")
    console.print("\n[dim]💡 Use --pillar <ID> for detailed pillar status[/dim]")


def _show_pillar_status(pillar_id: str):
    """Show detailed status for a specific pillar."""
    pillar_config = get_pillar_config(pillar_id)
    console.print(f"\n[bold blue]Status for {pillar_id} - {pillar_config['name']}[/bold blue]")

    # Get recent lessons
    try:
        recent_lessons = db.get_recent_notes(pillar_id, limit=3)
    except Exception as e:
        console.print(f"[red]Error fetching lessons: {e}[/red]")
        raise typer.Exit(1)

    # Get queue size
    try:
        queue_size = db.get_queue_size(pillar_id)
    except Exception as e:
        console.print(f"[red]Error fetching queue size: {e}[/red]")
        queue_size = 0

    # Create lessons table
    if recent_lessons:
        table = Table(title="Recent Notes", show_header=True, header_style="bold magenta")
        table.add_column("Paper ID", style="cyan", width=20)
        table.add_column("Problem", style="white")
        table.add_column("Method", style="blue")

        for note in recent_lessons:
            paper_id = note.paper_id[:20] + "..." if len(note.paper_id) > 20 else note.paper_id
            problem = f"{note.problem[:40]}..." if len(note.problem) > 40 else note.problem
            method = f"{note.method[:30]}..." if len(note.method) > 30 else note.method

            table.add_row(paper_id, problem, method)

        console.print("\n")
        console.print(table)
    else:
        console.print("\n[dim]No recent notes found[/dim]")

    # Show queue size
    console.print(f"\n[bold]Queue:[/bold] {queue_size} papers pending")

    # Get recent lessons count
    lessons = db.get_lessons(pillar_id, limit=10)
    console.print(f"[bold]Lessons:[/bold] {len(lessons)} available")

    # Get quiz cards count
    quiz_cards = db.get_quiz_cards_for_review(pillar_id, limit=100)
    console.print(f"[bold]Quiz cards:[/bold] {len(quiz_cards)} due for review")


@app.command()
def review(
    pillar: str = typer.Option(..., "--pillar", "-p", help="Pillar to review (P1-P5)")
):
    """Show review status for a specific pillar: due quiz cards count.
    
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
                result = supabase_client.table("quiz_cards").select("id", count="exact").eq("pillar_id", pillar_id).lte("due_date", today).execute()
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


@app.command()
def learn(
    topic: str = typer.Argument(..., help="Topic to learn about (e.g., 'transformers', 'attention mechanisms')"),
    pillar: Optional[str] = typer.Option(None, "--pillar", "-p", help="Specific pillar to target (P1-P5). If not provided, auto-selects best pillar"),
    papers: int = typer.Option(3, "--papers", "-n", help="Number of papers to process", min=1, max=10)
):
    """Learn about a specific topic by running the pipeline.
    
    This command provides a simplified interface to run the learning pipeline
    on a specific topic. It will automatically select the best pillar if none
    is specified, search for relevant papers, and process them.
    """
    _log_env_path()
    _setup_logging()

    try:
        # Auto-select pillar if not provided
        if pillar:
            pillar_id = _validate_pillar(pillar)
            pillar_name = pillar
        else:
            pillar_id, pillar_name = _auto_select_pillar(topic)

        console.print(f"\n[bold green]🌱 Learning about '{topic}'[/bold green]")
        console.print(f"[bold]Target Pillar:[/bold] {pillar_name} - {PILLAR_CONFIGS[pillar_name]['name']}")
        console.print(f"[bold]Papers to process:[/bold] {papers}")

        # Add topic-specific papers to queue first
        with console.status("[bold blue]Searching for relevant papers..."):
            orchestrator = Orchestrator(enable_quiz=True)

            # Use discovery agent to find papers for this topic
            from .agents.discovery_agent import DiscoveryAgent
            from .schemas import DiscoveryInput, PillarConfig

            # Create pillar config for the topic
            pillar_config = PillarConfig(
                id=pillar_id,
                name=PILLAR_CONFIGS[pillar_name]['name'],
                goal=f"Learn about {topic} in the context of {PILLAR_CONFIGS[pillar_name]['goal']}",
                papers_per_day=papers,
                focus_areas=[topic]
            )

            # Run discovery for this specific topic
            discovery_agent = DiscoveryAgent()
            discovery_input = DiscoveryInput(
                pillar=pillar_config,
                recent_papers=[],
                priority_topics=[topic]
            )

            try:
                discovery_result = discovery_agent.run(discovery_input)
                console.print(f"[dim]🔍 Generated {len(discovery_result.queries)} search queries for '{topic}'[/dim]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not run discovery agent: {e}[/yellow]")
                console.print(f"[dim]Proceeding with standard pipeline for pillar {pillar_name}[/dim]")

        # Run the pipeline
        with console.status("[bold green]Processing papers..."):
            result = orchestrator.run_daily(pillar_id, papers_limit=papers)

        # Show results
        _show_learn_results(topic, pillar_name, result)

        # Exit with appropriate code
        if result.success:
            console.print(f"\n[bold green]✅ Successfully learned about '{topic}'![/bold green]")
            raise typer.Exit(0)
        else:
            console.print(f"\n[bold red]❌ Learning about '{topic}' encountered issues[/bold red]")
            raise typer.Exit(1)

    except typer.Exit:
        raise  # Re-raise typer.Exit without logging
    except Exception as e:
        console.print(f"\n[bold red]❌ Learn error: {str(e)}[/bold red]")
        logging.error(f"Learn error: {e}", exc_info=True)
        raise typer.Exit(1)


def _auto_select_pillar(topic: str) -> tuple[str, str]:
    """Auto-select the best pillar for a given topic."""
    topic_lower = topic.lower()

    # Simple keyword matching for pillar selection
    # P1: Linguistic & Cognitive Foundations
    p1_keywords = ['linguistic', 'language', 'cognitive', 'syntax', 'semantics', 'pragmatics', 'morphology', 'phonology']

    # P2: Models & Architectures
    p2_keywords = ['transformer', 'bert', 'gpt', 'attention', 'architecture', 'model', 'network', 'neural', 'llm', 'language model']

    # P3: Data, Training & Methodologies
    p3_keywords = ['training', 'data', 'dataset', 'methodology', 'optimization', 'learning', 'fine-tuning', 'pretraining']

    # P4: Evaluation & Interpretability
    p4_keywords = ['evaluation', 'interpretability', 'explainability', 'benchmark', 'metric', 'analysis', 'testing']

    # P5: Ethics & Applications
    p5_keywords = ['ethics', 'bias', 'fairness', 'application', 'deployment', 'society', 'responsible', 'harmful']

    # Check which pillar has the most keyword matches
    keyword_sets = {
        'linguistic-cognitive-foundations': p1_keywords,
        'models-architectures': p2_keywords,
        'data-training-methodologies': p3_keywords,
        'evaluation-interpretability': p4_keywords,
        'ethics-applications': p5_keywords
    }

    best_pillar = 'models-architectures'  # Default to Models & Architectures for most NLP topics
    max_matches = 0

    for pillar, keywords in keyword_sets.items():
        matches = sum(1 for keyword in keywords if keyword in topic_lower)
        if matches > max_matches:
            max_matches = matches
            best_pillar = pillar

    return _validate_pillar(best_pillar), best_pillar


def _show_learn_results(topic: str, pillar_name: str, result):
    """Show formatted results from the learn command."""
    # Create summary panel
    summary_lines = [
        f"[bold]Topic:[/bold] {topic}",
        f"[bold]Pillar:[/bold] {pillar_name} - {PILLAR_CONFIGS[pillar_name]['name']}",
        f"[bold]Papers processed:[/bold] {len(result.papers_processed)}",
        f"[bold]Lessons created:[/bold] {len(result.lessons_created)}",
        f"[bold]Quiz cards generated:[/bold] {len(result.quizzes_generated)}",
        f"[bold]Processing time:[/bold] {result.total_time_seconds:.1f}s"
    ]

    # Add paper details if available
    if result.papers_processed:
        summary_lines.append("\n[bold green]Papers processed:[/bold green]")
        for paper_id in result.papers_processed[:3]:  # Show first 3
            summary_lines.append(f"  • {paper_id}")
        if len(result.papers_processed) > 3:
            summary_lines.append(f"  ... and {len(result.papers_processed) - 3} more")

    # Add error details if any
    if result.errors:
        summary_lines.append("\n[bold red]Issues encountered:[/bold red]")
        for error in result.errors[:2]:  # Show first 2 errors
            summary_lines.append(f"  • {error.get('message', 'Unknown error')}")
        if len(result.errors) > 2:
            summary_lines.append(f"  ... and {len(result.errors) - 2} more")

    panel = Panel(
        "\n".join(summary_lines),
        title=f"🎓 Learning Results: {topic}",
        border_style="green" if result.success else "yellow"
    )

    console.print("\n")
    console.print(panel)


@app.command()
def queue(
    pillar: Optional[str] = typer.Option(None, "--pillar", "-p", help="Pillar to show queue for (P1-P5). If not provided, shows all pillars")
):
    """Show the paper processing queue.
    
    Displays papers waiting to be processed, ordered by priority.
    """
    _log_env_path()
    _setup_logging()

    try:
        if pillar:
            pillar_id = _validate_pillar(pillar)
            _show_pillar_queue(pillar_id)
        else:
            _show_all_queues()

        raise typer.Exit(0)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[bold red]❌ Queue error: {str(e)}[/bold red]")
        logging.error(f"Queue error: {e}", exc_info=True)
        raise typer.Exit(1)


def _show_pillar_queue(pillar_id: str):
    """Show queue for a specific pillar."""
    pillar_config = get_pillar_config(pillar_id)
    console.print(f"\n[bold blue]📜 Queue for {pillar_id} - {pillar_config['name']}[/bold blue]")

    try:
        # Get unprocessed papers from queue
        client = db.get_client()
        response = (client.table('paper_queue')
                   .select('*')
                   .eq('pillar_id', pillar_id)
                   .eq('processed', False)
                   .order('priority', desc=True)
                   .order('added_at', desc=True)
                   .limit(20)
                   .execute())

        if response['error']:
            console.print(f"[red]Error fetching queue: {response['error']}[/red]")
            return

        queue_papers = response['data'] or []

        if queue_papers:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Priority", style="yellow", width=8)
            table.add_column("Paper ID", style="cyan", width=20)
            table.add_column("Title", style="white")
            table.add_column("Source", style="dim", width=10)

            for paper in queue_papers:
                priority_str = str(paper.get('priority', 0))
                if paper.get('priority', 0) > 5:
                    priority_str = f"[red]{priority_str}[/red]"
                elif paper.get('priority', 0) > 0:
                    priority_str = f"[yellow]{priority_str}[/yellow]"

                paper_id = paper['paper_id'][:20] + "..." if len(paper['paper_id']) > 20 else paper['paper_id']
                title = paper.get('title', 'Unknown')[:50] + "..." if len(paper.get('title', '')) > 50 else paper.get('title', 'Unknown')
                source = paper.get('source', 'unknown')

                table.add_row(priority_str, paper_id, title, source)

            console.print("\n")
            console.print(table)
            console.print(f"\n[bold]📈 Total:[/bold] {len(queue_papers)} papers in queue")
        else:
            console.print("\n[dim]🎉 Queue is empty! No papers waiting to be processed.[/dim]")

    except Exception as e:
        console.print(f"[red]Error fetching queue: {e}[/red]")


def _show_all_queues():
    """Show queue overview for all pillars."""
    console.print("\n[bold blue]📜 Paper Processing Queues[/bold blue]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Pillar", style="cyan", width=6)
    table.add_column("Name", style="white", width=25)
    table.add_column("Queue Size", style="yellow", width=10)
    table.add_column("High Priority", style="red", width=12)
    table.add_column("Next Paper", style="green")

    total_queued = 0
    total_high_priority = 0

    for pillar_name, config in PILLAR_CONFIGS.items():
        pillar_id = _validate_pillar(pillar_name)

        try:
            # Get queue info
            client = db.get_client()
            response = (client.table('paper_queue')
                       .select('*')
                       .eq('pillar_id', pillar_id)
                       .eq('processed', False)
                       .order('priority', desc=True)
                       .order('added_at', desc=True)
                       .execute())

            if response['error']:
                table.add_row(pillar_name, config["name"][:25], "[red]Error[/red]", "[red]Error[/red]", "[red]Error[/red]")
                continue

            queue_papers = response['data'] or []
            queue_size = len(queue_papers)
            high_priority = len([p for p in queue_papers if p.get('priority', 0) > 5])

            total_queued += queue_size
            total_high_priority += high_priority

            # Get next paper to process
            next_paper = queue_papers[0]['title'][:30] + "..." if queue_papers and len(queue_papers[0].get('title', '')) > 30 else (queue_papers[0].get('title', 'Unknown') if queue_papers else "None")

            # Format queue size with color coding
            queue_str = str(queue_size)
            if queue_size > 10:
                queue_str = f"[red]{queue_size}[/red]"
            elif queue_size > 5:
                queue_str = f"[yellow]{queue_size}[/yellow]"
            else:
                queue_str = f"[green]{queue_size}[/green]"

            table.add_row(
                pillar_name,
                config["name"][:25],
                queue_str,
                str(high_priority),
                next_paper if queue_papers else "[dim]Empty[/dim]"
            )

        except Exception as e:
            logger.error(f"Error getting queue for pillar {pillar_name}: {e}")
            table.add_row(pillar_name, config["name"][:25], "[red]Error[/red]", "[red]Error[/red]", "[red]Error[/red]")

    console.print("\n")
    console.print(table)
    console.print(f"\n[bold]📈 Totals:[/bold] {total_queued} papers queued • {total_high_priority} high priority")
    console.print("\n[dim]💡 Use --pillar <ID> for detailed queue view[/dim]")


@app.command()
def discover(
    pillar: str = typer.Option(..., "--pillar", "-p", help="Pillar to discover papers for (P1-P5)"),
    topics: Optional[str] = typer.Option(None, "--topics", "-t", help="Comma-separated topic hints (e.g., 'RAG,retrieval')"),
    interactive: bool = typer.Option(True, "--interactive", "-i", help="Prompt for paper selection"),
    auto_select: int = typer.Option(0, "--auto", "-a", help="Auto-select top N papers (0 for none)")
):
    """Discover papers for a pillar with optional user selection.

    Shows candidates from multiple sources and allows interactive selection
    of which papers to process through the pipeline.
    """
    _log_env_path()
    _setup_logging()

    try:
        pillar_id = _validate_pillar(pillar)
        pillar_name = pillar

        # Parse topics
        priority_topics = []
        if topics:
            priority_topics = [t.strip() for t in topics.split(",")]

        console.print(f"\n[bold green]🔍 Discovering papers for {pillar_name}[/bold green]")
        console.print(f"[bold]Pillar:[/bold] {PILLAR_CONFIGS[pillar_name]['name']}")
        if priority_topics:
            console.print(f"[bold]Topic hints:[/bold] {', '.join(priority_topics)}")

        # Run discovery
        with console.status("[bold blue]Searching multiple sources..."):
            orchestrator = Orchestrator(enable_quiz=True)
            candidates = orchestrator.run_discovery_with_selection(
                pillar_id=pillar_id,
                priority_topics=priority_topics,
                limit=10
            )

        if not candidates:
            console.print("\n[yellow]No candidates found. Try different topic hints.[/yellow]")
            raise typer.Exit(1)

        # Track attempted papers to exclude from re-selection
        attempted_indices = set()
        total_processed = 0
        total_errors = 0

        # Selection loop - allows re-selection after errors
        while True:
            # Build available candidates (exclude already attempted)
            available = [(i, c) for i, c in enumerate(candidates) if i not in attempted_indices]

            if not available:
                console.print("\n[yellow]No more candidates available.[/yellow]")
                break

            # Display candidates with PDF indicator
            console.print(f"\n[bold]Available candidates ({len(available)} remaining):[/bold]\n")

            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("#", style="cyan", width=3)
            table.add_column("Title", style="white", width=38)
            table.add_column("Year", style="yellow", width=6)
            table.add_column("Cites", style="green", width=6)
            table.add_column("Source", style="blue", width=10)
            table.add_column("PDF", style="dim", width=4)
            table.add_column("Score", style="dim", width=5)

            # Map display numbers to original indices
            display_to_original = {}
            for display_num, (orig_idx, candidate) in enumerate(available, 1):
                display_to_original[display_num] = orig_idx

                title = candidate.paper.title
                if len(title) > 38:
                    title = title[:35] + "..."

                # PDF indicator
                has_pdf = "[green]✓[/green]" if candidate.paper.url_pdf else "[red]✗[/red]"

                table.add_row(
                    str(display_num),
                    title,
                    str(candidate.paper.year or "?"),
                    str(candidate.citation_count),
                    candidate.source,
                    has_pdf,
                    f"{candidate.relevance_score:.2f}"
                )

            console.print(table)

            # Handle selection
            selected_indices = []

            if auto_select > 0 and not attempted_indices:
                # Auto-select top N (only on first iteration)
                selected_indices = [display_to_original[i] for i in range(1, min(auto_select + 1, len(available) + 1))]
                console.print(f"\n[dim]Auto-selected top {len(selected_indices)} papers[/dim]")
                auto_select = 0  # Only auto-select once

            elif interactive:
                # Prompt for selection
                if attempted_indices:
                    console.print("\n[bold]Select more papers to process:[/bold]")
                else:
                    console.print("\n[bold]Select papers to process:[/bold]")
                console.print("[dim]Enter numbers (e.g., '1,3'), 'all', or 'q' to quit[/dim]")
                console.print("[dim]Papers with [green]✓[/green] have PDF URLs available[/dim]")

                selection = typer.prompt("Selection", default="")

                if selection.lower() == 'q':
                    console.print("[yellow]Selection cancelled[/yellow]")
                    break
                elif selection.lower() == 'all':
                    selected_indices = [display_to_original[i] for i in display_to_original.keys()]
                else:
                    try:
                        display_nums = [int(x.strip()) for x in selection.split(",")]
                        # Convert display numbers to original indices
                        selected_indices = [display_to_original[d] for d in display_nums if d in display_to_original]
                    except ValueError:
                        console.print("[red]Invalid selection. Please enter numbers separated by commas.[/red]")
                        continue
            else:
                console.print("\n[dim]Use --interactive or --auto to select papers[/dim]")
                break

            if not selected_indices:
                console.print("[yellow]No valid papers selected[/yellow]")
                continue

            # Mark selected as attempted
            for idx in selected_indices:
                attempted_indices.add(idx)

            # Get selected papers (full PaperRef objects to preserve PDF URLs)
            selected_papers = [candidates[i].paper for i in selected_indices]

            console.print(f"\n[bold green]Processing {len(selected_papers)} selected papers...[/bold green]")

            # Process selected papers with full paper objects
            with console.status("[bold green]Processing papers through pipeline..."):
                result = orchestrator.process_selected_papers(
                    pillar_id=pillar_id,
                    papers=selected_papers
                )

            # Show results for this batch
            if result.papers_processed:
                console.print(f"\n[bold green]✅ Processed {len(result.papers_processed)} papers successfully[/bold green]")
                console.print(f"[bold]Lessons created:[/bold] {len(result.lessons_created)}")
                console.print(f"[bold]Quiz cards generated:[/bold] {len(result.quizzes_generated)}")
                total_processed += len(result.papers_processed)

            if result.errors:
                console.print(f"\n[bold yellow]⚠️ {len(result.errors)} papers failed:[/bold yellow]")
                for error in result.errors:
                    console.print(f"  • {error.get('message', 'Unknown error')}")
                total_errors += len(result.errors)

            # If not interactive or no more candidates, exit loop
            if not interactive:
                break

        # Final summary
        if total_processed > 0 or total_errors > 0:
            console.print("\n[bold]Session Summary:[/bold]")
            console.print(f"  Papers processed: [green]{total_processed}[/green]")
            console.print(f"  Papers failed: [red]{total_errors}[/red]")

        raise typer.Exit(0 if total_processed > 0 else 1)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[bold red]❌ Discovery error: {str(e)}[/bold red]")
        logging.error(f"Discovery error: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def history(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of recent papers to show", min=1, max=50),
    pillar: Optional[str] = typer.Option(None, "--pillar", "-p", help="Filter by pillar (P1-P5)")
):
    """Show recently processed papers.
    
    Displays papers that have been processed through the pipeline,
    with their processing status and generated content.
    """
    _log_env_path()
    _setup_logging()

    try:
        console.print("\n[bold blue]📋 Recent Processing History[/bold blue]")

        if pillar:
            pillar_id = _validate_pillar(pillar)
            console.print(f"[dim]Filtered by: {pillar} - {PILLAR_CONFIGS[pillar]['name']}[/dim]")
            _show_pillar_history(pillar_id, limit)
        else:
            _show_global_history(limit)

        raise typer.Exit(0)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[bold red]❌ History error: {str(e)}[/bold red]")
        logging.error(f"History error: {e}", exc_info=True)
        raise typer.Exit(1)


def _show_pillar_history(pillar_id: str, limit: int):
    """Show processing history for a specific pillar."""
    try:
        # Get recent papers
        papers = db.get_papers(pillar_id, limit=limit)

        if papers:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Paper ID", style="cyan", width=20)
            table.add_column("Title", style="white", width=40)
            table.add_column("Authors", style="blue", width=25)
            table.add_column("Year", style="green", width=6)
            table.add_column("Content", style="yellow", width=15)

            for paper in papers:
                paper_id = paper.id[:20] + "..." if len(paper.id) > 20 else paper.id
                title = paper.title[:40] + "..." if len(paper.title) > 40 else paper.title
                authors = ", ".join(paper.authors[:2])  # Show first 2 authors
                if len(paper.authors) > 2:
                    authors += f" +{len(paper.authors) - 2}"
                authors = authors[:25] + "..." if len(authors) > 25 else authors
                year = str(paper.year) if paper.year else "N/A"

                # Check what content exists for this paper
                content_items = []
                try:
                    # Check for notes
                    notes = db.get_recent_notes(pillar_id, limit=100)
                    if any(note.paper_id == paper.id for note in notes):
                        content_items.append("Notes")

                    # Check for lessons
                    lessons = db.get_lessons(pillar_id, limit=100)
                    if any(lesson.paper_id == paper.id for lesson in lessons):
                        content_items.append("Lesson")

                    # Check for quiz cards
                    quiz_cards = db.get_quiz_cards_for_review(pillar_id, limit=1000)
                    if any(card.paper_id == paper.id for card in quiz_cards):
                        content_items.append("Quiz")

                except Exception:
                    pass

                content_str = ", ".join(content_items) if content_items else "[dim]None[/dim]"

                table.add_row(paper_id, title, authors, year, content_str)

            console.print("\n")
            console.print(table)
            console.print(f"\n[bold]📆 Showing {len(papers)} recent papers[/bold]")
        else:
            console.print("\n[dim]No papers found in history[/dim]")

    except Exception as e:
        console.print(f"[red]Error fetching history: {e}[/red]")


def _show_global_history(limit: int):
    """Show processing history across all pillars."""
    console.print("[dim]Showing recent papers across all pillars[/dim]")

    all_papers = []

    # Collect papers from all pillars
    for pillar_name in PILLAR_CONFIGS.keys():
        try:
            pillar_id = _validate_pillar(pillar_name)
            papers = db.get_papers(pillar_id, limit=limit)
            for paper in papers:
                all_papers.append((pillar_name, paper))
        except Exception as e:
            logger.error(f"Error getting papers for pillar {pillar_name}: {e}")

    # Sort by year (or could sort by created_at if available)
    all_papers.sort(key=lambda x: x[1].year or 0, reverse=True)

    # Take only the requested limit
    all_papers = all_papers[:limit]

    if all_papers:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Pillar", style="cyan", width=6)
        table.add_column("Paper ID", style="blue", width=18)
        table.add_column("Title", style="white", width=35)
        table.add_column("Authors", style="green", width=20)
        table.add_column("Year", style="yellow", width=6)

        for pillar_name, paper in all_papers:
            paper_id = paper.id[:18] + "..." if len(paper.id) > 18 else paper.id
            title = paper.title[:35] + "..." if len(paper.title) > 35 else paper.title
            authors = ", ".join(paper.authors[:2])  # Show first 2 authors
            if len(paper.authors) > 2:
                authors += f" +{len(paper.authors) - 2}"
            authors = authors[:20] + "..." if len(authors) > 20 else authors
            year = str(paper.year) if paper.year else "N/A"

            table.add_row(pillar_name, paper_id, title, authors, year)

        console.print("\n")
        console.print(table)
        console.print(f"\n[bold]📆 Showing {len(all_papers)} recent papers across all pillars[/bold]")
    else:
        console.print("\n[dim]No papers found in history[/dim]")


@app.command()
def papers(
    recent: int = typer.Option(10, "--recent", "-n", help="Number of recent papers to show", min=1, max=50),
    pillar: Optional[str] = typer.Option(None, "--pillar", "-p", help="Filter by pillar (P1-P5)")
):
    """Browse recent papers in the system.
    
    Shows papers that have been discovered and added to the system,
    with their metadata and processing status.
    """
    _log_env_path()
    _setup_logging()

    try:
        console.print("\n[bold blue]📚 Recent Papers[/bold blue]")

        if pillar:
            pillar_id = _validate_pillar(pillar)
            console.print(f"[dim]Filtered by: {pillar} - {PILLAR_CONFIGS[pillar]['name']}[/dim]")
            papers_list = db.get_papers(pillar_id, limit=recent)
        else:
            # Get papers from all pillars
            all_papers = []
            for pillar_name in PILLAR_CONFIGS.keys():
                try:
                    pillar_id = _validate_pillar(pillar_name)
                    pillar_papers = db.get_papers(pillar_id, limit=recent)
                    for paper in pillar_papers:
                        all_papers.append((pillar_name, paper))
                except Exception as e:
                    logger.error(f"Error getting papers for pillar {pillar_name}: {e}")

            # Sort by year and take recent limit
            all_papers.sort(key=lambda x: x[1].year or 0, reverse=True)
            papers_list = all_papers[:recent]

        if papers_list:
            table = Table(show_header=True, header_style="bold magenta")

            if pillar:
                # Single pillar view - more detailed
                table.add_column("Paper ID", style="cyan", width=20)
                table.add_column("Title", style="white", width=40)
                table.add_column("Authors", style="blue", width=25)
                table.add_column("Year", style="green", width=6)
                table.add_column("Citations", style="yellow", width=9)

                for paper in papers_list:
                    paper_id = paper.id[:20] + "..." if len(paper.id) > 20 else paper.id
                    title = paper.title[:40] + "..." if len(paper.title) > 40 else paper.title
                    authors = ", ".join(paper.authors[:2])
                    if len(paper.authors) > 2:
                        authors += f" +{len(paper.authors) - 2}"
                    authors = authors[:25] + "..." if len(authors) > 25 else authors
                    year = str(paper.year) if paper.year else "N/A"
                    citations = str(paper.citation_count) if paper.citation_count else "N/A"

                    table.add_row(paper_id, title, authors, year, citations)
            else:
                # Multi-pillar view - more compact
                table.add_column("Pillar", style="cyan", width=6)
                table.add_column("Paper ID", style="blue", width=18)
                table.add_column("Title", style="white", width=35)
                table.add_column("Authors", style="green", width=20)
                table.add_column("Year", style="yellow", width=6)

                for pillar_name, paper in papers_list:
                    paper_id = paper.id[:18] + "..." if len(paper.id) > 18 else paper.id
                    title = paper.title[:35] + "..." if len(paper.title) > 35 else paper.title
                    authors = ", ".join(paper.authors[:2])
                    if len(paper.authors) > 2:
                        authors += f" +{len(paper.authors) - 2}"
                    authors = authors[:20] + "..." if len(authors) > 20 else authors
                    year = str(paper.year) if paper.year else "N/A"

                    table.add_row(pillar_name, paper_id, title, authors, year)

            console.print("\n")
            console.print(table)

            total_shown = len(papers_list)
            if pillar:
                console.print(f"\n[bold]📈 Showing {total_shown} papers from {pillar}[/bold]")
            else:
                console.print(f"\n[bold]📈 Showing {total_shown} recent papers across all pillars[/bold]")
        else:
            console.print("\n[dim]No papers found[/dim]")

        raise typer.Exit(0)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[bold red]❌ Papers error: {str(e)}[/bold red]")
        logging.error(f"Papers error: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def lessons(
    pillar: str = typer.Option(..., "--pillar", "-p", help="Pillar to show lessons from (P1-P5)"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of lessons to show", min=1, max=50)
):
    """Browse lessons from a specific pillar.
    
    Shows synthesized lessons that have been generated from processed papers,
    with their key takeaways and learning content.
    """
    _log_env_path()
    _setup_logging()

    try:
        pillar_id = _validate_pillar(pillar)

        console.print(f"\n[bold blue]🎓 Lessons from {pillar} - {PILLAR_CONFIGS[pillar]['name']}[/bold blue]")

        # Get lessons for this pillar
        lessons_list = db.get_lessons(pillar_id, limit=limit)

        if lessons_list:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Paper ID", style="cyan", width=20)
            table.add_column("TL;DR", style="white", width=45)
            table.add_column("Difficulty", style="yellow", width=10)
            table.add_column("Time", style="green", width=8)
            table.add_column("Takeaways", style="blue", width=12)

            for lesson in lessons_list:
                paper_id = lesson.paper_id[:20] + "..." if len(lesson.paper_id) > 20 else lesson.paper_id
                tl_dr = lesson.tl_dr[:45] + "..." if len(lesson.tl_dr) > 45 else lesson.tl_dr

                # Format difficulty
                difficulty_map = {1: "Easy", 2: "Medium", 3: "Hard"}
                difficulty = difficulty_map.get(lesson.difficulty, "Medium")

                time_str = f"{lesson.estimated_time}min"
                takeaways_count = str(len(lesson.takeaways))

                table.add_row(paper_id, tl_dr, difficulty, time_str, takeaways_count)

            console.print("\n")
            console.print(table)

            # Show a detailed view of the first lesson
            if lessons_list:
                first_lesson = lessons_list[0]
                console.print(f"\n[bold green]🔍 Preview: {first_lesson.paper_id}[/bold green]")

                preview_lines = [
                    f"[bold]TL;DR:[/bold] {first_lesson.tl_dr}",
                    f"[bold]Difficulty:[/bold] {difficulty_map.get(first_lesson.difficulty, 'Medium')} ({first_lesson.estimated_time} min read)"
                ]

                if first_lesson.takeaways:
                    preview_lines.append("\n[bold]Key Takeaways:[/bold]")
                    for i, takeaway in enumerate(first_lesson.takeaways[:3], 1):
                        preview_lines.append(f"  {i}. {takeaway}")
                    if len(first_lesson.takeaways) > 3:
                        preview_lines.append(f"  ... and {len(first_lesson.takeaways) - 3} more")

                if first_lesson.connections:
                    preview_lines.append("\n[bold]Connections:[/bold]")
                    for connection in first_lesson.connections[:2]:
                        preview_lines.append(f"  • {connection}")
                    if len(first_lesson.connections) > 2:
                        preview_lines.append(f"  ... and {len(first_lesson.connections) - 2} more")

                preview_panel = Panel(
                    "\n".join(preview_lines),
                    title="Lesson Preview",
                    border_style="green"
                )
                console.print(preview_panel)

            console.print(f"\n[bold]📈 Showing {len(lessons_list)} lessons from {pillar}[/bold]")
        else:
            console.print("\n[dim]No lessons found for this pillar[/dim]")
            console.print("[dim]Try running the pipeline to generate lessons: python -m nlp_pillars.cli run --pillar P1[/dim]")

        raise typer.Exit(0)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[bold red]❌ Lessons error: {str(e)}[/bold red]")
        logging.error(f"Lessons error: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def quiz(
    random: int = typer.Option(5, "--random", "-r", help="Number of random quiz questions", min=1, max=20),
    pillar: Optional[str] = typer.Option(None, "--pillar", "-p", help="Filter by pillar (P1-P5). If not provided, uses all pillars"),
    difficulty: Optional[str] = typer.Option(None, "--difficulty", "-d", help="Filter by difficulty (easy, medium, hard)")
):
    """Generate a random quiz from learned content.
    
    Creates a quiz using questions from processed papers across pillars.
    Great for testing your knowledge and spaced repetition practice.
    """
    _log_env_path()
    _setup_logging()

    try:
        console.print(f"\n[bold blue]🧠 Random Quiz ({random} questions)[/bold blue]")

        # Collect quiz cards
        all_cards = []

        if pillar:
            pillar_id = _validate_pillar(pillar)
            console.print(f"[dim]Source: {pillar} - {PILLAR_CONFIGS[pillar]['name']}[/dim]")
            cards = db.get_quiz_cards_for_review(pillar_id, limit=100)
            all_cards.extend(cards)
        else:
            console.print("[dim]Source: All pillars[/dim]")
            for pillar_name in PILLAR_CONFIGS.keys():
                try:
                    pillar_id = _validate_pillar(pillar_name)
                    cards = db.get_quiz_cards_for_review(pillar_id, limit=50)
                    all_cards.extend(cards)
                except Exception as e:
                    logger.error(f"Error getting quiz cards for pillar {pillar_name}: {e}")

        # Filter by difficulty if specified
        if difficulty:
            difficulty_map = {"easy": 1, "medium": 2, "hard": 3}
            if difficulty.lower() in difficulty_map:
                target_difficulty = difficulty_map[difficulty.lower()]
                all_cards = [card for card in all_cards if card.difficulty == target_difficulty]
                console.print(f"[dim]Difficulty filter: {difficulty.title()}[/dim]")

        if not all_cards:
            console.print("\n[yellow]No quiz cards found![/yellow]")
            console.print("[dim]Try running the pipeline to generate quiz cards: python -m nlp_pillars.cli run --pillar P1[/dim]")
            raise typer.Exit(0)

        # Randomly select cards
        import random as rand
        selected_cards = rand.sample(all_cards, min(random, len(all_cards)))

        console.print(f"\n[bold green]🎯 Starting quiz with {len(selected_cards)} questions[/bold green]")
        console.print("[dim]Press Enter to reveal answers, 'q' to quit[/dim]\n")

        correct_answers = 0
        for i, card in enumerate(selected_cards, 1):
            # Show question
            question_panel = Panel(
                f"[bold]Q{i}:[/bold] {card.question}",
                title=f"Question {i}/{len(selected_cards)}",
                border_style="blue"
            )
            console.print(question_panel)

            # Wait for user input
            user_input = typer.prompt("\nPress Enter to see answer (or 'q' to quit)", default="", show_default=False)
            if user_input.lower() == 'q':
                console.print("\n[yellow]Quiz ended early[/yellow]")
                break

            # Show answer
            answer_panel = Panel(
                f"[bold green]Answer:[/bold green] {card.answer}",
                title="Answer",
                border_style="green"
            )
            console.print(answer_panel)

            # Ask if they got it right
            got_it_right = typer.confirm("Did you get it right?", default=True)
            if got_it_right:
                correct_answers += 1
                console.print("[green]✅ Correct![/green]")
            else:
                console.print("[red]❌ Keep studying![/red]")

            # Show paper source
            console.print(f"[dim]Source: {card.paper_id} (Pillar {card.pillar_id})[/dim]")

            if i < len(selected_cards):
                console.print("\n" + "-" * 50 + "\n")

        # Show final results
        if correct_answers > 0 or len(selected_cards) > 0:
            percentage = (correct_answers / len(selected_cards)) * 100 if len(selected_cards) > 0 else 0

            if percentage >= 80:
                emoji = "🎉"
                message = "Excellent!"
                color = "green"
            elif percentage >= 60:
                emoji = "😊"
                message = "Good job!"
                color = "yellow"
            else:
                emoji = "💪"
                message = "Keep practicing!"
                color = "red"

            results_panel = Panel(
                f"[bold]{emoji} {message}[/bold]\n\n"
                f"Score: {correct_answers}/{len(selected_cards)} ({percentage:.0f}%)\n"
                f"Cards reviewed: {len(selected_cards)}",
                title="Quiz Results",
                border_style=color
            )
            console.print("\n")
            console.print(results_panel)

        raise typer.Exit(0)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[bold red]❌ Quiz error: {str(e)}[/bold red]")
        logging.error(f"Quiz error: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command("pillars")
def pillars_list():
    """List all available learning pillars.
    
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
