"""Command Line Interface (CLI) for Mathematical Investigation Agent (MIA)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional
import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from mia.agent.loop import MathematicalAgent
from mia.config import AppConfig, load_config
from mia.evaluation.runner import EvaluationRunner
from mia.execution.modal_backend import ModalExecutionBackend
from mia.models import AgentProblemView, ProblemRecord
from mia.reporting.reporter import Reporter
from mia.sources import SOURCES, get_source
from mia.storage.experiment_store import ExperimentStore

app = typer.Typer(
    name="mia",
    help="Mathematical Investigation Agent (MIA) - Autonomous mathematical exploration, experimentation, and verification.",
    add_completion=False
)
config_app = typer.Typer(help="Inspect and validate configuration.")
sources_app = typer.Typer(help="Manage benchmark and problem sources.")
problems_app = typer.Typer(help="Browse and inspect mathematical problems.")
modal_app_cmd = typer.Typer(help="Modal remote compute management.")

app.add_typer(config_app, name="config")
app.add_typer(sources_app, name="sources")
app.add_typer(problems_app, name="problems")
app.add_typer(modal_app_cmd, name="modal")

console = Console()


# ---------------------------------------------------------------------
# Config Commands
# ---------------------------------------------------------------------
@config_app.command("validate")
def validate_config(
    config_file: Optional[Path] = typer.Option(
        None, "--config-file", "-c", help="Path to YAML configuration file"
    )
):
    """Validate configuration and print effective settings with secrets redacted."""
    try:
        cfg = load_config(config_path=config_file)
        redacted = cfg.redact_secrets()
        console.print(Panel(
            yaml.dump(redacted, default_flow_style=False, sort_keys=False),
            title="[bold green]Effective MIA Configuration (Secrets Redacted)[/bold green]",
            border_style="green"
        ))
        typer.echo("Configuration is valid.")
    except Exception as e:
        console.print(f"[bold red]Configuration Validation Error:[/bold red] {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------
# Sources Commands
# ---------------------------------------------------------------------
@sources_app.command("list")
def list_sources():
    """List available benchmark and problem sources."""
    table = Table(title="Registered Problem Sources")
    table.add_column("Source Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")

    descriptions = {
        "imo": "International Mathematical Olympiad official documents and public PDFs",
        "hendrycks_math": "Hugging Face EleutherAI/hendrycks_math reasoning dataset",
        "controlled": "Controlled, programmatic mathematical tasks with verified independent ground truth"
    }

    for name in SOURCES:
        table.add_row(name, descriptions.get(name, "Mathematical problem source"))

    console.print(table)


@sources_app.command("sync")
def sync_sources(
    source: str = typer.Option("controlled", "--source", "-s", help="Source name ('imo', 'hendrycks_math', 'controlled')"),
    limit: Optional[int] = typer.Option(5, "--limit", "-n", help="Max problems to synchronize"),
    year: Optional[int] = typer.Option(None, "--year", "-y", help="Specific year for IMO sync")
):
    """Retrieve or update benchmark problems from official sources."""
    try:
        src = get_source(source)
        kwargs = {}
        if year:
            kwargs["years"] = [year]
        console.print(f"Syncing source '[bold cyan]{source}[/bold cyan]'...")
        new_records = src.sync(limit=limit, **kwargs)

        console.print(f"[bold green]Sync successful:[/bold green] {len(new_records)} new problem records acquired.")
        if src.failures:
            console.print(f"[yellow]Warnings/Failures encountered ({len(src.failures)}):[/yellow]")
            for f in src.failures:
                console.print(f"  - {f['identifier']}: {f['reason']}")
    except Exception as e:
        console.print(f"[bold red]Source Sync Failed:[/bold red] {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------
# Problems Commands
# ---------------------------------------------------------------------
@problems_app.command("list")
def list_problems(
    source: str = typer.Option("controlled", "--source", "-s", help="Source to list problems from"),
    subject: Optional[str] = typer.Option(None, "--subject", help="Filter by mathematical domain/subject"),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum problems to show")
):
    """List locally available problems and relevant metadata."""
    try:
        src = get_source(source)
        probs = src.list_problems(subject=subject, limit=limit)
        if not probs:
            # Attempt sync if empty
            probs = src.sync(limit=limit)

        table = Table(title=f"Problems in Source: {source} (Total: {len(probs)})")
        table.add_column("Problem ID", style="cyan", no_wrap=True)
        table.add_column("Subject", style="magenta")
        table.add_column("Difficulty", style="yellow")
        table.add_column("Statement Preview", style="white")

        for p in probs[:limit]:
            preview = p.statement[:70] + "..." if len(p.statement) > 70 else p.statement
            table.add_row(p.problem_id, p.subject or "general", p.difficulty or "N/A", preview)

        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Error listing problems:[/bold red] {e}")
        raise typer.Exit(code=1)


@problems_app.command("show")
def show_problem(
    problem_id: str = typer.Argument(..., help="Unique identifier of the problem"),
    show_answer: bool = typer.Option(False, "--show-answer", help="Display official reference answer (evaluator view only)")
):
    """Display problem statement without exposing reference answer to solving agent."""
    found = None
    for s_name in SOURCES:
        src = get_source(s_name)
        prob = src.get_problem(problem_id)
        if prob:
            found = prob
            break

    if not found:
        console.print(f"[bold red]Problem '{problem_id}' not found in any source.[/bold red]")
        raise typer.Exit(code=1)

    panel_content = f"[bold]Problem ID:[/bold] {found.problem_id}\n"
    panel_content += f"[bold]Source:[/bold] {found.source} ({found.source_url or 'N/A'})\n"
    panel_content += f"[bold]Subject:[/bold] {found.subject or 'N/A'} | [bold]Difficulty:[/bold] {found.difficulty or 'N/A'}\n\n"
    panel_content += f"[bold]Statement:[/bold]\n{found.statement}\n"

    if show_answer and found.official_answer:
        panel_content += f"\n[bold yellow]Official Answer (Evaluator Only):[/bold yellow]\n{found.official_answer}\n"
        if found.reference_solution:
            panel_content += f"\n[bold yellow]Reference Solution:[/bold yellow]\n{found.reference_solution}\n"

    console.print(Panel(panel_content, title=f"Problem: {problem_id}", border_style="cyan"))


# ---------------------------------------------------------------------
# Solve & Resume Commands
# ---------------------------------------------------------------------
@app.command("solve")
def solve_problem(
    problem_identifier: str = typer.Argument(..., help="Problem ID (e.g. 'controlled-euler-primality') or path to local problem file"),
    config_file: Optional[Path] = typer.Option(None, "--config-file", "-c", help="Configuration YAML file"),
    backend: Optional[str] = typer.Option(None, "--backend", "-b", help="Override backend ('local' or 'modal')"),
    max_steps: Optional[int] = typer.Option(None, "--max-steps", help="Override max investigation steps"),
    demo: bool = typer.Option(False, "--demo", help="Run in reproducible offline demonstration mode")
):
    """Start an autonomous mathematical investigation for a problem."""
    try:
        cli_overrides = {}
        if backend:
            cli_overrides["execution.backend"] = backend
        if max_steps:
            cli_overrides["agent.max_steps"] = max_steps
        if demo:
            cli_overrides["model.provider"] = "mock"

        cfg = load_config(config_path=config_file, cli_overrides=cli_overrides)

        # Ingest problem
        problem_view: Optional[AgentProblemView] = None
        prob_path = Path(problem_identifier)
        if prob_path.is_file():
            # Load local custom problem
            with open(prob_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                problem_view = AgentProblemView.model_validate(data)
        else:
            # Search in sources
            for s_name in SOURCES:
                src = get_source(s_name)
                found = src.get_problem(problem_identifier)
                if found:
                    problem_view = found.to_agent_view()
                    break

        if not problem_view:
            console.print(f"[bold red]Problem '{problem_identifier}' not found in registered sources or local path.[/bold red]")
            raise typer.Exit(code=1)

        console.print(Panel(
            f"[bold]Investigating Problem:[/bold] {problem_view.problem_id}\n"
            f"[bold]Statement:[/bold] {problem_view.statement}\n"
            f"[bold]Backend:[/bold] {cfg.execution.backend} | [bold]Provider:[/bold] {cfg.model.provider}",
            title="[bold green]MIA Investigation Started[/bold green]",
            border_style="green"
        ))

        agent = MathematicalAgent(cfg)
        report = agent.investigate(problem_view)

        # Print Investigation Outcome
        console.print("\n" + "=" * 60)
        console.print(f"[bold cyan]INVESTIGATION COMPLETE:[/bold cyan] {report.investigation_id}")
        console.print(f"Status: [bold]{report.status.value}[/bold] | Reason: {report.termination_reason}")
        console.print(f"Steps: {report.total_steps} | Experiments: {report.total_experiments} | Duration: {report.wall_clock_seconds:.2f}s")

        if report.hypotheses_evaluated:
            console.print("\n[bold]Hypotheses Evaluated:[/bold]")
            for h in report.hypotheses_evaluated:
                status_color = "green" if h.status.value in ("CONFIRMED", "ACTIVE") else "red"
                console.print(f"  - [{status_color}][{h.status.value}][/{status_color}] {h.text}")

        if report.candidate_solution:
            console.print(Panel(report.candidate_solution, title="Final Solution / Report", border_style="cyan"))

        if report.verification:
            v = report.verification
            v_color = "green" if v.status.value == "VERIFIED" else "yellow"
            console.print(f"\nVerification Level: {v.level.value} | Status: [{v_color}]{v.status.value}[/{v_color}] (Confidence: {v.confidence:.1%})")
            for ev in v.evidence:
                console.print(f"  ✓ {ev}")

    except Exception as e:
        console.print(f"[bold red]Investigation failed:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command("resume")
def resume_investigation(
    investigation_id: str = typer.Argument(..., help="ID of interrupted investigation to resume"),
    additional_steps: int = typer.Option(5, "--steps", help="Additional steps budget"),
    config_file: Optional[Path] = typer.Option(None, "--config-file", "-c")
):
    """Resume an interrupted investigation using its persisted state."""
    try:
        cfg = load_config(config_path=config_file)
        agent = MathematicalAgent(cfg)
        console.print(f"Resuming investigation '[bold cyan]{investigation_id}[/bold cyan]' with +{additional_steps} steps...")
        report = agent.resume(investigation_id, additional_steps=additional_steps)
        console.print(f"[bold green]Resumed investigation complete.[/bold green] Final status: {report.status.value}")
    except Exception as e:
        console.print(f"[bold red]Failed to resume investigation:[/bold red] {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------
# Experiments & Reporting
# ---------------------------------------------------------------------
@app.command("experiments")
def list_experiments(
    investigation_id: Optional[str] = typer.Option(None, "--investigation-id", "-i", help="Filter by investigation ID"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max experiments to display")
):
    """Show recorded experiments, results, and execution status."""
    try:
        cfg = load_config()
        store = ExperimentStore(cfg.database_url)
        exps = store.list_experiments(investigation_id=investigation_id, limit=limit)

        table = Table(title=f"Recorded Experiments (Showing {len(exps)})")
        table.add_column("Exp ID", style="cyan", no_wrap=True)
        table.add_column("Inv ID", style="magenta")
        table.add_column("Tool", style="yellow")
        table.add_column("Backend", style="white")
        table.add_column("Exit", style="white")
        table.add_column("Runtime", style="white")

        for exp in exps:
            exit_code = exp.get("exit_status", 0)
            exit_display = "[green]0[/green]" if exit_code == 0 else f"[red]{exit_code}[/red]"
            table.add_row(
                exp.get("id"),
                exp.get("investigation_id"),
                exp.get("tool_name"),
                exp.get("backend"),
                exit_display,
                f"{exp.get('runtime_seconds', 0.0):.3f}s"
            )

        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Error querying experiments:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command("evaluate")
def evaluate_benchmark(
    source: str = typer.Option("controlled", "--source", "-s", help="Benchmark source ('controlled', 'imo', 'hendrycks_math')"),
    configs: str = typer.Option("A,B,C", "--configs", help="Comma-separated configuration names (e.g. 'A,B,C')"),
    num_problems: int = typer.Option(5, "--num-problems", "-n", help="Number of problems to sample"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducible problem sampling"),
    config_file: Optional[Path] = typer.Option(None, "--config-file", "-c")
):
    """Run comparative evaluation suite across baseline configurations."""
    try:
        cfg = load_config(config_path=config_file)
        runner = EvaluationRunner(cfg)
        config_list = [c.strip() for c in configs.split(",") if c.strip()]

        console.print(Panel(
            f"[bold]Benchmark Source:[/bold] {source}\n"
            f"[bold]Configurations:[/bold] {', '.join(config_list)}\n"
            f"[bold]Problems to Sample:[/bold] {num_problems} (Seed: {seed})",
            title="[bold green]MIA Evaluation Suite Running[/bold green]",
            border_style="green"
        ))

        res = runner.run_evaluation(
            source_name=source,
            configs=config_list,
            num_problems=num_problems,
            seed=seed
        )

        console.print(f"\n[bold green]Evaluation completed successfully![/bold green]")
        console.print(f"Results JSONL: [cyan]{res['results_jsonl']}[/cyan]")
        console.print(f"Summary CSV:   [cyan]{res['summary_csv']}[/cyan]")
        console.print(f"Report MD:     [cyan]{res['report_md']}[/cyan]\n")

        # Print report markdown directly to console
        console.print(Path(res["report_md"]).read_text(encoding="utf-8"))

    except Exception as e:
        console.print(f"[bold red]Evaluation failed:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command("report")
def generate_report(
    investigation_id: Optional[str] = typer.Option(None, "--investigation-id", "-i", help="Investigation ID to report on"),
    eval_run_id: Optional[str] = typer.Option(None, "--eval-run-id", "-e", help="Evaluation Run ID to display"),
    trace: bool = typer.Option(False, "--trace", "-t", help="Display sequential event trace")
):
    """Produce a report from an existing evaluation or investigation without rerunning."""
    try:
        cfg = load_config()
        rep = Reporter(db_path=cfg.database_url, trace_dir=cfg.trace_dir)

        if eval_run_id:
            eval_path = Path(cfg.evaluation.output_dir) / eval_run_id / "report.md"
            if not eval_path.is_file():
                console.print(f"[bold red]Evaluation report not found at {eval_path}[/bold red]")
                raise typer.Exit(code=1)
            console.print(eval_path.read_text(encoding="utf-8"))
            return

        if not investigation_id:
            # Pick latest investigation from database
            store = ExperimentStore(cfg.database_url)
            with store._get_connection() as conn:
                row = conn.execute("SELECT id FROM investigations ORDER BY created_at DESC LIMIT 1").fetchone()
                if row:
                    investigation_id = row["id"]
                else:
                    console.print("[yellow]No investigations found in database.[/yellow]")
                    return

        if trace:
            summary = rep.print_trace_summary(investigation_id)
            console.print(summary)
        else:
            md = rep.generate_investigation_markdown(investigation_id)
            console.print(md)

    except Exception as e:
        console.print(f"[bold red]Report generation failed:[/bold red] {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------
# Modal Smoke Test Command
# ---------------------------------------------------------------------
@modal_app_cmd.command("smoke-test")
def modal_smoke_test():
    """Run small remote execution test against Modal cloud."""
    try:
        cfg = load_config()
        backend = ModalExecutionBackend(cfg.execution)
        console.print("Submitting remote smoke test to Modal serverless infrastructure...")
        res = backend.smoke_test()
        console.print(Panel(
            json.dumps(res, indent=2),
            title="[bold green]Modal Remote Execution Smoke Test: HEALTHY[/bold green]",
            border_style="green"
        ))
    except Exception as e:
        console.print(f"[bold red]Modal Remote Smoke Test Failed:[/bold red] {e}")
        raise typer.Exit(code=1)


def main():
    app()


if __name__ == "__main__":
    main()
