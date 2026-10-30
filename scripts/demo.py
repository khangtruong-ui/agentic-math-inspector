#!/usr/bin/env python3
"""User-Facing Mathematical Investigation Demonstration.

Reproduces the complete autonomous cycle on CPU:
Problem -> Initial Hypothesis -> Real Tool Experiment -> Contradiction Observed ->
Hypothesis Revision -> Independent Verification -> Final Mathematical Report.
"""

import sys
from rich.console import Console
from rich.panel import Panel

from mia.agent.loop import MathematicalAgent
from mia.config import AppConfig
from mia.sources import get_source

console = Console()


def run_demo():
    console.print(Panel(
        "[bold cyan]Mathematical Investigation Agent (MIA)[/bold cyan]\n"
        "[white]Autonomous Mathematical Exploration, Experimentation, and Verification Demonstration[/white]",
        border_style="cyan"
    ))

    # Configure offline deterministic investigation
    config = AppConfig()
    config.model.provider = "mock"
    config.execution.backend = "local"

    source = get_source("controlled")
    problem = source.get_problem("controlled-euler-primality")
    if not problem:
        console.print("[red]Problem not found.[/red]")
        sys.exit(1)

    console.print(f"[bold]Problem Statement:[/bold]\n{problem.statement}\n")

    agent = MathematicalAgent(config)
    report = agent.investigate(problem.to_agent_view())

    console.print("\n" + "=" * 60)
    console.print(f"[bold green]Demonstration Complete![/bold green] (Investigation ID: {report.investigation_id})")
    console.print(f"Total Steps: {report.total_steps} | Experiments Run: {report.total_experiments} | Duration: {report.wall_clock_seconds:.2f}s\n")

    console.print("[bold]Hypothesis Evolution:[/bold]")
    for idx, h in enumerate(report.hypotheses_evaluated, 1):
        color = "green" if h.status.value in ("CONFIRMED", "ACTIVE") else "red"
        console.print(f"  {idx}. [{color}][{h.status.value}][/{color}] {h.text}")

    if report.candidate_solution:
        console.print("\n[bold]Final Solution & Report:[/bold]")
        console.print(report.candidate_solution)

    if report.verification:
        v = report.verification
        v_color = "green" if v.status.value == "VERIFIED" else "yellow"
        console.print(f"\n[bold]Independent Verification:[/bold] [{v_color}]{v.status.value}[/{v_color}] (Confidence: {v.confidence:.1%})")
        for ev in v.evidence:
            console.print(f"  ✓ {ev}")


if __name__ == "__main__":
    run_demo()
