"""Unit tests for the Typer CLI commands."""

from pathlib import Path
import pytest
from typer.testing import CliRunner
from mia.cli import app

runner = CliRunner()


def test_cli_help():
    """Verify that `mia --help` executes cleanly with code 0."""
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "Mathematical Investigation Agent" in res.stdout


def test_cli_config_validate():
    """Verify `mia config validate` prints effective config."""
    res = runner.invoke(app, ["config", "validate"])
    assert res.exit_code == 0
    assert "Effective MIA Configuration" in res.stdout


def test_cli_sources_list():
    """Verify `mia sources list` lists registered sources."""
    res = runner.invoke(app, ["sources", "list"])
    assert res.exit_code == 0
    assert "controlled" in res.stdout
    assert "imo" in res.stdout


def test_cli_problems_list():
    """Verify `mia problems list` lists problems."""
    res = runner.invoke(app, ["problems", "list", "--source", "controlled"])
    assert res.exit_code == 0
    assert "controlled-euler-primality" in res.stdout


def test_cli_solve_demo():
    """Verify `mia solve --demo` runs end-to-end investigation."""
    res = runner.invoke(app, ["solve", "controlled-euler-primality", "--demo"])
    assert res.exit_code == 0
    assert "INVESTIGATION COMPLETE" in res.stdout
    assert "controlled-euler-primality" in res.stdout


def test_cli_experiments_list():
    """Verify `mia experiments` lists recorded experiments."""
    res = runner.invoke(app, ["experiments", "--limit", "5"])
    assert res.exit_code == 0


def test_cli_report():
    """Verify `mia report` generates report for existing investigation."""
    res = runner.invoke(app, ["report"])
    assert res.exit_code == 0
