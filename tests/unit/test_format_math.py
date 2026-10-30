"""Unit tests for the Terminal Math Formatter."""

import pytest
from mia.reporting.format_math import format_terminal_math


def test_format_delimiters():
    raw = r"\[ E = mc^2 \] and \( a^2 + b^2 = c^2 \) and $$ x = 0 $$"
    formatted = format_terminal_math(raw)
    assert r"\[" not in formatted
    assert r"\]" not in formatted
    assert r"\(" not in formatted
    assert r"\)" not in formatted
    assert "$$" not in formatted
    assert "mc²" in formatted
    assert "a² + b² = c²" in formatted


def test_format_fractions_and_roots():
    raw = r"\frac{1}{2} + \sqrt{x^2 + 1} = \frac{x + 1}{y - 1}"
    formatted = format_terminal_math(raw)
    assert "1/2" in formatted
    assert "√(x² + 1)" in formatted
    assert "(x + 1)/(y - 1)" in formatted


def test_format_greek_and_symbols():
    raw = r"\alpha + \beta \le \gamma \implies \Delta \ge 0 \land x \in \mathbb{R}"
    formatted = format_terminal_math(raw)
    assert "α + β ≤ γ ⇒ Δ ≥ 0" in formatted
    assert "∈ ℝ" in formatted


def test_format_subscripts_and_superscripts():
    raw = r"a_n = 5*a_{n-1} - 6*a_{n-2} and \sum_{k=1}^n k = \frac{n(n+1)}{2}"
    formatted = format_terminal_math(raw)
    assert "aₙ" in formatted
    assert "aₙ₋₁" in formatted
    assert "aₙ₋₂" in formatted
    assert "∑ₖ₌₁ⁿ" in formatted


def test_format_floor_and_congruence():
    raw = r"\lfloor k\alpha \rfloor \pmod{m} \equiv 0"
    formatted = format_terminal_math(raw)
    assert "⌊" in formatted
    assert "⌋" in formatted
    assert "(mod m)" in formatted
    assert "≡" in formatted
