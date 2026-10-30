"""Terminal Math Formatter.

Converts raw LaTeX expressions, equations, and mathematical markup into
clean, highly-readable Unicode text and structured Markdown for the terminal.
"""

from __future__ import annotations

import re


SUPERSCRIPTS = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
    "n": "ⁿ", "i": "ⁱ", "x": "ˣ", "y": "ʸ", "t": "ᵗ",
    "m": "ᵐ", "k": "ᵏ", "a": "ᵃ", "b": "ᵇ", "c": "ᶜ",
    "d": "ᵈ", "e": "ᵉ", "p": "ᵖ", "r": "ʳ", "s": "ˢ"
}

SUBSCRIPTS = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
    "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
    "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
    "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ",
    "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ",
    "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ",
    "v": "ᵥ", "x": "ₓ"
}

GREEK_AND_SYMBOLS = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\epsilon": "ε", r"\varepsilon": "ε", r"\zeta": "ζ", r"\eta": "η",
    r"\theta": "θ", r"\vartheta": "ϑ", r"\iota": "ι", r"\kappa": "κ",
    r"\lambda": "λ", r"\mu": "μ", r"\nu": "ν", r"\xi": "ξ",
    r"\pi": "π", r"\varpi": "ϖ", r"\rho": "ρ", r"\varrho": "ϱ",
    r"\sigma": "σ", r"\varsigma": "ς", r"\tau": "τ", r"\upsilon": "υ",
    r"\phi": "φ", r"\varphi": "ϕ", r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω",
    r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ", r"\Lambda": "Λ",
    r"\Xi": "Ξ", r"\Pi": "Π", r"\Sigma": "Σ", r"\Upsilon": "Υ",
    r"\Phi": "Φ", r"\Psi": "Ψ", r"\Omega": "Ω",
    r"\infty": "∞", r"\oo": "∞",
    r"\pm": "±", r"\mp": "∓",
    r"\times": "×", r"\cdot": "·", r"\div": "÷", r"\ast": "∗",
    r"\le": "≤", r"\leq": "≤", r"\ge": "≥", r"\geq": "≥",
    r"\neq": "≠", r"\ne": "≠",
    r"\approx": "≈", r"\equiv": "≡", r"\not\equiv": "≢", r"\sim": "∼", r"\cong": "≅",
    r"\in": "∈", r"\notin": "∉",
    r"\subset": "⊂", r"\subseteq": "⊆", r"\supset": "⊃", r"\supseteq": "⊇",
    r"\cup": "∪", r"\cap": "∩", r"\setminus": "∖", r"\emptyset": "∅",
    r"\forall": "∀", r"\exists": "∃", r"\nexists": "∄",
    r"\to": "→", r"\rightarrow": "→", r"\leftarrow": "←",
    r"\longrightarrow": "⟶", r"\longleftarrow": "⟵",
    r"\leftrightarrow": "↔", r"\longleftrightarrow": "⟷",
    r"\implies": "⇒", r"\iff": "⇔",
    r"\int": "∫", r"\sum": "∑", r"\prod": "∏",
    r"\partial": "∂", r"\nabla": "∇",
    r"\lfloor": "⌊", r"\rfloor": "⌋",
    r"\lceil": "⌈", r"\rceil": "⌉",
    r"\langle": "⟨", r"\rangle": "⟩",
    r"\mathbb{R}": "ℝ", r"\mathbb{Z}": "ℤ", r"\mathbb{Q}": "ℚ",
    r"\mathbb{N}": "ℕ", r"\mathbb{C}": "ℂ", r"\mathbb{P}": "ℙ",
    r"\mathbf{R}": "ℝ", r"\mathbf{Z}": "ℤ", r"\mathbf{Q}": "ℚ",
    r"\mathbf{N}": "ℕ", r"\mathbf{C}": "ℂ",
    r"\quad": "  ", r"\qquad": "    ", r"\,": " ", r"\:": " ", r"\;": " ",
    r"\cdots": "…", r"\ldots": "…", r"\dots": "…",
    r"\circ": "°", r"\triangle": "△", r"\angle": "∠", r"\perp": "⊥",
}


def format_terminal_math(text: str) -> str:
    """Transform LaTeX math notation into clean Unicode representations for terminal display."""
    if not text:
        return ""

    s = text

    # 1. Clean LaTeX display and inline math delimiters: \[ ... \], \( ... \), $$...$$, $...$
    s = re.sub(r"\\\[\s*(.*?)\s*\\\]", r"\n> \1\n", s, flags=re.DOTALL)
    s = re.sub(r"\\\(\s*(.*?)\s*\\\)", r"\1", s, flags=re.DOTALL)
    s = re.sub(r"\$\$\s*(.*?)\s*\$\$", r"\n> \1\n", s, flags=re.DOTALL)

    # 2. Strip \left and \right modifiers
    s = re.sub(r"\\left([\{\(\[\|])", r"\1", s)
    s = re.sub(r"\\right([\}\)\]\|])", r"\1", s)
    s = s.replace(r"\left.", "").replace(r"\right.", "")

    # 3. Handle \pmod{m} and \bmod
    s = re.sub(r"\\pmod\{([^{}]+)\}", r"(mod \1)", s)
    s = re.sub(r"\\pmod\s+([a-zA-Z0-9]+)", r"(mod \1)", s)
    s = s.replace(r"\bmod", "mod")

    # 4. Handle \text{...} and \mathrm{...}
    s = re.sub(r"\\(?:text|mathrm|mathbf|mathit)\{([^{}]+)\}", r"\1", s)

    # 5. Fractions: \frac{a}{b} -> (a)/(b) or a/b
    def _replace_frac(m: re.Match) -> str:
        num = m.group(1).strip()
        den = m.group(2).strip()
        if len(num) == 1 and len(den) == 1:
            return f"{num}/{den}"
        return f"({num})/({den})"

    # Handle fractions with possible nested braces
    frac_pattern = re.compile(r"(?:\\frac|\x0crac)\{((?:[^{}]|\{[^{}]*\})*)\}\{((?:[^{}]|\{[^{}]*\})*)\}")
    for _ in range(4):
        s = frac_pattern.sub(_replace_frac, s)

    # 6. Binomial coefficients: \binom{n}{k} -> C(n, k)
    binom_pattern = re.compile(r"\\binom\{([^{}]+)\}\{([^{}]+)\}")
    s = binom_pattern.sub(r"C(\1, \2)", s)

    # 7. Square roots: \sqrt{x} -> √(x)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"√(\1)", s)
    s = re.sub(r"\\sqrt\s*([a-zA-Z0-9])", r"√\1", s)

    # 8. Greek letters and standard mathematical symbols
    for tex, uni in GREEK_AND_SYMBOLS.items():
        s = s.replace(tex, uni)

    # 9. Superscripts: e.g. x^{2}, x^2, x^+, \mathbb{Z}^+ -> ℤ⁺
    def _replace_super(m: re.Match) -> str:
        content = m.group(1) or m.group(2)
        trans = "".join(SUPERSCRIPTS.get(c, c) for c in content)
        return trans

    s = re.sub(r"\^\{([0-9\+\-\(\)nixytmkabcdepr]+)\}", _replace_super, s)
    s = re.sub(r"\^([0-9nixytmkabcdepr\+\-])", _replace_super, s)

    # 10. Subscripts: e.g. a_{1}, a_n, a_{k=1}
    def _replace_sub(m: re.Match) -> str:
        content = m.group(1) or m.group(2)
        # Only substitute if all characters in subscript are convertible, or convert partial
        trans = "".join(SUBSCRIPTS.get(c, c) for c in content)
        return trans

    s = re.sub(r"_\{([0-9\+\-\=\(\)aehijklmnoprstuvx]+)\}", _replace_sub, s)
    s = re.sub(r"_([0-9aehijklmnoprstuvx])", _replace_sub, s)

    # 11. Clean up summation/product notation with subscripts/superscripts:
    # e.g. \sum_{k=1}^n -> ∑ₖ₌₁ⁿ
    s = re.sub(r"∑_([a-zA-Z0-9]+)\^([a-zA-Z0-9]+)", r"∑\1^\2", s)

    # 12. Strip leftover single dollar signs around isolated variables: $x$ -> x
    s = re.sub(r"(?<!\$)\$([^\$\n]+)\$(?!\$)", r"\1", s)

    # 13. Clean up backslashes before known function names
    s = re.sub(r"\\(sin|cos|tan|cot|sec|csc|arcsin|arccos|arctan|log|ln|exp|gcd|deg|min|max|det|lim|limsup|liminf|sup|inf)", r"\1", s)

    # 14. Clean double whitespace
    s = re.sub(r"  +", " ", s)

    return s
