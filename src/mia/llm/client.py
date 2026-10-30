"""Model Client interface and provider integrations for MIA."""

from __future__ import annotations

import abc
import json
import os
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from mia.config import ModelConfig
from mia.tools.registry import ToolDefinition


class ChatMessage(BaseModel):
    """Normalized chat message."""
    role: str  # "system", "user", "assistant", "tool"
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: Optional[int] = None
    total_tokens: int = 0


class ModelResponse(BaseModel):
    """Standardized response from an LLM call."""
    content: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_seconds: float = 0.0
    model: str = ""
    provider: str = ""


class BaseModelClient(abc.ABC):
    """Abstract interface for LLM communication."""

    def __init__(self, config: ModelConfig):
        self.config = config

    @abc.abstractmethod
    def generate(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None
    ) -> ModelResponse:
        """Generate response given chat messages and available tools."""
        pass


class OpenAICompatibleClient(BaseModelClient):
    """Client for OpenAI and OpenAI-compatible endpoints (Ollama, vLLM, OpenRouter)."""

    def generate(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None
    ) -> ModelResponse:
        api_key = os.environ.get(self.config.api_key_env_var)
        if not api_key:
            raise RuntimeError(
                f"Missing required API key for provider '{self.config.provider}'. "
                f"Please set environment variable '{self.config.api_key_env_var}'."
            )

        try:
            import openai
        except ImportError:
            raise RuntimeError("The 'openai' package is required for OpenAI provider. Run: pip install openai")

        client = openai.OpenAI(
            api_key=api_key,
            base_url=self.config.api_endpoint or None,
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries
        )

        formatted_messages = []
        for m in messages:
            msg_dict: Dict[str, Any] = {"role": m.role, "content": m.content or ""}
            if m.tool_calls:
                msg_dict["tool_calls"] = m.tool_calls
            if m.tool_call_id:
                msg_dict["tool_call_id"] = m.tool_call_id
            if m.name:
                msg_dict["name"] = m.name
            formatted_messages.append(msg_dict)

        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": formatted_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        if tools and self.config.structured_output:
            kwargs["tools"] = [t.to_openai_schema() for t in tools]

        start_time = time.monotonic()
        resp = client.chat.completions.create(**kwargs)
        duration = time.monotonic() - start_time

        choice = resp.choices[0].message
        content = choice.content

        tool_calls = []
        if choice.tool_calls:
            for tc in choice.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {"raw_arguments": tc.function.arguments}
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args
                })

        usage_data = TokenUsage(
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
            total_tokens=resp.usage.total_tokens if resp.usage else 0
        )

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage_data,
            latency_seconds=round(duration, 4),
            model=self.config.model,
            provider=self.config.provider
        )


class AnthropicClient(BaseModelClient):
    """Client for Anthropic Claude models."""

    def generate(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None
    ) -> ModelResponse:
        api_key = os.environ.get(self.config.api_key_env_var)
        if not api_key:
            raise RuntimeError(
                f"Missing required API key for provider '{self.config.provider}'. "
                f"Please set environment variable '{self.config.api_key_env_var}'."
            )

        import httpx

        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        # Separate system prompt from messages
        system_prompt = ""
        anthropic_msgs = []
        for m in messages:
            if m.role == "system":
                system_prompt += (m.content or "") + "\n"
            else:
                role = "assistant" if m.role == "assistant" else "user"
                anthropic_msgs.append({"role": role, "content": m.content or ""})

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": anthropic_msgs,
        }
        if system_prompt.strip():
            payload["system"] = system_prompt.strip()

        if tools and self.config.structured_output:
            payload["tools"] = [t.to_anthropic_schema() for t in tools]

        endpoint = self.config.api_endpoint or "https://api.anthropic.com/v1/messages"
        start_time = time.monotonic()
        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            resp = client.post(endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        duration = time.monotonic() - start_time

        content = ""
        tool_calls = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "arguments": block.get("input", {})
                })

        usage_raw = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_raw.get("input_tokens", 0),
            completion_tokens=usage_raw.get("output_tokens", 0),
            total_tokens=usage_raw.get("input_tokens", 0) + usage_raw.get("output_tokens", 0)
        )

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            latency_seconds=round(duration, 4),
            model=self.config.model,
            provider="anthropic"
        )


class GeminiClient(BaseModelClient):
    """Client for Google Gemini models via REST API."""

    def generate(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None
    ) -> ModelResponse:
        api_key = os.environ.get(self.config.api_key_env_var)
        if not api_key:
            raise RuntimeError(
                f"Missing required API key for provider '{self.config.provider}'. "
                f"Please set environment variable '{self.config.api_key_env_var}'."
            )

        import httpx

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.model}:generateContent?key={api_key}"

        contents = []
        for m in messages:
            role = "model" if m.role == "assistant" else "user"
            contents.append({
                "role": role,
                "parts": [{"text": m.content or ""}]
            })

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            }
        }

        start_time = time.monotonic()
        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        duration = time.monotonic() - start_time

        cand = data.get("candidates", [{}])[0]
        content_parts = cand.get("content", {}).get("parts", [{}])
        text = "".join(p.get("text", "") for p in content_parts)

        usage_raw = data.get("usageMetadata", {})
        usage = TokenUsage(
            prompt_tokens=usage_raw.get("promptTokenCount", 0),
            completion_tokens=usage_raw.get("candidatesTokenCount", 0),
            total_tokens=usage_raw.get("totalTokenCount", 0)
        )

        return ModelResponse(
            content=text,
            tool_calls=[],
            usage=usage,
            latency_seconds=round(duration, 4),
            model=self.config.model,
            provider="gemini"
        )


class MockDeterministicClient(BaseModelClient):
    """Deterministic, scriptable agent model client for testing, CI, and CPU demos without paid API keys.

    Simulates an intelligent mathematical researcher that:
    1. Formulates candidate hypotheses
    2. Issues relevant tool calls (counterexample search, symbolic limits, test cases)
    3. Detects contradictions and executes hypothesis revisions
    4. Triggers independent verification
    5. Returns structured solutions
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.call_count = 0

    def generate(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None
    ) -> ModelResponse:
        self.call_count += 1
        last_msg = messages[-1] if messages else ChatMessage(role="user", content="")
        context_str = "\n".join(f"{m.role}: {m.content or ''}" for m in messages)

        # Detect problem domain from conversation context
        is_euler = "n^2 + n + 41" in context_str or "euler" in context_str.lower()
        is_floor_sum = "floor" in context_str.lower() or "imo-2024" in context_str.lower() or r"\lfloor" in context_str or "⌊" in context_str or "alpha" in context_str.lower()
        is_exp_limit = "1/x^2" in context_str or "1/x**2" in context_str or "exponential-limit" in context_str
        is_cubic_sum = "x^3 + y^3 + z^3" in context_str or "cubic-sum" in context_str or "diophantine" in context_str.lower()
        is_limit = ("lim" in context_str.lower() and "sin(3" in context_str.lower()) or "controlled-analysis-limit" in context_str
        is_recurrence = "a_n" in context_str and "5*a" in context_str
        is_multivar = "f(x, y)" in context_str or "x^4 + y^4" in context_str

        tool_calls: List[Dict[str, Any]] = []
        content = ""

        # Step 1: Tool result received?
        if last_msg.role == "tool":
            tool_res_str = last_msg.content or ""
            tool_name = last_msg.name or ""

            # Case: IMO 2024 floor sum
            if is_floor_sum:
                if tool_name == "execute_experiment":
                    content = (
                        "CONTRADICTION DETECTED & EXPERIMENT OBSERVATION:\n"
                        "The computational experiment directly refutes Candidate Hypothesis H1 (all integers work):\n"
                        "1. Odd integers fail at n=2: S_2(1) = 3 % 2 = 1 != 0, and S_2(3) = 9 % 2 = 1 != 0.\n"
                        "   In general, for odd \\alpha = 2m + 1: S_2(\\alpha) = (2m+1) + (4m+2) = 6m + 3 = 2(3m+1) + 1 (always odd).\n"
                        "2. Non-integers fail at small n due to fractional discrepancy.\n"
                        "3. Even integers \\alpha = 2m (m \\in \\mathbb{Z}) succeed for all n: "
                        "S_n(2m) = \\sum_{k=1}^n 2mk = 2m * n(n+1)/2 = m * n(n+1), which is divisible by n for all n >= 1.\n"
                        "Revised hypothesis: The complete set of real numbers \\alpha satisfying the condition is all even integers \\alpha \\in 2\\mathbb{Z}.\n"
                        "Now executing independent verification of this result."
                    )
                    tool_calls.append({
                        "id": f"call_mock_{self.call_count}",
                        "name": "verify_solution",
                        "arguments": {
                            "problem_statement": "Determine all real numbers \\alpha such that \\sum_{k=1}^n \\lfloor k\\alpha \\rfloor is a multiple of n",
                            "candidate_solution": "All even integers: \\alpha \\in 2\\mathbb{Z} (i.e. \\alpha = 2m for m \\in \\mathbb{Z})",
                            "verification_level": "LEVEL_2"
                        }
                    })
                elif tool_name == "verify_solution":
                    content = (
                        "FINAL SOLUTION AND MATHEMATICAL REPORT:\n\n"
                        "1. Claimed Result:\n"
                        "The set of all real numbers \\alpha such that \\sum_{k=1}^n \\lfloor k\\alpha \\rfloor is a multiple of n "
                        "for every positive integer n is precisely the set of all even integers:\n"
                        "\\alpha \\in 2\\mathbb{Z} (i.e., \\alpha = 2m for m \\in \\mathbb{Z}).\n\n"
                        "2. Proven Facts and Mathematical Proof:\n"
                        "- Sufficiency: For any even integer \\alpha = 2m (m \\in \\mathbb{Z}), \\lfloor k\\alpha \\rfloor = 2mk. "
                        "The sum is S_n(2m) = \\sum_{k=1}^n 2mk = 2m \\frac{n(n+1)}{2} = m n(n+1). "
                        "Since m(n+1) is an integer, S_n(2m) is a multiple of n for all n >= 1.\n"
                        "- Necessity (Odd Integers): For any odd integer \\alpha = 2m + 1, testing n = 2 gives:\n"
                        "  S_2(\\alpha) = \\lfloor \\alpha \\rfloor + \\lfloor 2\\alpha \\rfloor = (2m+1) + (4m+2) = 6m+3 = 2(3m+1) + 1,\n"
                        "  which is odd and therefore never divisible by 2. Thus no odd integer works.\n"
                        "- Necessity (Non-integers): If \\alpha \\notin \\mathbb{Z}, let \\alpha = m + r with 0 < r < 1. "
                        "The average S_n/n cannot remain integral for all n without \\alpha \\in \\mathbb{Z}.\n"
                        "Hence, \\alpha must be an even integer.\n\n"
                        "3. Computational Evidence:\n"
                        "- Tested candidates in [-10, 10] across n \\in [1, 20].\n"
                        "- Every odd candidate failed at n=2; non-integers failed at small n.\n\n"
                        "4. Verification and Assumptions:\n"
                        "- Verified at Level 2 (Symbolic and exact divisibility verification).\n"
                        "- Confidence: 1.0 (Exact mathematical certainty). No unverified steps remain."
                    )
                else:
                    content = f"Observed experiment results: {tool_res_str[:150]}"

            # Case: Indeterminate exponential limit
            elif is_exp_limit:
                if tool_name == "numerical_evaluate":
                    content = (
                        "NUMERICAL FEEDBACK OBSERVED:\n"
                        "Numerical evaluation confirms monotonic convergence towards 0.84648172489...\n"
                        "Notice that exp(-1/6) = 0.84648172489..., matching the numerical evidence to high precision.\n"
                        "Now executing symbolic series expansion of ln(f(x)) around x=0 to establish the exact limit and determine the next nonzero asymptotic term."
                    )
                    tool_calls.append({
                        "id": f"call_mock_{self.call_count}",
                        "name": "symbolic_compute",
                        "arguments": {
                            "operation": "series",
                            "expression": "log(sin(x)/x) / x**2",
                            "variable": "x",
                            "point": "0"
                        }
                    })
                elif tool_name == "symbolic_compute":
                    content = (
                        "FINAL SOLUTION AND MATHEMATICAL REPORT:\n\n"
                        "1. Claimed Result:\n"
                        "The exact value of the limit is L = e^{-1/6}.\n"
                        "The asymptotic expansion as x \\to 0 is:\n"
                        "f(x) = e^{-1/6} \\left( 1 - \\frac{x^2}{180} + O(x^4) \\right).\n\n"
                        "2. Proven Facts and Mathematical Derivation:\n"
                        "- As x \\to 0, \\sin(x)/x = 1 - x^2/6 + x^4/120 - O(x^6).\n"
                        "- Taking the natural logarithm: \\ln(1 - u) = -u - u^2/2 - O(u^3).\n"
                        "- Substituting u = x^2/6 - x^4/120 yields \\ln(\\sin(x)/x) = -x^2/6 - x^4/180 - O(x^6).\n"
                        "- Dividing by x^2: \\ln f(x) = -1/6 - x^2/180 - O(x^4).\n"
                        "- Exponentiating: f(x) = \\exp(-1/6 - x^2/180) = e^{-1/6} (1 - x^2/180 + O(x^4)).\n\n"
                        "3. Computational Evidence:\n"
                        "- Numerical evaluation confirmed values matching e^{-1/6} to 6 decimal places.\n"
                        "- Symbolic series verified via SymPy algebraic engine.\n\n"
                        "4. Verification Status: Level 2 verified via exact symbolic derivation."
                    )
                else:
                    content = f"Observed experiment results: {tool_res_str[:150]}"

            # Case: Diophantine cubic sum
            elif is_cubic_sum:
                if tool_name == "execute_experiment":
                    content = (
                        "HYPOTHESIS CONTRADICTED & EXPERIMENT FEEDBACK:\n"
                        "The initial conjecture of uniqueness is REFUTED! The computational experiment discovered 7 solutions:\n"
                        "[(1, 6, 8, 9), (2, 12, 16, 18), (3, 4, 5, 6), (3, 10, 18, 19), (6, 8, 10, 12), (7, 14, 17, 20), (9, 12, 15, 18)].\n"
                        "Analysis of discovered solutions:\n"
                        "- 4 primitive quadruples: (1, 6, 8, 9), (3, 4, 5, 6), (3, 10, 18, 19), (7, 14, 17, 20)\n"
                        "- 3 derived multiples: (2, 12, 16, 18) = 2*(1, 6, 8, 9), (6, 8, 10, 12) = 2*(3, 4, 5, 6), (9, 12, 15, 18) = 3*(3, 4, 5, 6)\n"
                        "Revised hypothesis: There are exactly 7 integer quadruples (x, y, z, w) satisfying x^3 + y^3 + z^3 = w^3 with 1 <= x < y < z < w <= 20.\n"
                        "Now executing independent verification of the 7 solutions."
                    )
                    tool_calls.append({
                        "id": f"call_mock_{self.call_count}",
                        "name": "verify_solution",
                        "arguments": {
                            "problem_statement": "Find all positive integer quadruples x^3 + y^3 + z^3 = w^3 with 1 <= x < y < z < w <= 20",
                            "candidate_solution": (
                                "7 solutions: (1, 6, 8, 9), (2, 12, 16, 18), (3, 4, 5, 6), "
                                "(3, 10, 18, 19), (6, 8, 10, 12), (7, 14, 17, 20), (9, 12, 15, 18)"
                            ),
                            "verification_level": "LEVEL_2"
                        }
                    })
                elif tool_name == "verify_solution":
                    content = (
                        "FINAL SOLUTION AND MATHEMATICAL REPORT:\n\n"
                        "1. Claimed Result:\n"
                        "There are exactly 7 positive integer quadruples satisfying x^3 + y^3 + z^3 = w^3 "
                        "with 1 \\le x < y < z < w \\le 20:\n"
                        "(1, 6, 8, 9), (2, 12, 16, 18), (3, 4, 5, 6), (3, 10, 18, 19), "
                        "(6, 8, 10, 12), (7, 14, 17, 20), (9, 12, 15, 18).\n\n"
                        "2. Proven Facts and Classification:\n"
                        "- Primitive solutions:\n"
                        "  * 1^3 + 6^3 + 8^3 = 1 + 216 + 512 = 729 = 9^3\n"
                        "  * 3^3 + 4^3 + 5^3 = 27 + 64 + 125 = 216 = 6^3\n"
                        "  * 3^3 + 10^3 + 18^3 = 27 + 1000 + 5832 = 6859 = 19^3\n"
                        "  * 7^3 + 14^3 + 17^3 = 343 + 2744 + 4913 = 8000 = 20^3\n"
                        "- Homothetic scaled solutions:\n"
                        "  * (2, 12, 16, 18) = 2 \\times (1, 6, 8, 9)\n"
                        "  * (6, 8, 10, 12) = 2 \\times (3, 4, 5, 6)\n"
                        "  * (9, 12, 15, 18) = 3 \\times (3, 4, 5, 6)\n\n"
                        "3. Computational Completeness:\n"
                        "- Exhaustive search over all C(20, 4) = 4845 combinations guarantees completeness in this domain.\n"
                        "- Verification Level: Level 2 verified."
                    )
                else:
                    content = f"Observed experiment results: {tool_res_str[:150]}"

            # Case: Counterexample search returned
            elif tool_name == "search_counterexamples":
                content = (
                    "HYPOTHESIS CONTRADICTED: The computational experiment revealed that for n = 40, "
                    "P(40) = 40^2 + 40 + 41 = 1681 = 41^2, which is composite.\n"
                    "Revised hypothesis: P(n) is NOT prime for all positive integers n. "
                    "The smallest positive integer n for which P(n) is composite is n = 40, "
                    "with prime factorization 41^2.\n"
                    "Now executing independent verification of this counterexample."
                )
                tool_calls.append({
                    "id": f"call_mock_{self.call_count}",
                    "name": "verify_solution",
                    "arguments": {
                        "problem_statement": "Investigate P(n) = n^2 + n + 41",
                        "candidate_solution": "n = 40 is composite: P(40) = 1681 = 41^2",
                        "verification_level": "LEVEL_2"
                    }
                })

            # Case: Verification tool returned
            elif tool_name == "verify_solution":
                content = (
                    "FINAL SOLUTION AND MATHEMATICAL REPORT:\n\n"
                    "1. Claimed Result:\n"
                    "The polynomial P(n) = n^2 + n + 41 is NOT prime for all positive integers n.\n"
                    "The smallest positive integer n for which P(n) is composite is n = 40.\n\n"
                    "2. Proven Facts and Computational Evidence:\n"
                    "- For 1 <= n <= 39, P(n) generates 39 distinct primes.\n"
                    "- At n = 40: P(40) = 40*(40 + 1) + 41 = 40*41 + 41 = 41*(40 + 1) = 41^2 = 1681.\n"
                    "- 1681 has factors 41 * 41, definitively refuting universal primality.\n\n"
                    "3. Verification:\n"
                    "- Verified at Level 2 (exact integer arithmetic and prime factorization).\n"
                    "- Confidence: 1.0 (Exact mathematical certainty).\n"
                    "4. Assumptions and Limitations:\n"
                    "- No unverified deductive steps remain."
                )

            # Case: Symbolic compute returned
            elif tool_name == "symbolic_compute":
                if "81/40" in tool_res_str or "2.025" in tool_res_str:
                    content = (
                        "FINAL SOLUTION AND MATHEMATICAL REPORT:\n\n"
                        "1. Claimed Result:\n"
                        "The exact value of the limit L is 81/40.\n\n"
                        "2. Mathematical Proof:\n"
                        "By Maclaurin expansion of sin(u) = u - u^3/6 + u^5/120 + O(u^7).\n"
                        "Substituting u = 3x:\n"
                        "sin(3x) = 3x - 27x^3/6 + 243x^5/120 + O(x^7) = 3x - (9/2)x^3 + (81/40)x^5 + O(x^7).\n"
                        "Numerator = (81/40)x^5 + O(x^7).\n"
                        "Dividing by x^5 as x -> 0 gives L = 81/40.\n\n"
                        "3. Verification Status: Level 2 verified via exact symbolic limit."
                    )
                else:
                    content = (
                        "Symbolic computation complete. Synthesizing final mathematical solution based on exact results:\n"
                        f"{tool_res_str[:200]}"
                    )

            else:
                content = (
                    f"Observed experiment results: {tool_res_str[:150]}... "
                    "Synthesizing final mathematical conclusion based on accumulated evidence."
                )

        # Initial prompt / first step
        else:
            if is_floor_sum:
                content = (
                    "INITIAL ASSESSMENT & CANDIDATE HYPOTHESIS:\n"
                    "We investigate whether there exist real numbers \\alpha such that for every positive integer n, "
                    "S_n(\\alpha) = \\sum_{k=1}^n \\lfloor k\\alpha \\rfloor is a multiple of n.\n"
                    "Candidate Hypothesis H1: All integers \\alpha \\in \\mathbb{Z} satisfy this divisibility property for all n.\n"
                    "Uncertainty / Failure Modes: Odd integers might fail for even n; non-integers might fail as fractional parts accumulate.\n"
                    "Action: Execute a computational experiment testing integer candidates {-2, -1, 1, 2, 3, 4} and rational candidates {1.5, 2.5} "
                    "across n \\in {1, 2, 3, 4, 5, 6}."
                )
                tool_calls.append({
                    "id": f"call_mock_{self.call_count}",
                    "name": "execute_experiment",
                    "arguments": {
                        "description": "Test divisibility of sum_{k=1}^n floor(k*alpha) by n for integer and rational candidates",
                        "code": (
                            "import math\n"
                            "def test_alpha(a, max_n=6):\n"
                            "    return [(n, sum(math.floor(k*a) for k in range(1, n+1))) "
                            "for n in range(1, max_n+1) if sum(math.floor(k*a) for k in range(1, n+1)) % n != 0]\n"
                            "for a in [1, 2, 3, 4, -1, -2, 1.5, 2.5]:\n"
                            "    fails = test_alpha(a)\n"
                            "    print(f'alpha={a}: ' + ('VALID' if not fails else f'FAILS at n={[f[0] for f in fails]}'))\n"
                        )
                    }
                })

            elif is_exp_limit:
                content = (
                    "INITIAL FORMALIZATION & CANDIDATE HYPOTHESIS:\n"
                    "We evaluate the asymptotic behavior of f(x) = (\\frac{\\sin x}{x})^{1/x^2} as x \\to 0.\n"
                    "This is an indeterminate form of type 1^\\infty. In logarithm: \\ln f(x) = \\frac{\\ln(\\sin x / x)}{x^2}.\n"
                    "Candidate Hypothesis H1: The limit L exists and equals e^{-1/6} \\approx 0.84648172.\n"
                    "Action: Execute numerical evaluation at x \\in {0.1, 0.01, 0.001} to observe convergence behavior."
                )
                tool_calls.append({
                    "id": f"call_mock_{self.call_count}",
                    "name": "numerical_evaluate",
                    "arguments": {
                        "expression": "(sin(x)/x)**(1/x**2)",
                        "points": [0.1, 0.01, 0.001],
                        "variable": "x"
                    }
                })

            elif is_cubic_sum:
                content = (
                    "INITIAL FORMALIZATION & CANDIDATE HYPOTHESIS:\n"
                    "We seek positive integer quadruples 1 \\le x < y < z < w \\le 20 such that x^3 + y^3 + z^3 = w^3.\n"
                    "Candidate Hypothesis H1: The quadruple (3, 4, 5, 6) is a valid solution since 3^3 + 4^3 + 5^3 = 27 + 64 + 125 = 216 = 6^3.\n"
                    "Action: Execute programmatic exhaustive search in the execution backend across [1, 20] to test uniqueness."
                )
                tool_calls.append({
                    "id": f"call_mock_{self.call_count}",
                    "name": "execute_experiment",
                    "arguments": {
                        "description": "Exhaustive search for x^3 + y^3 + z^3 = w^3 with 1 <= x < y < z < w <= 20",
                        "code": (
                            "sols = []\n"
                            "for x in range(1, 21):\n"
                            "    for y in range(x+1, 21):\n"
                            "        for z in range(y+1, 21):\n"
                            "            for w in range(z+1, 21):\n"
                            "                if x**3 + y**3 + z**3 == w**3:\n"
                            "                    sols.append((x, y, z, w))\n"
                            "print(f'Found solutions: {sols}. Total: {len(sols)}')\n"
                        )
                    }
                })

            elif is_euler:
                content = (
                    "INITIAL ASSESSMENT AND CANDIDATE CONJECTURE:\n"
                    "We investigate whether P(n) = n^2 + n + 41 is prime for all positive integers n.\n"
                    "Initial testing for n in {1, 2, 3} yields 43, 47, 53 (all prime).\n"
                    "Candidate Hypothesis H1: P(n) is prime for all positive integers n.\n"
                    "Potential Failure Mode: There may exist a counterexample n where P(n) is composite.\n"
                    "Executing automated counterexample search across n in [1, 50] to test H1 rigorously."
                )
                tool_calls.append({
                    "id": f"call_mock_{self.call_count}",
                    "name": "search_counterexamples",
                    "arguments": {
                        "conjecture_type": "primality",
                        "expression": "n^2 + n + 41",
                        "variable": "n",
                        "search_min": 1,
                        "search_max": 50
                    }
                })

            elif is_limit:
                content = (
                    "INITIAL FORMALIZATION:\n"
                    "We need to evaluate the limit L = lim_{x->0} (sin(3x) - 3x + (9/2)x^3)/x^5.\n"
                    "The form is 0/0. Applying Taylor series expansion around x=0 to order 5."
                )
                tool_calls.append({
                    "id": f"call_mock_{self.call_count}",
                    "name": "symbolic_compute",
                    "arguments": {
                        "operation": "limits",
                        "expression": "(sin(3*x) - 3*x + (9/2)*x^3) / x^5",
                        "variable": "x",
                        "point": "0"
                    }
                })

            elif is_recurrence:
                content = (
                    "INITIAL FORMALIZATION:\n"
                    "Second-order linear recurrence a_n = 5*a_{n-1} - 6*a_{n-2} with a_0=1, a_1=3.\n"
                    "Solving characteristic equation r^2 - 5r + 6 = 0."
                )
                tool_calls.append({
                    "id": f"call_mock_{self.call_count}",
                    "name": "symbolic_compute",
                    "arguments": {
                        "operation": "solve",
                        "expression": "r^2 - 5*r + 6",
                        "variable": "r"
                    }
                })

            else:
                content = (
                    "Formalizing mathematical problem statement and identifying domain, definitions, and goals.\n"
                    "Generating initial test cases to observe mathematical invariants."
                )
                tool_calls.append({
                    "id": f"call_mock_{self.call_count}",
                    "name": "generate_test_cases",
                    "arguments": {
                        "domain": "positive_integers",
                        "count": 10
                    }
                })

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            usage=TokenUsage(prompt_tokens=150, completion_tokens=100, total_tokens=250),
            latency_seconds=0.05,
            model=self.config.model,
            provider="mock"
        )


class ModalHuggingFaceClient(BaseModelClient):
    """Client for running open-weights Hugging Face LLMs on Modal GPUs."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        import sys
        from pathlib import Path
        cwd_str = str(Path.cwd().resolve())
        if cwd_str not in sys.path:
            sys.path.insert(0, cwd_str)

    def generate(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None
    ) -> ModelResponse:
        import re
        import uuid
        import modal_app

        formatted_msgs = []
        tools_instructions = ""
        if tools:
            tools_list = [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters_schema
                }
                for t in tools
            ]
            tools_instructions = (
                "\n\nYou have access to the following mathematical tools:\n"
                f"{json.dumps(tools_list, indent=2)}\n\n"
                "To invoke a tool, output a JSON block in the format:\n"
                "<tool_call>\n"
                "{\"name\": \"<tool_name>\", \"arguments\": {<args>}}\n"
                "</tool_call>\n"
            )

        for m in messages:
            content = m.content or ""
            if m.role == "system":
                content += tools_instructions
            elif m.role == "tool":
                content = f"[Tool '{m.name}' Result]: {content}"
            role = "assistant" if m.role == "assistant" else ("user" if m.role == "user" else "system")
            formatted_msgs.append({"role": role, "content": content})

        # Run remote GPU inference on Modal
        try:
            import modal
            llm_cls = modal.Cls.from_name("mia-executor", "ModalHuggingFaceLLM")
            llm_engine = llm_cls(model_id=self.config.model)
            res = llm_engine.generate.remote(
                messages=formatted_msgs,
                max_new_tokens=min(self.config.max_tokens, 2048),
                temperature=self.config.temperature
            )
        except Exception:
            with modal_app.app.run():
                llm_engine = modal_app.ModalHuggingFaceLLM(model_id=self.config.model)
                res = llm_engine.generate.remote(
                    messages=formatted_msgs,
                    max_new_tokens=min(self.config.max_tokens, 2048),
                    temperature=self.config.temperature
                )

        raw_content = res.get("content", "")

        # Extract tool calls if model produced any
        tool_calls = []
        pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        matches = pattern.findall(raw_content)
        if not matches:
            json_blocks = re.findall(r"```(?:json)?\s*(\{\s*\"(?:name|tool)\".*?\})\s*```", raw_content, re.DOTALL)
            matches = json_blocks

        for m_str in matches:
            try:
                parsed = json.loads(m_str.strip())
                name = parsed.get("name") or parsed.get("tool")
                args = parsed.get("arguments") or parsed.get("args") or {}
                if name:
                    tool_calls.append({
                        "id": f"call_hf_{uuid.uuid4().hex[:6]}",
                        "name": name,
                        "arguments": args
                    })
            except Exception:
                pass

        return ModelResponse(
            content=raw_content,
            tool_calls=tool_calls,
            usage=TokenUsage(
                prompt_tokens=res.get("prompt_tokens", 0),
                completion_tokens=res.get("completion_tokens", 0),
                total_tokens=res.get("prompt_tokens", 0) + res.get("completion_tokens", 0)
            ),
            latency_seconds=res.get("latency_seconds", 0.0),
            model=self.config.model,
            provider="modal-huggingface"
        )


class LocalHuggingFaceClient(BaseModelClient):
    """Client for running Hugging Face models locally using transformers."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._pipe = None

    def generate(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None
    ) -> ModelResponse:
        try:
            import torch
            from transformers import pipeline
        except ImportError:
            raise RuntimeError("Install torch and transformers to run local Hugging Face models: pip install torch transformers")

        if self._pipe is None:
            self._pipe = pipeline(
                "text-generation",
                model=self.config.model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )

        formatted_msgs = [{"role": m.role, "content": m.content or ""} for m in messages]
        start_time = time.monotonic()
        out = self._pipe(formatted_msgs, max_new_tokens=min(self.config.max_tokens, 2048), temperature=self.config.temperature)
        duration = time.monotonic() - start_time

        gen_text = out[0]["generated_text"][-1]["content"] if out else ""
        return ModelResponse(
            content=gen_text,
            tool_calls=[],
            usage=TokenUsage(prompt_tokens=100, completion_tokens=100, total_tokens=200),
            latency_seconds=round(duration, 4),
            model=self.config.model,
            provider="huggingface"
        )


def get_model_client(config: ModelConfig) -> BaseModelClient:
    """Factory to instantiate the configured model client."""
    if config.provider == "mock":
        return MockDeterministicClient(config)
    elif config.provider == "modal-huggingface":
        return ModalHuggingFaceClient(config)
    elif config.provider == "huggingface":
        return LocalHuggingFaceClient(config)
    elif config.provider == "openai":
        return OpenAICompatibleClient(config)
    elif config.provider == "anthropic":
        return AnthropicClient(config)
    elif config.provider == "gemini":
        return GeminiClient(config)
    else:
        raise ValueError(
            f"Unsupported provider: '{config.provider}'. Supported: 'mock', 'modal-huggingface', 'huggingface', 'openai', 'anthropic', 'gemini'."
        )
