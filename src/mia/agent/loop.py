"""Autonomous Mathematical Investigation Agent Loop."""

from __future__ import annotations

import datetime
import hashlib
import json
import uuid
from typing import Any, Dict, List, Optional

from mia.agent.prompts import INITIAL_ASSESSMENT_PROMPT, SYSTEM_PROMPT
from mia.agent.state import ActiveInvestigationState
from mia.config import AppConfig
from mia.execution.base import BaseExecutionBackend
from mia.execution import get_execution_backend
from mia.llm.client import BaseModelClient, ChatMessage, get_model_client
from mia.models import (
    AgentProblemView,
    ExperimentExecutionResult,
    HypothesisStatus,
    InvestigationReport,
    InvestigationStatus,
    VerificationLevel,
    VerificationResult,
    VerificationStatus,
)
from mia.storage.experiment_store import ExperimentStore
from mia.storage.traces import TraceLogger
from mia.tools.registry import ToolRegistry
from mia.tools import create_default_registry


class MathematicalAgent:
    """Core autonomous agent managing mathematical investigation, experiments, and verification."""

    def __init__(
        self,
        config: AppConfig,
        model_client: Optional[BaseModelClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        execution_backend: Optional[BaseExecutionBackend] = None,
        store: Optional[ExperimentStore] = None,
        trace_logger: Optional[TraceLogger] = None
    ):
        self.config = config
        self.client = model_client or get_model_client(config.model)
        self.tools = tool_registry or create_default_registry()
        self.backend = execution_backend or get_execution_backend(config.execution)
        self.store = store or ExperimentStore(config.database_url)
        self.traces = trace_logger or TraceLogger(config.trace_dir)

    def investigate(
        self,
        problem: AgentProblemView,
        investigation_id: Optional[str] = None,
        max_steps_override: Optional[int] = None
    ) -> InvestigationReport:
        """Run the autonomous mathematical investigation cycle."""
        inv_id = investigation_id or f"inv-{uuid.uuid4().hex[:8]}"
        state = ActiveInvestigationState(investigation_id=inv_id, problem=problem)

        max_steps = max_steps_override or self.config.agent.max_steps

        # 1. Initialize persistent storage and trace
        self.store.create_investigation(inv_id, problem)
        self.traces.log_event(inv_id, "investigation_started", {
            "problem_id": problem.problem_id,
            "subject": problem.subject,
            "statement": problem.statement,
            "config": self.config.redact_secrets()
        })

        # 2. Build initial conversation history
        messages: List[ChatMessage] = [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=INITIAL_ASSESSMENT_PROMPT.format(
                    statement=problem.statement,
                    subject=problem.subject or "General Mathematics",
                    difficulty=problem.difficulty or "Unknown"
                )
            )
        ]

        state.status = InvestigationStatus.EXPLORING
        self.store.update_investigation(inv_id, status=state.status.value)

        # 3. Enter main investigation loop
        while state.step_count < max_steps:
            state.step_count += 1

            # Check hard resource limits
            if state.elapsed_seconds() >= self.config.agent.max_wall_clock_seconds:
                state.status = InvestigationStatus.EXHAUSTED
                state.termination_reason = f"Exceeded max wall-clock budget ({self.config.agent.max_wall_clock_seconds}s)"
                break

            if state.model_calls_count >= self.config.agent.max_model_calls:
                state.status = InvestigationStatus.EXHAUSTED
                state.termination_reason = f"Exceeded model call budget ({self.config.agent.max_model_calls})"
                break

            if state.consecutive_nonproductive_steps >= self.config.agent.max_nonproductive_steps:
                state.status = InvestigationStatus.EXHAUSTED
                state.termination_reason = "Halting due to repeated nonproductive steps"
                break

            # Execute Model Call
            state.model_calls_count += 1
            self.traces.log_event(inv_id, "model_request", {
                "step": state.step_count,
                "history_length": len(messages)
            })

            try:
                response = self.client.generate(
                    messages=messages,
                    tools=self.tools.list_tools()
                )
            except Exception as e:
                self.traces.log_event(inv_id, "model_error", {"error": str(e), "step": state.step_count})
                state.status = InvestigationStatus.FAILED
                state.termination_reason = f"Model call failed: {e}"
                break

            self.traces.log_event(inv_id, "model_response", {
                "step": state.step_count,
                "content": response.content,
                "tool_calls_count": len(response.tool_calls),
                "tokens": response.usage.model_dump(),
                "latency": response.latency_seconds
            })

            # Append assistant response to history
            assistant_msg = ChatMessage(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls if response.tool_calls else None
            )
            messages.append(assistant_msg)

            # Extract any newly stated hypotheses or conjectures
            if response.content:
                self._extract_hypotheses_and_evidence(state, response.content)

            # If no tool calls, model provided final reasoning / answer
            if not response.tool_calls:
                state.consecutive_nonproductive_steps += 1
                state.candidate_solution = response.content

                # Check if we should automatically trigger independent verification
                if not state.verification_result and state.candidate_solution:
                    ver_tool_res = self.tools.execute(
                        "verify_solution",
                        {
                            "problem_statement": problem.statement,
                            "candidate_solution": state.candidate_solution,
                            "verification_level": self.config.agent.required_verification_level
                        }
                    )
                    if ver_tool_res.get("success"):
                        res_data = ver_tool_res["result"]
                        v_result = VerificationResult(
                            level=VerificationLevel(res_data["verification_level"]),
                            status=VerificationStatus(res_data["status"]),
                            evidence=res_data["evidence"],
                            unverified_steps=res_data["unverified_steps"],
                            assumptions=res_data["assumptions"],
                            counterexample=res_data["counterexample"],
                            confidence=res_data["confidence"],
                            summary=res_data["summary"]
                        )
                        state.verification_result = v_result
                        self.store.record_verification(inv_id, v_result)

                if state.verification_result and state.verification_result.status == VerificationStatus.VERIFIED:
                    state.status = InvestigationStatus.SOLVED
                    state.termination_reason = "Verified mathematical solution produced"
                    break
                elif state.step_count >= 2:
                    # Provide partial solution if no more tool calls
                    state.status = InvestigationStatus.PARTIALLY_SOLVED
                    state.termination_reason = "Model concluded investigation with candidate result"
                    break
                continue

            # Process tool calls
            state.consecutive_nonproductive_steps = 0
            for tc in response.tool_calls:
                state.tool_calls_count += 1
                tool_name = tc.get("name", "")
                args = tc.get("arguments", {})
                call_id = tc.get("id", f"call_{uuid.uuid4().hex[:6]}")

                # Detect repeated identical actions
                sig = self._action_signature(tool_name, args)
                if state.is_action_repeated(sig, max_allowed=self.config.agent.max_repeated_actions):
                    messages.append(ChatMessage(
                        role="tool",
                        name=tool_name,
                        tool_call_id=call_id,
                        content=(
                            f"[MIA REPETITION DETECTED]: Tool '{tool_name}' was invoked with identical arguments "
                            f"{self.config.agent.max_repeated_actions} times without new evidence. "
                            "You must revise your hypothesis or try an alternative mathematical method."
                        )
                    ))
                    continue
                state.action_hashes.append(sig)

                # Prepare execution context
                context = {
                    "backend": self.backend,
                    "state": {
                        "investigation_id": inv_id,
                        "problem_id": problem.problem_id,
                        "status": state.status.value,
                        "step_count": state.step_count,
                        "active_hypothesis": state.active_hypothesis
                    },
                    "experiments": [e.model_dump() for e in state.experiments_history],
                    "hypotheses": [h.model_dump() for h in state.hypotheses_history]
                }

                # Execute tool
                tool_exec_res = self.tools.execute(tool_name, args, context=context)

                if not tool_exec_res.get("success"):
                    state.failed_tool_calls_count += 1
                    err_msg = tool_exec_res.get("error", "Unknown tool error")
                    self.traces.log_event(inv_id, "tool_failure", {"tool": tool_name, "error": err_msg})
                    messages.append(ChatMessage(
                        role="tool",
                        name=tool_name,
                        tool_call_id=call_id,
                        content=f"Tool execution failed: {err_msg}"
                    ))
                    continue

                tool_result_payload = tool_exec_res.get("result", {})

                # Record experiment in state and SQLite
                exp_record = self._convert_tool_result_to_experiment(
                    tool_name=tool_name,
                    args=args,
                    result_payload=tool_result_payload
                )
                state.experiments_history.append(exp_record)
                self.store.record_experiment(inv_id, tool_name, args, exp_record)
                self.traces.log_event(inv_id, "tool_success", {
                    "tool": tool_name,
                    "result": tool_result_payload
                })

                # Check for contradiction / counterexample discovery
                if tool_name == "search_counterexamples" and tool_result_payload.get("found_counterexample"):
                    ce = tool_result_payload.get("first_counterexample")
                    state.status = InvestigationStatus.REVISING
                    # Contradiction found -> force hypothesis revision
                    if state.active_hypothesis:
                        for h in state.hypotheses_history:
                            if h.status == HypothesisStatus.ACTIVE:
                                h.status = HypothesisStatus.REFUTED
                                h.contradicting_evidence.append(f"Counterexample discovered: {ce}")
                                self.store.record_hypothesis(inv_id, h)
                    self.traces.log_event(inv_id, "counterexample_discovered", {"counterexample": ce})

                # Check for verification result
                if tool_name == "verify_solution":
                    v_res = VerificationResult(
                        level=VerificationLevel(tool_result_payload["verification_level"]),
                        status=VerificationStatus(tool_result_payload["status"]),
                        evidence=tool_result_payload.get("evidence", []),
                        unverified_steps=tool_result_payload.get("unverified_steps", []),
                        assumptions=tool_result_payload.get("assumptions", []),
                        counterexample=tool_result_payload.get("counterexample"),
                        confidence=tool_result_payload.get("confidence", 0.0),
                        summary=tool_result_payload.get("summary", "")
                    )
                    state.verification_result = v_res
                    self.store.record_verification(inv_id, v_res)
                    if v_res.status == VerificationStatus.VERIFIED:
                        state.status = InvestigationStatus.SOLVED

                # Append tool output to chat history
                messages.append(ChatMessage(
                    role="tool",
                    name=tool_name,
                    tool_call_id=call_id,
                    content=json.dumps(tool_result_payload, default=str)
                ))

            # Truncate old conversation messages if exceeding limits
            if len(messages) > self.config.agent.max_history_messages:
                # Keep system prompt and initial problem, keep last N messages
                preserved = [messages[0], messages[1]] + messages[-(self.config.agent.max_history_messages - 2):]
                messages = preserved

        # Final state update
        if state.candidate_solution:
            self._extract_report_sections(state, state.candidate_solution)

        if state.status in (InvestigationStatus.EXPLORING, InvestigationStatus.REVISING):
            if state.verification_result and state.verification_result.status in (VerificationStatus.VERIFIED, VerificationStatus.PARTIAL):
                state.status = InvestigationStatus.SOLVED
                state.termination_reason = "Verified mathematical solution produced"
            elif state.candidate_solution:
                state.status = InvestigationStatus.PARTIALLY_SOLVED
                state.termination_reason = "Model concluded investigation with candidate result"
            else:
                state.status = InvestigationStatus.EXHAUSTED
                state.termination_reason = state.termination_reason or "Finished available investigation steps"

        self.store.update_investigation(
            inv_id,
            status=state.status.value,
            active_hypothesis=state.active_hypothesis,
            candidate_solution=state.candidate_solution,
            termination_reason=state.termination_reason
        )

        final_report = state.to_report()
        self.traces.log_event(inv_id, "investigation_completed", final_report.model_dump())
        return final_report

    @staticmethod
    def _extract_report_sections(state: ActiveInvestigationState, text: str) -> None:
        """Extract structured report sections if present in text."""
        lines = text.split("\n")
        current_section = None
        for line in lines:
            trimmed = line.strip()
            lower = trimmed.lower()
            if "proven facts" in lower or "claimed result" in lower:
                current_section = "proven"
            elif "computational evidence" in lower:
                current_section = "evidence"
            elif "assumptions" in lower:
                current_section = "assumptions"
            elif "limitations" in lower or "unresolved" in lower:
                current_section = "unresolved"
            elif trimmed.startswith("- ") or trimmed.startswith("* "):
                bullet = trimmed[2:].strip()
                if current_section == "proven":
                    state.proven_results.append(bullet)
                elif current_section == "evidence":
                    state.computational_evidence.append(bullet)
                elif current_section == "assumptions":
                    state.assumptions_made.append(bullet)
                elif current_section == "unresolved":
                    state.unresolved_questions.append(bullet)

    def resume(self, investigation_id: str, additional_steps: int = 5) -> InvestigationReport:
        """Resume an interrupted investigation from persistent state."""
        inv_data = self.store.get_investigation(investigation_id)
        if not inv_data:
            raise ValueError(f"Investigation '{investigation_id}' not found in database.")

        problem: AgentProblemView = inv_data["problem"]
        return self.investigate(
            problem=problem,
            investigation_id=investigation_id,
            max_steps_override=additional_steps
        )

    def _extract_hypotheses_and_evidence(self, state: ActiveInvestigationState, text: str) -> None:
        """Scan assistant text for explicit hypotheses and update state."""
        lines = text.split("\n")
        for line in lines:
            line_str = line.strip()
            if any(prefix in line_str.lower() for prefix in ["hypothesis:", "conjecture:", "candidate hypothesis:"]):
                hyp_text = line_str.split(":", 1)[-1].strip()
                if hyp_text and hyp_text != state.active_hypothesis:
                    rec = state.record_hypothesis(hyp_text)
                    self.store.record_hypothesis(state.investigation_id, rec)
                    self.traces.log_event(state.investigation_id, "hypothesis_formed", {"hypothesis": hyp_text})
            elif "revised hypothesis:" in line_str.lower():
                rev_text = line_str.split(":", 1)[-1].strip()
                rec = state.record_hypothesis(rev_text, status=HypothesisStatus.ACTIVE)
                self.store.record_hypothesis(state.investigation_id, rec)
                self.traces.log_event(state.investigation_id, "hypothesis_revised", {"revised": rev_text})

    @staticmethod
    def _action_signature(tool_name: str, args: Dict[str, Any]) -> str:
        serialized = json.dumps({"tool": tool_name, "args": args}, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _convert_tool_result_to_experiment(
        self,
        tool_name: str,
        args: Dict[str, Any],
        result_payload: Dict[str, Any]
    ) -> ExperimentExecutionResult:
        exp_id = result_payload.get("experiment_id") or f"exp-{uuid.uuid4().hex[:8]}"
        action_id = f"act-{uuid.uuid4().hex[:8]}"
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return ExperimentExecutionResult(
            action_id=action_id,
            experiment_id=exp_id,
            input_parameters=args,
            execution_backend=result_payload.get("backend", self.backend.backend_name),
            start_time=now,
            finish_time=now,
            exit_status=result_payload.get("exit_status", 0),
            stdout=str(result_payload.get("stdout") or result_payload.get("result") or ""),
            stderr=str(result_payload.get("stderr") or result_payload.get("error") or ""),
            numerical_results=result_payload.get("numerical_results"),
            symbolic_results=result_payload.get("symbolic_results"),
            runtime_seconds=result_payload.get("runtime_seconds", 0.05),
            resource_usage={"backend": self.backend.backend_name},
            error_classification=result_payload.get("error_classification"),
            artifact_references=[],
            is_interrupted=result_payload.get("is_interrupted", False)
        )
