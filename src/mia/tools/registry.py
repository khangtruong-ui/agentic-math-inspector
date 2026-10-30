"""Tool Registry for Mathematical Investigation Agent (MIA).

Provides strongly-typed tool definitions, argument validation via Pydantic,
and standardized JSON schemas for LLM tool calling.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Type
from pydantic import BaseModel, ValidationError


class ToolDefinition(BaseModel):
    """Metadata and schema for an agent tool."""
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    arg_model: Type[BaseModel]
    handler: Callable[..., Any]

    def to_openai_schema(self) -> Dict[str, Any]:
        """Format as an OpenAI-compatible tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema
            }
        }

    def to_anthropic_schema(self) -> Dict[str, Any]:
        """Format as an Anthropic-compatible tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters_schema
        }


class ToolRegistry:
    """Manages registration, validation, and execution of mathematical tools."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        arg_model: Type[BaseModel]
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a typed tool handler."""
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            schema = arg_model.model_json_schema()
            # Clean up schema for LLM providers
            schema.pop("title", None)
            tool_def = ToolDefinition(
                name=name,
                description=description.strip(),
                parameters_schema=schema,
                arg_model=arg_model,
                handler=fn
            )
            self._tools[name] = tool_def
            return fn
        return decorator

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    def execute(self, name: str, arguments: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        """Validate arguments and execute the requested tool."""
        if name not in self._tools:
            return {
                "success": False,
                "tool_name": name,
                "error": f"Tool '{name}' is not registered. Available tools: {list(self._tools.keys())}",
                "result": None
            }

        tool_def = self._tools[name]
        try:
            validated_args = tool_def.arg_model.model_validate(arguments)
        except ValidationError as e:
            return {
                "success": False,
                "tool_name": name,
                "error": f"Invalid arguments for tool '{name}': {e.errors()}",
                "result": None
            }

        try:
            # Check if handler accepts context
            sig = inspect.signature(tool_def.handler)
            kwargs = validated_args.model_dump()
            if "context" in sig.parameters:
                kwargs["context"] = context

            result = tool_def.handler(**kwargs)
            return {
                "success": True,
                "tool_name": name,
                "result": result,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "tool_name": name,
                "error": f"Execution error in tool '{name}': {type(e).__name__}: {e}",
                "result": None
            }
