"""LangGraph agent loop: model -> tools -> model ... until the model stops calling tools.

The graph is a classic ReAct loop built explicitly with ``StateGraph`` so that the model can be
swapped at runtime (``/model``) while the conversation state lives in the checkpointer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

NODE_AGENT = "agent"
NODE_TOOLS = "tools"


def trim_history(messages: Sequence[AnyMessage], max_tokens: int) -> list[AnyMessage]:
    """Keep the most recent whole turns that fit into ``max_tokens`` (approximate count)."""
    if not messages:
        return []
    trimmed = trim_messages(
        list(messages),
        max_tokens=max_tokens,
        token_counter=count_tokens_approximately,
        strategy="last",
        start_on="human",
        include_system=False,
        allow_partial=False,
    )
    if trimmed:
        return list(trimmed)
    # A single oversized turn: keep it anyway rather than sending an empty conversation.
    return list(messages[-1:])


def build_graph(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt: str | Callable[[], str],
    checkpointer: BaseCheckpointSaver | None = None,
    *,
    context_budget_tokens: int = 120_000,
    cache_system_prompt: bool = False,
) -> CompiledStateGraph:
    model = llm.bind_tools(list(tools)) if tools else llm

    def system_message() -> SystemMessage:
        text = system_prompt() if callable(system_prompt) else system_prompt
        if cache_system_prompt:  # Anthropic prompt caching on the stable prefix
            return SystemMessage(
                content=[{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
            )
        return SystemMessage(content=text)

    def call_model(state: MessagesState) -> dict:
        history = trim_history(state["messages"], context_budget_tokens)
        response = model.invoke([system_message(), *history])
        return {"messages": [response]}

    def route(state: MessagesState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return NODE_TOOLS
        return END

    graph = StateGraph(MessagesState)
    graph.add_node(NODE_AGENT, call_model)
    graph.add_node(NODE_TOOLS, ToolNode(list(tools)))
    graph.add_edge(START, NODE_AGENT)
    graph.add_conditional_edges(NODE_AGENT, route, {NODE_TOOLS: NODE_TOOLS, END: END})
    graph.add_edge(NODE_TOOLS, NODE_AGENT)
    return graph.compile(checkpointer=checkpointer)
