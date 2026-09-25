"""LangGraph agent loop: model -> tools -> model ... until the model stops calling tools.

The graph is a classic ReAct loop built explicitly with ``StateGraph`` so that the model can be
swapped at runtime (``/model``) while the conversation state lives in the checkpointer.

For endpoints without function calling the graph runs without tools: a ``retriever`` callable
is invoked with the latest user message and its result is appended to the system prompt
(classic retrieval-augmented generation).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

NODE_AGENT = "agent"
NODE_TOOLS = "tools"

Retriever = Callable[[str], str]


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


def message_text(message: AnyMessage) -> str:
    """Plain text of a message whose content may be a string or a list of content blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def build_graph(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt: str | Callable[[], str],
    checkpointer: BaseCheckpointSaver | None = None,
    *,
    context_budget_tokens: int = 120_000,
    cache_system_prompt: bool = False,
    retriever: Retriever | None = None,
) -> CompiledStateGraph:
    tools = list(tools)
    model = llm.bind_tools(tools) if tools else llm

    def system_message(history: Sequence[AnyMessage]) -> SystemMessage:
        text = system_prompt() if callable(system_prompt) else system_prompt
        if retriever is not None and not tools:
            question = next((message_text(m) for m in reversed(history) if isinstance(m, HumanMessage)), "")
            context = retriever(question) if question.strip() else ""
            if context:
                text = f"{text}\n\n# Retrieved course material for the latest user message\n{context}"
        if cache_system_prompt:  # Anthropic prompt caching on the stable prefix
            return SystemMessage(
                content=[{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
            )
        return SystemMessage(content=text)

    def call_model(state: MessagesState) -> dict:
        history = trim_history(state["messages"], context_budget_tokens)
        response = model.invoke([system_message(history), *history])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node(NODE_AGENT, call_model)
    graph.add_edge(START, NODE_AGENT)
    if tools:

        def route(state: MessagesState) -> str:
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return NODE_TOOLS
            return END

        graph.add_node(NODE_TOOLS, ToolNode(tools))
        graph.add_conditional_edges(NODE_AGENT, route, {NODE_TOOLS: NODE_TOOLS, END: END})
        graph.add_edge(NODE_TOOLS, NODE_AGENT)
    else:
        graph.add_edge(NODE_AGENT, END)
    return graph.compile(checkpointer=checkpointer)
