from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

from iu_agent.agent.graph import build_graph, trim_history
from iu_agent.agent.tools import AgentContext, build_tools
from iu_agent.rag.ingest import Manifest


class ScriptedChatModel(BaseChatModel):
    responses: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        message = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs):
        return self


def _tool_call(name: str, args: dict, call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


@tool
def echo(text: str) -> str:
    """Echo the text."""
    return f"echo:{text}"


@tool
def guarded(action: str) -> str:
    """Needs approval."""
    decision = interrupt({"kind": "guarded", "action": action})
    return "ran" if decision.get("approved") else "denied"


CONFIG = {"configurable": {"thread_id": "t1"}}


def test_tool_loop_runs_tools_then_answers():
    model = ScriptedChatModel(responses=[_tool_call("echo", {"text": "hi"}), AIMessage(content="done")])
    graph = build_graph(model, [echo], "system", InMemorySaver())
    result = graph.invoke({"messages": [HumanMessage(content="x")]}, CONFIG)
    messages = result["messages"]
    assert isinstance(messages[2], ToolMessage) and messages[2].content == "echo:hi"
    assert messages[-1].content == "done"
    assert model.calls == 2


def test_interrupt_pauses_and_resumes_with_decision():
    model = ScriptedChatModel(
        responses=[_tool_call("guarded", {"action": "rm"}), AIMessage(content="finished")]
    )
    graph = build_graph(model, [guarded], "system", InMemorySaver())
    interrupts = []
    for _mode, data in graph.stream(
        {"messages": [HumanMessage(content="go")]}, CONFIG, stream_mode=["updates"]
    ):
        if "__interrupt__" in data:
            interrupts.extend(data["__interrupt__"])
    assert len(interrupts) == 1
    assert interrupts[0].value == {"kind": "guarded", "action": "rm"}

    result = graph.invoke(Command(resume={interrupts[0].id: {"approved": True}}), CONFIG)
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages[-1].content == "ran"
    assert result["messages"][-1].content == "finished"


def test_trim_history_keeps_recent_turns():
    messages = [
        HumanMessage(content="a" * 4000),
        AIMessage(content="b"),
        HumanMessage(content="c"),
        AIMessage(content="d"),
    ]
    trimmed = trim_history(messages, max_tokens=50)
    assert trimmed[0].content == "c"
    assert trim_history([HumanMessage(content="x" * 10000)], max_tokens=10)[0].content.startswith("x")


def _context(settings, approval_mode: str) -> AgentContext:
    return AgentContext(
        settings=settings,
        manifest=Manifest(settings.manifest_path),
        store_factory=lambda: None,
        workspace=Path(settings.workspace_dir).resolve(),
        approval_mode=approval_mode,
    )


def test_tools_without_store_and_with_denied_approval(settings):
    tools = {t.name: t for t in build_tools(_context(settings, "deny"))}
    assert "not available" in tools["search_course_material"].invoke({"query": "x"}).lower()
    assert "nothing is indexed" in tools["list_courses"].invoke({}).lower()
    assert "declined" in tools["run_shell"].invoke({"command": "echo hi"})
    assert "declined" in tools["write_file"].invoke({"path": "a.txt", "content": "x"})
    assert "outside the workspace" in tools["read_file"].invoke(
        {"path": str(Path(settings.data_dir).resolve() / "manifest.json")}
    )


def test_tools_with_auto_approval(settings):
    tools = {t.name: t for t in build_tools(_context(settings, "allow"))}
    assert "Wrote" in tools["write_file"].invoke({"path": "notes/a.txt", "content": "hello\nworld"})
    assert (Path(settings.workspace_dir) / "notes" / "a.txt").read_text(encoding="utf-8") == "hello\nworld"
    listing = tools["list_directory"].invoke({"path": "notes"})
    assert "a.txt" in listing
    assert "2  world" in tools["read_file"].invoke({"path": "notes/a.txt"})
    output = tools["run_shell"].invoke({"command": "echo agent-ok"})
    assert "exit code: 0" in output and "agent-ok" in output


def test_graph_without_tools_injects_retrieved_context():
    seen: list = []

    class RecordingModel(ScriptedChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            seen.append(list(messages))
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    model = RecordingModel(responses=[AIMessage(content="answer")])
    graph = build_graph(model, [], "system", InMemorySaver(), retriever=lambda q: f"CONTEXT for {q}")
    config = {"configurable": {"thread_id": "t9"}}
    result = graph.invoke({"messages": [HumanMessage(content="eigenvalues?")]}, config)
    assert result["messages"][-1].content == "answer"
    system = seen[0][0]
    assert isinstance(system, SystemMessage)
    assert system.content.startswith("system")
    assert "CONTEXT for eigenvalues?" in system.content
