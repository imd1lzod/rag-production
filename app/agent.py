from typing import Optional
from typing_extensions import TypedDict, Annotated
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langsmith import traceable

from app.config import get_settings


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str


class ProductionAgent:

    def __init__(self):
        self.settings = get_settings()

        self.primary_model = ChatGoogleGenerativeAI(
            model=self.settings.primary_model,
            temperature=0,
            timeout=30,
            max_retries=0
        )

        self.fallback_model = ChatGoogleGenerativeAI(
            model=self.settings.fallback_model,
            temperature=0,
            timeout=30,
            max_retries=0
        )

        self.max_retries = self.settings.max_retries
        self.graph = self.build_graph()

    def build_graph(self) -> StateGraph[AgentState]:
        def process_messages(state: AgentState) -> AgentState:
            if state["retry_count"] > self.max_retries:
                state["error"] = "Max retries exceeded"
                return state

            try:
                model = self.primary_model
                response = model.invoke(state["messages"])
                state["model_used"] = self.settings.primary_model
                state["messages"].append(AIMessage(content=response.content))
            except Exception as e:
                state["error"] = str(e)
                state["retry_count"] += 1
                state["model_used"] = self.settings.fallback_model
                response = self.fallback_model.invoke(state["messages"])

                state["messages"].append(AIMessage(content=response.content))

            return state

        def try_fallback(state: AgentState) -> AgentState:
            try:
                response = self.fallback_model.invoke(state["messages"])

                state["model_used"] = self.settings.fallback_model
                state["messages"].append(AIMessage(content=response.content))
            except Exception as e:
                state["error"] = str(e)
                state["retry_count"] += 1

            return state

        def handle_error(state: AgentState) -> AgentState:
            # Handle error state, e.g., log it or perform some action
            return {
                "messages": [
                    AIMessage(content=(
                        "I'm sorry, but I encountered an error while processing your request. ",
                        "Please try again later or contact support if the issue persists."
                    )),
                ]
            }

        def route_after_process(state: AgentState) -> StateGraph[AgentState]:
            if state.get("error") is None:
                return "done"
            elif state["retry_count"] <= self.max_retries:
                return "fallback"
            else:
                return "error"

        def route_after_fallback(state: AgentState) -> StateGraph[AgentState]:
            if state.get("error") is None:
                return "done"
            else:
                return "error"

        graph = StateGraph(AgentState)
        graph.add_node("process", process_messages)
        graph.add_node("fallback", try_fallback)
        graph.add_node("error", handle_error)

        graph.add_edge(START, "process")
        graph.add_conditional_edges(
            "fallback",
            route_after_fallback,
            {"done": END, "error": "error"}
        )
        graph.add_edge("error", END)

        return graph.compile()

    @traceable
    def invoke(self, messages: list[BaseMessage]) -> list[BaseMessage]:

        result = self.graph.invoke({
            "messages": messages,
            "error": None,
            "retry_count": 0,
            "model_used": ""
        })

        return {
            "response": result["messages"][-1].content,
            "model_used": result["model_used"],
            "error": result.get("error")
        }