from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from agents import budget_agent, finalizer_agent, planner_agent, validator_agent
from state import AgentState


def correction_node(state: Dict[str, Any]) -> Dict[str, Any]:
    iteration = int(state.get("iteration", 0)) + 1
    trace = state.get("execution_trace", [])
    trace.append(f"Correction Node: Replanning with feedback (iteration {iteration}).")
    return {"iteration": iteration, "execution_trace": trace}


def route_after_validation(state: Dict[str, Any]) -> str:
    validation = state.get("validation", {})
    is_valid = validation.get("is_valid", False)
    iteration = int(state.get("iteration", 0))
    max_iterations = int(state.get("max_iterations", 2))

    if is_valid or iteration >= max_iterations:
        return "finalize"
    return "retry"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_agent)
    graph.add_node("budgeter", budget_agent)
    graph.add_node("validator", validator_agent)
    graph.add_node("correction", correction_node)
    graph.add_node("finalizer", finalizer_agent)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "budgeter")
    graph.add_edge("budgeter", "validator")
    graph.add_conditional_edges(
        "validator",
        route_after_validation,
        {"retry": "correction", "finalize": "finalizer"},
    )
    graph.add_edge("correction", "planner")
    graph.add_edge("finalizer", END)

    return graph.compile()

   
from agents import budget_agent, finalizer_agent, planner_agent, validator_agent, image_agent

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("image_fetcher", image_agent)  # <--- Added image agent
    graph.add_node("planner", planner_agent)
    graph.add_node("budgeter", budget_agent)
    graph.add_node("validator", validator_agent)
    graph.add_node("correction", correction_node)
    graph.add_node("finalizer", finalizer_agent)

    # Route START to image_fetcher, then proceed to planner
    graph.add_edge(START, "image_fetcher")
    graph.add_edge("image_fetcher", "planner")
    graph.add_edge("planner", "budgeter")
    graph.add_edge("budgeter", "validator")
    graph.add_conditional_edges(
        "validator",
        route_after_validation,
        {"retry": "correction", "finalize": "finalizer"},
    )
    graph.add_edge("correction", "planner")
    graph.add_edge("finalizer", END)

    return graph.compile()
