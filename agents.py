from __future__ import annotations

import json
import os
import urllib.parse
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from state import BudgetSchema, ItinerarySchema, ValidationSchema


def _get_llm(state: Dict[str, Any]) -> ChatGroq:
    model_name = state.get("model_name", "llama-3.1-8b-instant")
    temperature = 0.2
    return ChatGroq(model=model_name, temperature=temperature)


def _fallback_itinerary(state: Dict[str, Any]) -> Dict[str, Any]:
    destination = state["destination"]
    duration_days = max(1, int(state["duration_days"]))
    interests = ", ".join(state.get("interests", [])) or "local highlights"

    days = []
    for day in range(1, duration_days + 1):
        days.append(
            {
                "day": day,
                "title": f"Day {day} in {destination}",
                "activities": [
                    f"Morning: Explore a popular {interests} spot",
                    "Afternoon: Visit a local market or museum",
                    "Evening: Food walk and relaxed city stroll",
                ],
                "estimated_day_cost": 60.0,
            }
        )
    return {"city": destination, "days": days}


def planner_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    previous_issues = state.get("correction_history", [])

    prompt = f"""
Create a practical day-by-day travel itinerary in strict JSON style.

Destination: {state["destination"]}
Trip length (days): {state["duration_days"]}
Budget (USD): {state["budget"]}
Interests: {", ".join(state.get("interests", []))}
Constraints: {", ".join(state.get("constraints", []))}
Previous validation issues to fix: {previous_issues}

Return a realistic plan with one entry per day.
Keep activities concrete and feasible.
"""
    trace = state.get("execution_trace", [])
    try:
        llm = _get_llm(state)
        itinerary_llm = llm.with_structured_output(ItinerarySchema)
        itinerary = itinerary_llm.invoke(
            [SystemMessage(content="You are a precise travel planner."), HumanMessage(content=prompt)]
        )
        itinerary_data = itinerary.model_dump()
        trace.append(f"Planner Agent: Generated itinerary (iteration {state.get('iteration', 0)}).")
    except Exception as e:
        trace.append(f"Planner Agent Error: {str(e)}. Using fallback itinerary.")
        itinerary_data = _fallback_itinerary(state)

    return {"itinerary": itinerary_data, "execution_trace": trace}


def budget_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    itinerary = state.get("itinerary", {})

    prompt = f"""
Estimate a realistic budget breakdown in USD for this itinerary.
Total budget limit: {state["budget"]}
Destination: {state["destination"]}
Trip length (days): {state["duration_days"]}
Itinerary JSON:
{json.dumps(itinerary, indent=2)}

Return accommodation, food, local_transport, activities, misc, total, and within_budget.
"""
    trace = state.get("execution_trace", [])
    try:
        llm = _get_llm(state)
        budget_llm = llm.with_structured_output(BudgetSchema)
        budget = budget_llm.invoke(
            [SystemMessage(content="You estimate realistic travel costs."), HumanMessage(content=prompt)]
        )
        budget_data = budget.model_dump()
        trace.append("Budget Agent: Calculated estimated trip cost.")
    except Exception as e:
        trace.append(f"Budget Agent Error: {str(e)}. Using fallback budget calculation.")
        days = max(1, int(state["duration_days"]))
        total = float(days * 90)
        budget_data = {
            "accommodation": days * 35.0,
            "food": days * 20.0,
            "local_transport": days * 10.0,
            "activities": days * 20.0,
            "misc": days * 5.0,
            "total": total,
            "within_budget": total <= float(state["budget"]),
        }

    return {"budget_breakdown": budget_data, "execution_trace": trace}


def validator_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []
    suggestions: List[str] = []

    budget_limit = float(state["budget"])
    budget_data = state.get("budget_breakdown", {})
    itinerary = state.get("itinerary", {})
    days = itinerary.get("days", [])

    total_estimated = float(budget_data.get("total", 0))
    if total_estimated > budget_limit:
        issues.append(f"Estimated total ${total_estimated:.2f} exceeds budget ${budget_limit:.2f}.")
        suggestions.append("Reduce paid activities and choose lower-cost accommodation.")

    expected_days = int(state["duration_days"])
    if len(days) != expected_days:
        issues.append(f"Itinerary has {len(days)} days but trip requires {expected_days} days.")
        suggestions.append("Ensure itinerary includes exactly one plan per trip day.")

    constraints = [c.lower() for c in state.get("constraints", [])]
    if "no night travel" in constraints:
        night_hits = []
        for idx, day in enumerate(days, start=1):
            text = " ".join(day.get("activities", [])).lower()
            if "night bus" in text or "overnight train" in text or "red-eye" in text:
                night_hits.append(idx)
        if night_hits:
            issues.append(f"Night travel constraint violated on day(s): {night_hits}.")
            suggestions.append("Replace overnight transfers with daytime options.")

    if not issues and not budget_data.get("within_budget", True):
        issues.append("Budget flag indicates plan is not within budget.")
        suggestions.append("Recalculate with lower average daily spend.")

    validation_obj = ValidationSchema(is_valid=len(issues) == 0, issues=issues, suggestions=suggestions)
    validation_data = validation_obj.model_dump()

    trace = state.get("execution_trace", [])
    if validation_data["is_valid"]:
        trace.append("Validator Agent: Plan validated successfully.")
    else:
        trace.append("Validator Agent: Issues found, sending plan for correction.")

    new_history = state.get("correction_history", [])
    if issues:
        new_history.extend(issues)

    return {
        "validation": validation_data,
        "correction_history": new_history,
        "execution_trace": trace,
    }


def finalizer_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""
Create a concise, user-friendly final travel plan in markdown.
Include:
1) Trip snapshot
2) Day-by-day itinerary
3) Budget summary
4) Packing tips
5) Local etiquette tip

Trip input:
Destination: {state["destination"]}
Days: {state["duration_days"]}
Budget: {state["budget"]}
Interests: {", ".join(state.get("interests", []))}
Constraints: {", ".join(state.get("constraints", []))}

Itinerary JSON:
{json.dumps(state.get("itinerary", {}), indent=2)}

Budget JSON:
{json.dumps(state.get("budget_breakdown", {}), indent=2)}
"""
    trace = state.get("execution_trace", [])
    try:
        llm = _get_llm(state)
        output = llm.invoke(
            [SystemMessage(content="You create clear, practical travel plans."), HumanMessage(content=prompt)]
        )
        final_text = output.content if isinstance(output.content, str) else str(output.content)
        trace.append("Finalizer Agent: Final markdown plan created.")
    except Exception as e:
        trace.append(f"Finalizer Agent Error: {str(e)}. Using fallback markdown plan.")
        final_text = (
            f"# {state['destination']} Travel Plan\n\n"
            f"- Duration: {state['duration_days']} days\n"
            f"- Budget: ${state['budget']}\n\n"
            "A complete itinerary and budget have been generated."
        )

    return {"final_plan": final_text, "execution_trace": trace}


import requests

UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "")

def fetch_destination_images(destination: str, per_page: int = 3) -> List[str]:
    """Fetches one or more destination image URLs from Unsplash."""
    destination_query = urllib.parse.quote(destination or "travel")
    fallback_url = f"https://source.unsplash.com/featured/1200x800/?{destination_query}"
    if not UNSPLASH_ACCESS_KEY or UNSPLASH_ACCESS_KEY == "YOUR_UNSPLASH_ACCESS_KEY":
        return [fallback_url] * per_page

    query = urllib.parse.quote(f"{destination} travel landmark")
    url = f"https://api.unsplash.com/search/photos?query={query}&orientation=landscape&per_page={per_page}"
    headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}

    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            if results:
                return [item["urls"]["regular"] for item in results]
    except Exception:
        pass
    return [fallback_url] * per_page


def fetch_destination_image(destination: str) -> str:
    """Fetches a high-quality photo URL for the destination via Unsplash API."""
    return fetch_destination_images(destination, per_page=1)[0]


def image_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Agent node that resolves dynamic header photography for the state."""
    destination = state.get("destination", "travel")
    trace = state.get("execution_trace", [])
    
    image_url = fetch_destination_image(destination)
    trace.append(f"Image Agent: Fetched photography banner for {destination}.")
    
    return {
        "destination_image_url": image_url,
        "execution_trace": trace
    }