import os
import json
from workflow import build_graph

# Set API key and model
os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY", "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")

initial_state = {
    "destination": "Bali",
    "duration_days": 3,
    "budget": 500.0,
    "interests": ["beaches", "temples"],
    "constraints": ["No night travel"],
    "model_name": "llama-3.1-8b-instant",
    "itinerary": {},
    "budget_breakdown": {},
    "validation": {},
    "correction_history": [],
    "iteration": 0,
    "max_iterations": 2,
    "final_plan": "",
    "execution_trace": [],
}

print("Compiling LangGraph workflow...")
graph = build_graph()

print("Executing workflow...")
try:
    result = graph.invoke(initial_state)
    print("\n--- Execution Trace ---")
    for idx, trace in enumerate(result.get("execution_trace", []), 1):
        print(f"{idx}. {trace}")
        
    print("\n--- Final Itinerary (JSON) ---")
    print(json.dumps(result.get("itinerary", {}), indent=2))
    
    print("\n--- Budget Breakdown (JSON) ---")
    print(json.dumps(result.get("budget_breakdown", {}), indent=2))
    
    print("\n--- Final Plan (Markdown) ---")
    print(result.get("final_plan", ""))
except Exception as e:
    print(f"Error executing graph: {e}")
