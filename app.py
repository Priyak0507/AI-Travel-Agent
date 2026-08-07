from __future__ import annotations

import os
from typing import Any, Dict, List

import streamlit as st

from dotenv import load_dotenv

from workflow import build_graph
# In app.py inside: with st.sidebar:

load_dotenv()

default_api_key = os.getenv("GROQ_API_KEY", "")
if "groq_api_key_input" not in st.session_state:
    st.session_state["groq_api_key_input"] = default_api_key

st.set_page_config(page_title="AI Travel Planning Agent", page_icon="🧭", layout="wide")

# Inject custom CSS
def load_css(file_name: str):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# Custom Animated Hero Banner
st.markdown("""
    <div class="hero-banner">
        <h1>🧭 AI Travel Planning Agent</h1>
        <p style="color: #cbd5e1; font-size: 1.1rem; margin: 0;">
            Multi-agent itinerary planning powered by LangGraph & Groq
        </p>
    </div>
""", unsafe_allow_html=True)
# Inject the styles
load_css("style.css")

st.title("AI Travel Planning Agent")
st.caption("Built with LangGraph + Groq (free API key friendly)")

with st.sidebar:
    st.header("Trip Inputs")
    destination = st.text_input("Destination", value="Bali")
    duration_days = st.number_input("Duration (days)", min_value=1, max_value=30, value=5, step=1)
    budget = st.number_input("Budget (USD)", min_value=50.0, value=800.0, step=50.0)
    interests_raw = st.text_input("Interests (comma-separated)", value="beaches, food, culture")
    constraints_raw = st.text_area(
        "Constraints (one per line)",
        value="No night travel\nVegetarian food preferred",
        height=120,
    )
    model_name = st.selectbox(
        "Groq Model",
        options=[
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
        ],
        index=0,
    )
    api_key = st.text_input(
        "Groq API Key",
        value=st.session_state.get("groq_api_key_input", ""),
        type="password",
        help="Get a free key from console.groq.com",
        key="groq_api_key_input",
    )
    max_iterations = st.slider("Max correction loops", min_value=1, max_value=4, value=2, step=1)
    run_clicked = st.button("Generate Travel Plan", use_container_width=True, type="primary")

final_state: Dict[str, Any] = {}


def _normalize_list_from_csv(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _normalize_list_from_lines(text: str) -> List[str]:
    return [item.strip() for item in text.splitlines() if item.strip()]


if run_clicked:
    if not api_key.strip():
        st.error("Please enter your Groq API key.")
        st.stop()

    os.environ["GROQ_API_KEY"] = api_key.strip()

    graph = build_graph()
    interests = _normalize_list_from_csv(interests_raw)
    constraints = _normalize_list_from_lines(constraints_raw)

    initial_state: Dict[str, Any] = {
        "destination": destination.strip(),
        "duration_days": int(duration_days),
        "budget": float(budget),
        "interests": interests,
        "constraints": constraints,
        "model_name": model_name,
        "itinerary": {},
        "budget_breakdown": {},
        "validation": {},
        "correction_history": [],
        "iteration": 0,
        "max_iterations": int(max_iterations),
        "final_plan": "",
        "execution_trace": [],
    }

    st.subheader("Live Agent Trace")
    trace_box = st.empty()
    progress = st.progress(0)
    trace_so_far: List[str] = []
    final_state: Dict[str, Any] = {}

    with st.spinner("Running multi-agent workflow..."):
        step_count = 0
        for step_count, state_snapshot in enumerate(graph.stream(initial_state, stream_mode="values"), start=1):
            final_state = state_snapshot
            trace_so_far = state_snapshot.get("execution_trace", [])
            rendered_trace = "\n".join([f"{idx + 1}. {line}" for idx, line in enumerate(trace_so_far)])
            trace_box.code(rendered_trace if rendered_trace else "Starting...", language="text")
            progress.progress(min(95, 15 + step_count * 15))
        progress.progress(100)

    if not final_state:
        st.error("Workflow did not return a final state. Please try again.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Final Plan", "Budget", "Itinerary JSON"])

    with tab1:
        st.markdown(final_state.get("final_plan", "No final plan generated."))

        validation = final_state.get("validation", {})
        if validation.get("is_valid", False):
            st.success("Validation passed.")
        else:
            st.warning("Validation found issues; final output may include unresolved constraints.")
            issues = validation.get("issues", [])
            if issues:
                st.write("**Issues found:**")
                for issue in issues:
                    st.write(f"- {issue}")

    with tab2:
        budget_data = final_state.get("budget_breakdown", {})
        total = float(budget_data.get("total", 0.0))
        st.metric("Estimated Total", f"${total:,.2f}")
        st.metric("Budget Limit", f"${float(budget):,.2f}")

        if budget > 0:
            ratio = min(total / float(budget), 1.0)
            st.progress(ratio, text=f"Budget utilization: {ratio * 100:.1f}%")

        for key in ["accommodation", "food", "local_transport", "activities", "misc"]:
            value = float(budget_data.get(key, 0.0))
            st.write(f"- **{key.replace('_', ' ').title()}**: ${value:,.2f}")

    with tab3:
        st.json(final_state.get("itinerary", {}))

st.divider()
st.markdown(
    "Tip: If model calls fail due to quota or model availability, switch Groq model in the sidebar and retry."
)

if run_clicked and final_state:
    img_url = final_state.get(
        "destination_image_url",
        "https://images.unsplash.com/photo-1488646953014-85cb44e25828?auto=format&fit=crop&w=1200&q=80",
    )

    st.markdown(
        f"""
        <style>
            .dynamic-hero-card {{
                background: linear-gradient(180deg, rgba(15, 23, 42, 0.2) 0%, rgba(15, 23, 42, 0.9) 100%),
                            url('{img_url}');
                background-size: cover;
                background-position: center;
                padding: 3rem 2rem;
                border-radius: 16px;
                border: 1px solid rgba(255, 255, 255, 0.15);
                margin-bottom: 1.5rem;
                box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            }}
        </style>
        <div class="dynamic-hero-card">
            <h1 style="color: white; margin: 0;">📍 {final_state.get('destination', 'Destination')} Plan</h1>
            <p style="color: #e2e8f0; font-size: 1.1rem;">Custom generated {final_state.get('duration_days')} Day Itinerary</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
