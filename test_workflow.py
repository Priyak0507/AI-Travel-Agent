from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from agents import budget_agent, finalizer_agent, planner_agent, validator_agent
from state import BudgetSchema, ItinerarySchema, ValidationSchema
from workflow import build_graph, route_after_validation


class TestTravelPlanningAgent(unittest.TestCase):
    def setUp(self):
        self.initial_state = {
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

    @patch("agents.ChatGroq")
    def test_planner_agent_success(self, mock_chat_groq):
        # Setup mock for ChatGroq and structured output
        mock_llm = MagicMock()
        mock_chat_groq.return_value = mock_llm

        mock_structured_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        # Mock the returned ItinerarySchema
        mock_itinerary = ItinerarySchema(
            city="Bali",
            days=[
                {
                    "day": 1,
                    "title": "Day 1 in Bali",
                    "activities": ["Visit temple", "Relax at beach"],
                    "estimated_day_cost": 50.0,
                },
                {
                    "day": 2,
                    "title": "Day 2 in Bali",
                    "activities": ["Scuba diving", "Seafood dinner"],
                    "estimated_day_cost": 100.0,
                },
                {
                    "day": 3,
                    "title": "Day 3 in Bali",
                    "activities": ["Spa day", "Souvenir shopping"],
                    "estimated_day_cost": 40.0,
                },
            ],
        )
        mock_structured_llm.invoke.return_value = mock_itinerary

        res = planner_agent(self.initial_state)

        self.assertIn("itinerary", res)
        self.assertEqual(res["itinerary"]["city"], "Bali")
        self.assertEqual(len(res["itinerary"]["days"]), 3)
        self.assertIn("Planner Agent: Generated itinerary", res["execution_trace"][0])

    @patch("agents.ChatGroq")
    def test_planner_agent_fallback(self, mock_chat_groq):
        # Force an exception to trigger the fallback itinerary
        mock_chat_groq.side_effect = Exception("API Error")

        res = planner_agent(self.initial_state)

        self.assertIn("itinerary", res)
        self.assertEqual(res["itinerary"]["city"], "Bali")
        self.assertEqual(len(res["itinerary"]["days"]), 3)
        self.assertIn("Planner Agent Error: API Error. Using fallback itinerary.", res["execution_trace"][0])

    @patch("agents.ChatGroq")
    def test_budget_agent_success(self, mock_chat_groq):
        mock_llm = MagicMock()
        mock_chat_groq.return_value = mock_llm

        mock_structured_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        mock_budget = BudgetSchema(
            accommodation=150.0,
            food=100.0,
            local_transport=50.0,
            activities=100.0,
            misc=30.0,
            total=430.0,
            within_budget=True,
        )
        mock_structured_llm.invoke.return_value = mock_budget

        res = budget_agent(self.initial_state)

        self.assertIn("budget_breakdown", res)
        self.assertEqual(res["budget_breakdown"]["total"], 430.0)
        self.assertTrue(res["budget_breakdown"]["within_budget"])

    def test_validator_agent_valid(self):
        state = self.initial_state.copy()
        state["itinerary"] = {
            "city": "Bali",
            "days": [
                {"day": 1, "activities": ["Visit temple"], "estimated_day_cost": 50.0},
                {"day": 2, "activities": ["Relax at beach"], "estimated_day_cost": 50.0},
                {"day": 3, "activities": ["Spa day"], "estimated_day_cost": 50.0},
            ],
        }
        state["budget_breakdown"] = {
            "accommodation": 150.0,
            "food": 100.0,
            "local_transport": 50.0,
            "activities": 100.0,
            "misc": 30.0,
            "total": 430.0,
            "within_budget": True,
        }

        res = validator_agent(state)

        self.assertTrue(res["validation"]["is_valid"])
        self.assertEqual(len(res["validation"]["issues"]), 0)

    def test_validator_agent_invalid_budget(self):
        state = self.initial_state.copy()
        state["itinerary"] = {
            "city": "Bali",
            "days": [
                {"day": 1, "activities": ["Visit temple"], "estimated_day_cost": 50.0},
                {"day": 2, "activities": ["Relax at beach"], "estimated_day_cost": 50.0},
                {"day": 3, "activities": ["Spa day"], "estimated_day_cost": 50.0},
            ],
        }
        state["budget_breakdown"] = {
            "accommodation": 300.0,
            "food": 200.0,
            "local_transport": 100.0,
            "activities": 200.0,
            "misc": 50.0,
            "total": 850.0,  # Exceeds budget limit of 500.0
            "within_budget": False,
        }

        res = validator_agent(state)

        self.assertFalse(res["validation"]["is_valid"])
        self.assertTrue(any("exceeds budget" in issue for issue in res["validation"]["issues"]))

    def test_validator_agent_invalid_constraint(self):
        state = self.initial_state.copy()
        state["constraints"] = ["No night travel"]
        state["itinerary"] = {
            "city": "Bali",
            "days": [
                {"day": 1, "activities": ["Visit temple", "Night bus to Ubud"], "estimated_day_cost": 50.0},
                {"day": 2, "activities": ["Relax at beach"], "estimated_day_cost": 50.0},
                {"day": 3, "activities": ["Spa day"], "estimated_day_cost": 50.0},
            ],
        }
        state["budget_breakdown"] = {
            "accommodation": 150.0,
            "food": 100.0,
            "local_transport": 50.0,
            "activities": 100.0,
            "misc": 30.0,
            "total": 430.0,
            "within_budget": True,
        }

        res = validator_agent(state)

        self.assertFalse(res["validation"]["is_valid"])
        self.assertTrue(any("Night travel constraint violated" in issue for issue in res["validation"]["issues"]))

    def test_route_after_validation(self):
        # Valid plan -> finalize
        state = {"validation": {"is_valid": True}, "iteration": 0, "max_iterations": 2}
        self.assertEqual(route_after_validation(state), "finalize")

        # Invalid plan, under max_iterations -> retry
        state = {"validation": {"is_valid": False}, "iteration": 0, "max_iterations": 2}
        self.assertEqual(route_after_validation(state), "retry")

        # Invalid plan, reached max_iterations -> finalize
        state = {"validation": {"is_valid": False}, "iteration": 2, "max_iterations": 2}
        self.assertEqual(route_after_validation(state), "finalize")

    @patch("agents.ChatGroq")
    def test_full_graph_execution(self, mock_chat_groq):
        # Mock LLM to avoid real API calls
        mock_llm = MagicMock()
        mock_chat_groq.return_value = mock_llm

        mock_structured_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured_llm

        # Mock responses
        mock_itinerary = ItinerarySchema(
            city="Bali",
            days=[
                {"day": 1, "title": "Day 1", "activities": ["Temple"], "estimated_day_cost": 50.0},
                {"day": 2, "title": "Day 2", "activities": ["Beach"], "estimated_day_cost": 50.0},
                {"day": 3, "title": "Day 3", "activities": ["Spa"], "estimated_day_cost": 50.0},
            ],
        )
        mock_budget = BudgetSchema(
            accommodation=150.0,
            food=100.0,
            local_transport=50.0,
            activities=100.0,
            misc=30.0,
            total=430.0,
            within_budget=True,
        )
        mock_structured_llm.invoke.side_effect = [mock_itinerary, mock_budget]

        mock_final_plan = MagicMock()
        mock_final_plan.content = "# Bali Travel Plan\n\nPerfect itinerary!"
        mock_llm.invoke.return_value = mock_final_plan

        # Build and compile graph
        compiled_graph = build_graph()

        # Run graph
        result = compiled_graph.invoke(self.initial_state)

        self.assertIn("final_plan", result)
        self.assertIn("Perfect itinerary!", result["final_plan"])
        self.assertTrue(result["validation"]["is_valid"])
        self.assertEqual(result["iteration"], 0)


if __name__ == "__main__":
    unittest.main()
