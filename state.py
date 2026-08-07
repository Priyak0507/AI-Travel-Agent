from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from pydantic import BaseModel, Field

class AgentState(TypedDict):
    destination: str
    duration_days: int
    budget: float
    interests: List[str]
    constraints: List[str]
    model_name: str
    itinerary: Dict[str, Any]
    budget_breakdown: Dict[str, Any]
    validation: Dict[str, Any]
    correction_history: List[str]
    iteration: int
    max_iterations: int
    final_plan: str
    execution_trace: List[str]
    destination_image_url: str  # <--- ADD THIS LINE


class DayPlan(BaseModel):
    day: int = Field(..., ge=1)
    title: str
    activities: List[str] = Field(default_factory=list)
    estimated_day_cost: float = Field(..., ge=0)


class ItinerarySchema(BaseModel):
    city: str
    days: List[DayPlan] = Field(default_factory=list)


class BudgetSchema(BaseModel):
    accommodation: float = Field(..., ge=0)
    food: float = Field(..., ge=0)
    local_transport: float = Field(..., ge=0)
    activities: float = Field(..., ge=0)
    misc: float = Field(..., ge=0)
    total: float = Field(..., ge=0)
    within_budget: bool


class ValidationSchema(BaseModel):
    is_valid: bool
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
