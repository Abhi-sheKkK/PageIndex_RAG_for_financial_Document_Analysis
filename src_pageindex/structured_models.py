"""
Pydantic models for structured LLM outputs, isolated for PageIndex pipeline.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class ExplainOutputs(BaseModel):
    """Structured output with explanation and answer."""
    explanation: str
    answer: str


class ExecutiveCompensation(BaseModel):
    """16-field executive compensation model per company."""
    company_name: str
    company_ceo: str
    coverage_period: str
    total_target_lti: Optional[float] = Field(None, description="Target LTI in USD")
    lti_grant_date: Optional[str] = None
    annual_lti_grant: Optional[bool] = None
    time_based_rsu_vesting_schedule: Optional[str] = None
    performance_based_rsu_vesting_schedule: Optional[str] = None
    compensation_governance_arrangements: Optional[str] = None
    ceo_pay_alignment_mechanisms: Optional[str] = None
    performance_metrics_used: Optional[str] = None
    realized_base_salary: Optional[float] = None
    realized_stis: Optional[float] = None
    realized_long_term_awards: Optional[float] = None
    realized_other_compensation: Optional[float] = None
    realized_total_compensation: Optional[float] = None


class ExecutiveCompensationReport(BaseModel):
    """Report containing compensation data for multiple companies."""
    companies: List[ExecutiveCompensation]


class EvaluationOutput(BaseModel):
    """LLM Judge evaluation result for a single field."""
    field: str
    model_answer: str
    ground_truth: str
    score: float
    explanation: str
