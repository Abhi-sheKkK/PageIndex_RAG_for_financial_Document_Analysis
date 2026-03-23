"""
LLM-as-Judge evaluator for the PageIndex pipeline.
"""

import pandas as pd

from src_pageindex.config import OPENAI_MODEL_NAME
from src_pageindex.rag_pipeline import _call_openai_with_retry
from src_pageindex.structured_models import EvaluationOutput


def run_llm_judge(
    field: str,
    model_answer: str,
    ground_truth: str,
    model_name: str = OPENAI_MODEL_NAME,
) -> EvaluationOutput:
    """Evaluate accuracy of the model's response compared to ground truth."""
    prompt = f"""
    Evaluate the accuracy of the model's response by comparing it to the ground truth.
    Provide a score between 0 and 1:
    0 means completely incorrect or missing.
    1 means fully correct and accurate.

    Additionally, provide an explanation of why the score was given.

    Field: {field}

    Model Answer: {model_answer}
    Ground Truth: {ground_truth}

    Compare the two and assign a score, along with an explanation for the score.
    """

    messages = [
        {"role": "system", "content": "You are an expert evaluator comparing model outputs to ground truth."},
        {"role": "user", "content": prompt},
    ]

    parsed = _call_openai_with_retry(
        messages=messages,
        response_model=EvaluationOutput,
        model_name=model_name
    )

    return parsed


def evaluate_batch(combined_df: pd.DataFrame, model_name: str = OPENAI_MODEL_NAME) -> pd.DataFrame:
    """Score all rows in combined_df using the LLM judge."""
    scores = []
    for i, (_, row) in enumerate(combined_df.iterrows()):
        print(f"  [Evaluation] Scoring {i + 1}/{len(combined_df)}: {row['field']} for {row['company_name']}")
        result = run_llm_judge(
            field=row["field"],
            model_answer=str(row["executive_compensation_report_value"]),
            ground_truth=str(row["ground_truth_db_value"]),
            model_name=model_name,
        )
        scores.append(result.score)

    combined_df = combined_df.copy()
    combined_df["executive_compensation_score"] = scores
    return combined_df
