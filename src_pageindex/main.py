"""
Main orchestrator for the standardized PageIndex pipeline.
"""

import argparse
import pandas as pd
from pathlib import Path

from src_pageindex.config import BASE_DIR, PDF_DIR, GROUND_TRUTH_PATH, PAGEINDEX_CACHE_DIR
from src_pageindex.pageindex_client import submit_and_get_tree
from src_pageindex.rag_pipeline import run_pageindex_rag_query
from src_pageindex.evaluator import evaluate_batch
from src_pageindex.visualize import create_accuracy_heatmap
from src_pageindex.structured_models import ExecutiveCompensationReport


EXTRACTION_CACHE_PATH = BASE_DIR / "pageindex_extraction_report.csv"
EVALUATION_CACHE_PATH = BASE_DIR / "pageindex_evaluation_results.csv"
HEATMAP_PATH = "heatmap_pageindex.png"
COMPANIES = ["Apple Inc.", "Amazon.com, Inc.", "Microsoft Corporation"]

EXTRACTION_QUERY_SINGLE_COMPANY = """You are an expert financial analysis assistant specializing in extracting detailed executive compensation data from proxy statements. Your task is to extract the following fields from the provided proxy statement of {company} and output the information in the given JSON structure. For any missing or not applicable information, use "N/A".

Extract these fields:
1. Company Name: Full legal name of the company.
2. Company CEO: Name of the Chief Executive Officer.
3. Coverage Period: The fiscal period covered by the proxy (e.g., "Fiscal Year 2024").
4. Total Target LTI (Full Grant Amount): The total target long-term incentive amount, including all equity components.
5. LTI Package Grant Date: The date when the LTI grant was awarded.
6. Annual LTI Grant?: Indicate "Yes" if equity awards are granted annually, or "No" if not.
7. Time-Based RSU Vesting Schedule: Details on vesting for time-based equity. If not applicable, indicate "N/A".
8. Performance-Based RSU Vesting Schedule: Details on vesting for performance-based equity. If not applicable, indicate "N/A".
9. Compensation Governance Arrangements: Information on the oversight mechanisms.
10. CEO Pay Alignment Mechanisms: How the compensation is structured to align CEO pay with long-term shareholder value.
11. Performance Metrics Used (Detailed): Provide specifics of the performance metrics applied.
12. Realized Base Salary: Actual base salary paid in the period.
13. Realized STIs: Actual short-term incentives (bonuses) paid.
14. Realized Long-Term Awards: Value of equity awards that have vested in the period.
15. Realized Other Compensation: Additional benefits.
16. Realized Total Compensation: The sum of all compensation elements actually received in the period.
"""


def get_pdf_path_for_company(company: str) -> Path:
    """Helper to find the PDF file for a company from the known filenames."""
    mapping = {
        "Apple Inc.": "Apple Inc.- DEF14A.pdf",
        "Amazon.com, Inc.": "Amazon Inc.- DEF14A.pdf",
        "Microsoft Corporation": "Microsoft Inc.- DEF14A.pdf"
    }
    fname = mapping.get(company)
    if not fname:
        raise ValueError(f"No PDF mapping found for {company}")
    return PDF_DIR / fname


def step_extract():
    """Extract data across all company proxy PDFs using PageIndex reasoning RAG."""
    print("\\n" + "=" * 60)
    print("STEP 1: PageIndex Multi-Company Extraction")
    print("=" * 60)

    all_company_reports = []
    
    for company in COMPANIES:
        print(f"\\nProcessing {company}...")
        pdf_path = get_pdf_path_for_company(company)
        
        # 1. Ensure document is submitted and we have the tree
        tree = submit_and_get_tree(pdf_path)
        
        # 2. Extract 
        query = EXTRA_QUERY_SINGLE_COMPANY.format(company=company) if 'EXTRA_QUERY_SINGLE_COMPANY' in locals() else EXTRACTION_QUERY_SINGLE_COMPANY.format(company=company)
        result = run_pageindex_rag_query(query, tree, response_model=ExecutiveCompensationReport)
        
        if result and result.companies:
            all_company_reports.extend(result.companies)
            
    # Combine results
    df = pd.DataFrame([c.model_dump() for c in all_company_reports])
    print("\\nExtracted Data:")
    print(df.to_string(index=False))

    df.to_csv(EXTRACTION_CACHE_PATH, index=False)
    print(f"\\n💾 Extraction saved to {EXTRACTION_CACHE_PATH}")
    return df


def step_evaluate(report_df):
    """Evaluate against ground truth using LLM judge."""
    print("\\n" + "=" * 60)
    print("STEP 2: LLM-as-Judge Evaluation (PageIndex)")
    print("=" * 60)

    if not GROUND_TRUTH_PATH.exists():
        print(f"⚠ Ground truth file not found: {GROUND_TRUTH_PATH}")
        return None

    ground_truth_df = pd.read_excel(GROUND_TRUTH_PATH)

    gt_melted = ground_truth_df.melt(
        id_vars=["company_name"], var_name="field", value_name="ground_truth_db_value"
    )
    report_melted = report_df.melt(
        id_vars=["company_name"], var_name="field", value_name="executive_compensation_report_value"
    )

    # Normalize company names for robust merging
    gt_melted["company_name"] = gt_melted["company_name"].str.upper().str.strip()
    report_melted["company_name"] = report_melted["company_name"].str.upper().str.strip()

    combined_df = gt_melted.merge(report_melted, on=["company_name", "field"])
    
    # ── Evaluation Caching Logic ───────────────────────────────────────────
    if EVALUATION_CACHE_PATH.exists():
        existing_eval = pd.read_csv(EVALUATION_CACHE_PATH)
        existing_eval["company_name"] = existing_eval["company_name"].str.upper().str.strip()
        existing_scores = existing_eval[["company_name", "field", "executive_compensation_score"]].copy()
        combined_df = combined_df.merge(existing_scores, on=["company_name", "field"], how="left")
    else:
        combined_df["executive_compensation_score"] = pd.NA

    # Identify rows that still need scoring
    needs_scoring = combined_df[combined_df["executive_compensation_score"].isna()].copy()
    
    if not needs_scoring.empty:
        print(f"  [Evaluation] Found {len(needs_scoring)} fields requiring valuation. Starting judge calls...")
        scored_rows = evaluate_batch(needs_scoring)
        combined_df.set_index(["company_name", "field"], inplace=True)
        scored_rows.set_index(["company_name", "field"], inplace=True)
        combined_df.update(scored_rows)
        combined_df.reset_index(inplace=True)
    else:
        print("  [Evaluation] All fields are already scored in the cache. Skipping LLM judge calls.")

    # Save progress back to cache
    combined_df.to_csv(EVALUATION_CACHE_PATH, index=False)
    
    # ── Results & Visualization ──────────────────────────────────────────────
    print("\\nEvaluation Results:")
    print(combined_df[["company_name", "field", "executive_compensation_score"]].to_string(index=False))

    print("\\n" + "=" * 60)
    print("STEP 3: Generating Accuracy Heatmap")
    print("=" * 60)
    create_accuracy_heatmap(combined_df, save_path=HEATMAP_PATH)

    return combined_df


def main():
    parser = argparse.ArgumentParser(description="PageIndex Reasoning RAG Pipeline")
    parser.add_argument("--step", choices=["extract", "evaluate", "all"], default="all")
    args = parser.parse_args()

    if args.step in ("extract", "all"):
        report_df = step_extract()
    else:
        if EXTRACTION_CACHE_PATH.exists():
            report_df = pd.read_csv(EXTRACTION_CACHE_PATH)
        else:
            print("No cached extraction found. Run --step extract first.")
            return

    if args.step in ("evaluate", "all"):
        step_evaluate(report_df)


if __name__ == "__main__":
    main()
