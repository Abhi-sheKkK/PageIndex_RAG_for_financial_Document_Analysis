"""
Visualization — accuracy heatmap for LLM evaluation scores (PageIndex Pipeline).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_accuracy_heatmap(df: pd.DataFrame, save_path: str = None):
    """
    Create a heatmap of LLM accuracy scores by company × field.
    """
    df = df.copy()
    
    # Ensure scores are numeric to avoid "dtype object" errors during plotting
    df["executive_compensation_score"] = pd.to_numeric(df["executive_compensation_score"], errors="coerce")
    # Fill any gaps with 0.0 or leave as NaN if you want them invisible
    df["executive_compensation_score"] = df["executive_compensation_score"].fillna(0.0)

    field_rename_map = {
        "company_ceo": "CEO",
        "coverage_period": "Coverage Period",
        "total_target_lti": "Total Target LTI",
        "lti_grant_date": "LTI Grant Date",
        "annual_lti_grant": "Annual LTI Grant?",
        "time_based_rsu_vesting_schedule": "Time-Based RSU Vesting",
        "performance_based_rsu_vesting_schedule": "Performance-Based RSU Vesting",
        "compensation_governance_arrangements": "Governance Arrangements",
        "ceo_pay_alignment_mechanisms": "CEO Pay Alignment",
        "performance_metrics_used": "Performance Metrics",
        "realized_base_salary": "Realized Base Salary",
        "realized_stis": "Realized STIs",
        "realized_long_term_awards": "Realized LT Awards",
        "realized_other_compensation": "Realized Other Compensation",
        "realized_total_compensation": "Realized Total Compensation",
    }

    df["field"] = df["field"].map(field_rename_map)

    correct_field_order = [
        "CEO", "Coverage Period", "Total Target LTI", "LTI Grant Date",
        "Annual LTI Grant?", "Time-Based RSU Vesting", "Performance-Based RSU Vesting",
        "Governance Arrangements", "CEO Pay Alignment", "Performance Metrics",
        "Realized Base Salary", "Realized STIs", "Realized LT Awards",
        "Realized Other Compensation", "Realized Total Compensation",
    ]

    df["field"] = pd.Categorical(df["field"], categories=correct_field_order, ordered=True)

    pivot_df = df.pivot(
        index="company_name",
        columns="field",
        values="executive_compensation_score",
    )

    plt.figure(figsize=(14, 6))
    sns.heatmap(
        pivot_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        linewidths=0.5,
        cbar_kws={"label": "Accuracy Score"},
        vmin=0,
        vmax=1,
    )
    plt.title("LLM Accuracy Heatmap — Executive Compensation Extraction (PageIndex Reasoning RAG)")
    plt.xlabel("Variable")
    plt.ylabel("Company")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()
