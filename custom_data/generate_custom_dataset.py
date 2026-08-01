"""
generate_custom_dataset.py
Generates FinQA-style numerical questions from verified 10-K figures.
All answers are computed in Python directly from the verified figures,
never by an AI, so every answer traces back to official source data.
Handles the different field names each company uses in its filing.
Output is saved as JSON that custom_loader.py reads.
"""
import os
import sys
import json

from custom_dataset_source import COMPANY_DATA

# every name a company might use for its top-line revenue
REVENUE_KEYS = [
    "total_net_sales", "total_revenue", "total_revenues",
    "net_operating_revenues", "total_net_revenue",
    "total_revenues_and_other_income",
]


def get_revenue(figures: dict):
    for key in REVENUE_KEYS:
        if key in figures:
            return figures[key]
    return None


def build_context(company: dict) -> str:
    f = company["figures"]
    prior = company.get("prior_year_figures", {})
    fy = company["fiscal_year"]
    prior_fy = int(fy) - 1
    lines = [f"{company['company']} fiscal year {fy} "
             f"financial results (in millions of USD):"]
    for key, value in f.items():
        readable = key.replace("_", " ")
        lines.append(f"- {readable}: {value}")
    if prior:
        lines.append("")
        lines.append(f"{company['company']} fiscal year {prior_fy} "
                     f"figures for comparison (in millions of USD):")
        for key, value in prior.items():
            readable = key.replace("_", " ")
            lines.append(f"- {readable} ({prior_fy}): {value}")
    return "\n".join(lines)


def make_questions(company: dict) -> list[dict]:
    f = company["figures"]
    prior = company.get("prior_year_figures", {})
    name = company["company"]
    fy = company["fiscal_year"]
    ticker = company["ticker"].lower()

    revenue = get_revenue(f)
    prior_revenue = get_revenue(prior)

    questions = []

    # --- Direct lookups ---
    if revenue is not None:
        questions.append({
            "question": f"What was {name}'s total revenue in fiscal {fy}, in millions?",
            "answer": str(revenue),
        })
    for key, label in [
        ("net_income", "net income"),
        ("operating_income", "operating income"),
        ("research_and_development", "research and development expense"),
        ("gross_profit", "gross profit"),
        ("gross_margin", "gross margin"),
    ]:
        if key in f:
            questions.append({
                "question": f"What was {name}'s {label} in fiscal {fy}, in millions?",
                "answer": str(f[key]),
            })

    # --- Margin calculations (answers as decimals) ---
    if "net_income" in f and revenue:
        questions.append({
            "question": f"What was {name}'s net income as a proportion of total "
                        f"revenue in fiscal {fy}? Express as a decimal.",
            "answer": f"{f['net_income'] / revenue:.5f}",
        })
    if "operating_income" in f and revenue:
        questions.append({
            "question": f"What was {name}'s operating margin in fiscal {fy}? "
                        f"Express as a decimal.",
            "answer": f"{f['operating_income'] / revenue:.5f}",
        })
    if "gross_margin" in f and revenue:
        questions.append({
            "question": f"What was {name}'s gross margin as a proportion of total "
                        f"revenue in fiscal {fy}? Express as a decimal.",
            "answer": f"{f['gross_margin'] / revenue:.5f}",
        })
    if "gross_profit" in f and revenue:
        questions.append({
            "question": f"What was {name}'s gross profit as a proportion of total "
                        f"revenue in fiscal {fy}? Express as a decimal.",
            "answer": f"{f['gross_profit'] / revenue:.5f}",
        })
    if "research_and_development" in f and revenue:
        questions.append({
            "question": f"What was {name}'s research and development expense as a "
                        f"proportion of total revenue in fiscal {fy}? Express as a decimal.",
            "answer": f"{f['research_and_development'] / revenue:.5f}",
        })

    # --- Segment / product share questions ---
    segment_keys = {
        "iphone_sales": "iPhone",
        "services_sales": "Services",
        "mac_sales": "Mac",
        "automotive_revenue": "automotive",
        "intelligent_cloud_revenue": "Intelligent Cloud",
        "productivity_business_processes_revenue": "Productivity and Business Processes",
        "north_america_revenue": "North America",
        "greater_china_revenue": "Greater China",
        "net_interest_income": "net interest income",
    }
    for key, label in segment_keys.items():
        if key in f and revenue:
            questions.append({
                "question": f"What proportion of {name}'s total revenue came from "
                            f"{label} in fiscal {fy}? Express as a decimal.",
                "answer": f"{f[key] / revenue:.5f}",
            })

    # --- Year over year growth ---
    if "net_income" in f and "net_income" in prior and prior["net_income"] != 0:
        growth = (f["net_income"] - prior["net_income"]) / abs(prior["net_income"])
        questions.append({
            "question": f"What was the growth rate in {name}'s net income from "
                        f"fiscal {int(fy)-1} to fiscal {fy}? Express as a decimal.",
            "answer": f"{growth:.5f}",
        })
    if revenue and prior_revenue:
        growth = (revenue - prior_revenue) / prior_revenue
        questions.append({
            "question": f"What was the growth rate in {name}'s total revenue from "
                        f"fiscal {int(fy)-1} to fiscal {fy}? Express as a decimal.",
            "answer": f"{growth:.5f}",
        })
    if "operating_income" in f and "operating_income" in prior and prior["operating_income"] != 0:
        growth = (f["operating_income"] - prior["operating_income"]) / abs(prior["operating_income"])
        questions.append({
            "question": f"What was the growth rate in {name}'s operating income from "
                        f"fiscal {int(fy)-1} to fiscal {fy}? Express as a decimal.",
            "answer": f"{growth:.5f}",
        })

    # attach context and ids
    context = build_context(company)
    items = []
    for i, q in enumerate(questions):
        items.append({
            "id": f"custom-{ticker}-{i}",
            "context": context,
            "question": q["question"],
            "answer": q["answer"],
            "company": name,
            "source": company["source"],
        })
    return items


def main():
    all_items = []
    for company in COMPANY_DATA:
        items = make_questions(company)
        all_items.extend(items)
        print(f"Generated {len(items)} questions for {company['company']}")

    os.makedirs("fineval/datasets/custom", exist_ok=True)
    out_path = "fineval/datasets/custom/custom_dataset.json"
    with open(out_path, "w") as f:
        json.dump(all_items, f, indent=2)

    print(f"\nTotal: {len(all_items)} questions saved to {out_path}")


if __name__ == "__main__":
    main()