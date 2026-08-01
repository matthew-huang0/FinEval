"""
custom_dataset_source.py
Real verified figures from SEC EDGAR 10-K filings.
All numbers are pulled from official company 10-K annual reports.
Questions are generated from these figures using generate_custom_dataset.py.
All answers are computed in Python directly from these verified figures,
so every answer traces back to official source data.
"""

# All figures in millions of USD.
# Source: official 10-K filings on SEC EDGAR.

COMPANY_DATA = [
    {
        "company": "Apple Inc.",
        "ticker": "AAPL",
        "fiscal_year": "2023",
        "source": "SEC EDGAR 10-K filed 2023-11-03",
        "figures": {
            "total_net_sales": 383285,
            "iphone_sales": 200583,
            "mac_sales": 29357,
            "ipad_sales": 28300,
            "wearables_home_accessories_sales": 39845,
            "services_sales": 85200,
            "total_cost_of_sales": 214137,
            "gross_margin": 169148,
            "research_and_development": 29915,
            "selling_general_administrative": 24932,
            "total_operating_expenses": 54847,
            "operating_income": 114301,
            "net_income": 96995,
            "provision_for_income_taxes": 16741,
        },
        "prior_year_figures": {
            "total_net_sales": 394328,
            "net_income": 99803,
            "operating_income": 119437,
        },
    },
    {
        "company": "Microsoft Corporation",
        "ticker": "MSFT",
        "fiscal_year": "2023",
        "source": "SEC EDGAR 10-K filed 2023-07-27",
        "figures": {
            "total_revenue": 211915,
            "productivity_business_processes_revenue": 69274,
            "intelligent_cloud_revenue": 87907,
            "more_personal_computing_revenue": 54734,
            "gross_margin": 146052,
            "operating_income": 88523,
            "net_income": 72361,
        },
        "prior_year_figures": {
            "total_revenue": 198270,
            "net_income": 72738,
            "operating_income": 83383,
        },
    },
    {
        "company": "Tesla, Inc.",
        "ticker": "TSLA",
        "fiscal_year": "2023",
        "source": "SEC EDGAR 10-K filed 2024-01-26",
        "figures": {
            "total_revenue": 96773,
            "automotive_revenue": 82419,
            "energy_generation_storage_revenue": 6035,
            "services_and_other_revenue": 8319,
            "gross_profit": 17660,
            "research_and_development": 3969,
            "operating_income": 8891,
            "net_income": 14997,
        },
        "prior_year_figures": {
            "total_revenue": 81462,
            "net_income": 12556,
            "operating_income": 13656,
        },
    },
    {
        "company": "The Coca-Cola Company",
        "ticker": "KO",
        "fiscal_year": "2023",
        "source": "SEC EDGAR 10-K filed 2024-02-20",
        "figures": {
            "net_operating_revenues": 45754,
            "cost_of_goods_sold": 18520,
            "gross_profit": 27234,
            "selling_general_administrative": 13972,
            "operating_income": 11311,
            "net_income": 10714,
        },
        "prior_year_figures": {
            "net_operating_revenues": 43004,
            "net_income": 9542,
            "operating_income": 10909,
        },
    },
    {
        "company": "NIKE, Inc.",
        "ticker": "NKE",
        "fiscal_year": "2023",
        "source": "SEC EDGAR 10-K filed 2023-07-20",
        "figures": {
            "total_revenues": 51217,
            "north_america_revenue": 21608,
            "emea_revenue": 13418,
            "greater_china_revenue": 7248,
            "gross_profit": 22292,
            "operating_income": 5915,
            "net_income": 5070,
        },
        "prior_year_figures": {
            "total_revenues": 46710,
            "net_income": 6046,
            "operating_income": 6935,
        },
    },
    {
        "company": "Amazon.com, Inc.",
        "ticker": "AMZN",
        "fiscal_year": "2023",
        "source": "SEC EDGAR 10-K filed 2024-02-01",
        "figures": {
            "total_net_sales": 574785,
            "gross_profit": 270046,
            "operating_income": 36852,
            "net_income": 30425,
        },
        "prior_year_figures": {
            "total_net_sales": 513983,
            "net_income": -2722,
            "operating_income": 12248,
        },
    },
    {
        "company": "JPMorgan Chase & Co.",
        "ticker": "JPM",
        "fiscal_year": "2023",
        "source": "SEC EDGAR 10-K filed 2024-02-16",
        "figures": {
            "total_net_revenue": 158104,
            "net_interest_income": 89267,
            "noninterest_revenue": 68837,
            "noninterest_expense": 87172,
            "net_income": 49552,
        },
        "prior_year_figures": {
            "total_net_revenue": 128695,
            "net_income": 37676,
        },
    },
    {
        "company": "Walmart Inc.",
        "ticker": "WMT",
        "fiscal_year": "2023",
        "source": "SEC EDGAR 10-K filed 2023-03-17",
        "figures": {
            "total_revenues": 611289,
            "net_sales": 605881,
            "operating_income": 20428,
            "net_income": 11292,
        },
        "prior_year_figures": {
            "total_revenues": 572754,
            "net_income": 13940,
            "operating_income": 25942,
        },
    },
    {
        "company": "Exxon Mobil Corporation",
        "ticker": "XOM",
        "fiscal_year": "2023",
        "source": "SEC EDGAR 10-K filed 2024-02-28",
        "figures": {
            "total_revenues_and_other_income": 344582,
            "net_income": 36010,
        },
        "prior_year_figures": {
            "total_revenues_and_other_income": 413680,
            "net_income": 55740,
        },
    },
]