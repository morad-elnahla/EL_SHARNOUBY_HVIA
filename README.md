# EL SHARNOUBY GROUP - HVIA Project

End-to-end business discovery, analytics and AI prototype for EL SHARNOUBY GROUP (LEO MODA retail/online, Wholesale Desk, LEO Production Hub / LEO PRINT).

## Business objective
Find out whether growth is converting into profit, explain *why* across sales, customers, marketing, inventory, factory, quality, printing and suppliers, solve the 10 HVIA case briefs with data, and identify one strong opportunity that was not requested.

## Main findings
- Sales EGP 421.3M, gross profit EGP 64.9M (15.4%). Growth slowed from +10.8% (FY25) to +3.3% (FY26); margin is flat.
- **Wholesale = 85% of sales but 11.9% margin** (retail 37%, online 35%). Unit cost is identical across channels; the gap is discount.
- Inventory is mirrored: wholesale 11 days of cover (36/59 SKUs below reorder point) vs ~115 days in retail/online (EGP 7.5M excess).
- Two suppliers (SUP04, SUP09) fell from ~65% to ~15% on-time in 2025; factory delay is driven by Sewing/Finishing and LINE-04, not by suppliers.
- Cost of poor quality is about EGP 20.3M (31% of gross profit); marketing attribution over-claims online revenue 5.5x.

## Main opportunity (not requested): Wholesale Discount Governance
Wholesale discounts (28-38%) have no relationship to order size, customer, product or cost (a 10-feature model gets R2 = 0.008), buy no extra volume (p = 0.58), and cause 4,876 below-cost invoices.

## Main solution: Wholesale Deal Desk
Volume-tiered discount cap (28/30/32%), 15% margin floor and approval routing. Simulated on three years of invoices: **+EGP 2.9-4.9M gross profit per year (Phase 1)**, +EGP 5.5-8.8M with Phase 2; still profitable if re-priced customers cut volume by 48%.

## Project structure
```
EL_SHARNOUBY_HVIA/
├── README.md
├── requirements.txt
├── run_pipeline.py            # one command: pipeline -> analysis -> models -> figures -> tables
├── data/
│   ├── raw/                   # original files (untouched)
│   └── processed/             # cleaned/integrated parquet datasets (generated)
├── src/
│   ├── utils.py               # paths, formatting, plot style
│   ├── data_pipeline.py       # validation, cleaning, integration, analytical datasets
│   ├── analysis.py            # KPI framework + analysis for the 10 case briefs
│   ├── modeling.py            # demand forecast, replenishment, discount-governance engine, scenarios
│   └── visuals.py             # figures
├── notebooks/                 # 01 data prep | 02 business analysis | 03 case briefs | 04 opportunity & solution
├── dashboard/app.py           # Streamlit executive dashboard (7 sections)
├── outputs/
│   ├── figures/  tables/  models/
└── reports/final_report.md
```
`visuals.py` is separate from `analysis.py` so the dashboard can reuse the analysis without importing plotting code.

## Setup
Python 3.11+ (tested on 3.12).
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python run_pipeline.py             # about 1 minute; add --skip-figures to skip PNGs
streamlit run dashboard/app.py     # opens http://localhost:8501
```
Notebooks (already executed; re-run if you wish):
```bash
jupyter lab notebooks/
# or: jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
```

## Notes on the data
The dataset is officially *modelled*. Documented reconciliation gaps: marketing-attributed revenue is 5.5x online sales; web orders < online invoices; production-order volume is 3.8x units sold; procurement does not match BOM quantities. The analysis uses rates and rankings where absolute values do not reconcile (see `outputs/tables/data_quality_report.csv` and section 11 of the report).
