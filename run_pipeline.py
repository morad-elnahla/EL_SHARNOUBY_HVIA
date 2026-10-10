"""Run the full project end to end.

    python run_pipeline.py            # data pipeline + analysis + models + figures
    python run_pipeline.py --skip-figures

Outputs: data/processed/*.parquet, outputs/tables/*.csv, outputs/figures/*.png,
outputs/models/*, outputs/tables/headline_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

import analysis as A  # noqa: E402
import data_pipeline  # noqa: E402
import modeling as M  # noqa: E402
import visuals as V  # noqa: E402
from utils import TABLES, ensure_dirs, save_fig, save_table  # noqa: E402

warnings.filterwarnings("ignore")
PROBLEM_SUPPLIERS = ["SUP04", "SUP09"]
BACKTEST_ORIGINS = ["2025-12-01", "2026-03-01", "2026-06-01"]


def build_results(d: dict[str, pd.DataFrame]) -> dict:
    """Compute every table and model output once; notebooks, dashboard and report all read from this."""
    s, c360 = d["sales"], d["customer_360"]
    R: dict = {}

    # ---- KPI / Case 01 / Case 03
    R["kpi_fy"] = A.kpi_by_period(s)
    R["kpi_total"] = A.kpi_total(s)
    R["monthly"] = A.monthly_trend(s)
    R["seasonality"] = A.seasonality_index(s, by_family=True)
    R["product_contribution"] = A.product_contribution(s)
    R["discount_impact"] = A.discount_impact(s)
    R["discount_leakage"] = A.discount_leakage(s)
    R["promo_test"] = A.promo_test(s)
    R["channel_scorecard"] = A.channel_scorecard(s)
    R["channel_growth"] = A.channel_growth(s)
    R["geo_demand"] = A.geo_demand(s)
    R["return_reasons"] = A.return_reasons(s)

    # ---- Case 02
    rfm = A.rfm_segments(c360)
    R["rfm"] = rfm
    R["rfm_segments"] = A.segment_summary(rfm)
    R["cohorts"] = A.cohort_retention(s)
    R["churn_pool"] = A.churn_pool(c360)
    R["interpurchase"] = A.interpurchase_gap(s)
    R["repeat_by_acquisition"] = A.repeat_metrics(c360, "Acquisition_Channel")
    R["repeat_by_type"] = A.repeat_metrics(c360, "Customer_Type")
    R["hv_governorate"] = A.high_value_profile(rfm, "Governorate")
    R["hv_age"] = A.high_value_profile(rfm, "Age_Band")

    # ---- Case 04
    diag = A.inventory_diagnosis(d["inventory"], s)
    R["inventory_diag"] = diag
    R["inventory_location"] = A.inventory_by_location(diag)
    R["stockout_month"] = A.stockout_by_month(d["inventory"])
    R["stockout_loss"] = A.stockout_margin_loss(d["inventory"], s)
    hist = M.build_history(d["inventory"], d["products"])
    bt = M.backtest(hist, BACKTEST_ORIGINS, 3)
    R["history"], R["backtest"] = hist, bt
    R["backtest_metrics"] = M.backtest_metrics(bt, ["Method"])
    R["backtest_by_horizon"] = M.backtest_metrics(bt, ["Method", "Horizon"])
    R["backtest_by_location"] = M.backtest_metrics(bt[bt.Method == "si_growth"], ["Location_ID"])
    fc12 = M.forecast_from_origin(hist, 12, "si_growth")
    R["forecast_12m"] = fc12
    R["forecast_horizons"] = M.horizon_summary(fc12)
    latest = d["inventory"][d["inventory"].Month == d["inventory"].Month.max()][["Location_ID", "Location_Name", "Product_ID", "Product_Name", "Closing_Stock", "Supplier_Lead_Time_Days"]]
    wape = R["backtest_by_location"].set_index("Location_ID").WAPE.to_dict()
    plan = M.replenishment_plan(fc12, latest, A.unit_costs(s), wape)
    R["replenishment"] = plan
    R["rebalancing"] = M.rebalancing_suggestions(plan)
    R["fy27_base"] = M.fy27_base(fc12, s)

    # ---- Case 05
    m = d["marketing"]
    R["marketing_platform"] = A.marketing_by(m, "Platform")
    R["marketing_product_family"] = A.marketing_by(m, ["Platform", "Product_Family"])
    R["marketing_campaigns"] = A.campaign_outliers(m)
    R["spend_response"] = A.spend_response(m)
    R["breakeven_roas"] = A.breakeven_roas(s)
    R["attribution"] = A.attribution_reconciliation(m, s)
    R["web_funnel"] = A.web_funnel(d["web_traffic"])
    R["load_time"] = A.load_time_effect(d["web_traffic"])

    # ---- Case 06-09
    p = d["production"]
    R["line_scorecard"], R["line_significance"] = A.line_scorecard(p), A.line_delay_significance(p)
    R["bottleneck"], R["line_bottleneck"] = A.bottleneck_pareto(p), A.line_bottleneck_matrix(p)
    R["production_monthly"] = A.production_monthly(p, d["group_operations"])
    R["overtime"] = A.overtime_effect(p)
    q = d["quality"]
    R["defect_pareto"], R["stage_summary"], R["root_cause"] = A.defect_pareto(q), A.stage_summary(q), A.root_cause_summary(q)
    R["stage_defect_loss"] = A.stage_defect_loss(q)
    R["quality_return_link"] = A.quality_return_link(q, s)
    R["copq"] = A.cost_of_poor_quality(q, d["printing"], s)
    R["product_quality_priority"] = A.product_quality_ranking(d["product_quality_returns"])
    R["machines"] = A.machine_scorecard(d["printing"], d["downtime"])
    R["machine_significance"] = A.machine_reject_significance(d["printing"])
    R["machine_link"] = A.machine_month_link(d["printing"], d["downtime"])
    R["downtime_reasons"], R["downtime_printing"] = A.downtime_reasons(d["downtime"]), A.downtime_reasons(d["downtime"], "Printing")
    R["downtime_area"] = d["downtime"].groupby("Business_Area").agg(Events=("Downtime_ID", "count"), Hours=("Downtime_Hours", "sum")).reset_index().sort_values("Hours", ascending=False)
    po = d["procurement"]
    R["suppliers"] = A.supplier_scorecard(po)
    R["supplier_actions"] = A.supplier_action_matrix(R["suppliers"])
    R["supplier_trend"] = A.supplier_otd_trend(po)
    R["materials"] = A.material_scorecard(po, d["bom"], p)
    R["procurement_trend"] = A.procurement_trend(po)
    R["otd_problem_split"] = A.otd_ex_problem_suppliers(po, PROBLEM_SUPPLIERS)

    # ---- Case 10
    chain = A.integrated_monthly(d)
    R["chain"], R["chain_corr"] = chain, A.chain_correlations(chain)
    R["supplier_to_production_lag"] = A.lagged_link(chain, "Supplier_OTD", "Production_Delay_Days")

    # ---- New opportunity: wholesale discount governance
    w = M.wholesale_lines(s)
    R["wholesale"] = w
    R["discount_diagnostic"] = M.discount_diagnostic(w)
    R["price_response"] = M.price_response_test(w)
    R["policy_table"] = M.policy_table(w)
    R["breakeven_volume_loss"] = {p_: M.breakeven_volume_loss(w, p_) for p_ in M.POLICIES}
    R["leakage_product"] = M.leakage_breakdown(w, "Volume-tiered cap", "Product_Family")
    R["leakage_customer"] = M.leakage_breakdown(w, "Volume-tiered cap", "Customer_ID").head(25)
    R["policy_artifact"] = M.save_policy_artifact(w)

    # ---- value at stake
    realloc = {k: A.reallocation_scenario(R["marketing_platform"], marginal_efficiency=e, haircut=h) for k, (e, h) in {"low": (0.25, 0.1815), "base": (0.5, 0.345), "high": (1.0, 1.0)}.items()}
    years = 3.0
    lost_gp = R["stockout_loss"].set_index("Location_Name").Lost_GP_EGP.sum()
    online_sizefit = s[(s.Channel == "Online") & (s.Return_Reason == "Size / fit")].Refund_Value_EGP.sum()
    at_risk_gp = R["rfm_segments"].set_index("Segment").loc["At Risk (high frequency)", "Gross_Profit"]
    context = {
        "winback_pool_annual_gp": at_risk_gp / years, "qc_loss_annual": q.Estimated_Loss_EGP.sum() / years,
        "online_sizefit_refund_annual": online_sizefit / years, "stockout_gp_annual": lost_gp / years,
        "online_gm": R["channel_scorecard"].set_index("Channel").loc["Online", "Gross_Margin"],
        "realloc_gp_low": realloc["low"]["net_gross_profit"] / years, "realloc_gp_base": realloc["base"]["net_gross_profit"] / years, "realloc_gp_high": realloc["high"]["net_gross_profit"] / years,
    }
    R["value_context"], R["reallocation"] = context, realloc
    R["value_at_stake"] = M.value_at_stake(s, w, context)
    return R


def headline_metrics(d: dict[str, pd.DataFrame], R: dict) -> dict:
    s = d["sales"]
    ch = R["channel_scorecard"].set_index("Channel")
    kfy = R["kpi_fy"].set_index("Fiscal_Year")
    seg = R["rfm_segments"].set_index("Segment")
    loc = R["inventory_location"].set_index("Location_Name")
    mk = R["marketing_platform"].set_index("Platform")
    lines = R["line_scorecard"].set_index("Line_ID")
    bn = R["bottleneck"].set_index("Primary_Bottleneck")
    copq = R["copq"]
    mach = R["machines"].set_index("Machine_ID")
    sup = R["suppliers"].set_index("Supplier_ID")
    pol = R["policy_table"]
    tier = pol[pol.Policy == "Volume-tiered cap"].set_index("Volume_Loss_On_Repriced")
    both = pol[pol.Policy == "Tiered cap + margin floor"].set_index("Volume_Loss_On_Repriced")
    cohorts = R["cohorts"]
    wh = loc.loc["El-Sharnouby Wholesale Desk"]
    bt = R["backtest_metrics"].set_index("Method")
    va = R["value_at_stake"]
    w = R["wholesale"]
    price = R["price_response"]
    dg = R["discount_diagnostic"]
    first_cohort_q1 = cohorts.iloc[0, 2]
    recent = cohorts[cohorts.index >= pd.Period("2025Q1")][1].mean()
    return {
        "period": f"{s.Order_Date.min():%Y-%m-%d} to {s.Order_Date.max():%Y-%m-%d}",
        "net_sales": float(s.Net_Sales_EGP.sum()), "gross_profit": float(s.Gross_Profit_EGP.sum()), "gross_margin": float(s.Gross_Profit_EGP.sum() / s.Net_Sales_EGP.sum()),
        "units": int(s.Quantity.sum()), "invoices": int(len(s)), "customers": int(s.Customer_ID.nunique()),
        "net_sales_fy": {k: float(v) for k, v in kfy.Net_Sales.items()}, "gp_fy": {k: float(v) for k, v in kfy.Gross_Profit.items()}, "gm_fy": {k: float(v) for k, v in kfy.Gross_Margin.items()},
        "net_growth_fy25": float(kfy.Net_Sales["FY25"] / kfy.Net_Sales["FY24"] - 1), "net_growth_fy26": float(kfy.Net_Sales["FY26"] / kfy.Net_Sales["FY25"] - 1),
        "gp_growth_fy26": float(kfy.Gross_Profit["FY26"] / kfy.Gross_Profit["FY25"] - 1),
        "channel": {c: {"net_share": float(r.Net_Share), "gp_share": float(r.GP_Share), "gm": float(r.Gross_Margin), "refund_rate": float(r.Refund_Rate), "gp_per_unit": float(r.GP_per_Unit)} for c, r in ch.iterrows()},
        "wholesale_loss_making_lines": int(s.Is_Loss_Making.sum()), "wholesale_loss_amount": float(s.loc[s.Is_Loss_Making, "Gross_Profit_EGP"].sum()),
        "wholesale_discount_given": float(R["discount_leakage"].set_index("Channel").loc["Wholesale", "Discount_Given"]),
        "promo_2026": {"discount_promo": float(R["promo_test"].iloc[-1].Discount_Promo), "volume_lift": float(R["promo_test"].iloc[-1].Volume_Lift), "gp_per_day_change": float(R["promo_test"].iloc[-1].GP_per_Day_Change)},
        "retail_customers": int(R["rfm"].shape[0]), "repeat_rate_retail": float(R["repeat_by_type"].set_index("Customer_Type").loc["Retail", "Repeat_Rate"]),
        "at_risk_hf_customers": int(seg.loc["At Risk (high frequency)", "Customers"]), "at_risk_hf_customer_share": float(seg.loc["At Risk (high frequency)", "Customer_Share"]),
        "at_risk_hf_sales_share": float(seg.loc["At Risk (high frequency)", "Sales_Share"]), "at_risk_hf_net": float(seg.loc["At Risk (high frequency)", "Net_Sales"]),
        "champions_sales_share": float(seg.loc["Champions", "Sales_Share"]), "lapsed_retail_customers": int(R["churn_pool"].set_index("Customer_Type").loc["Retail", "Lapsed"]),
        "retention_q1_first_cohort": float(first_cohort_q1), "retention_q1_recent_cohorts": float(recent),
        "wholesale_cover_days": float(wh.Median_Cover_Days), "retail_cover_days": float(loc.loc["LEO MODA - Sidi Bishr Store", "Median_Cover_Days"]), "online_cover_days": float(loc.loc["LEO MODA Online Store", "Median_Cover_Days"]),
        "excess_stock_value": float(loc.Excess_Value.sum()), "stock_value": float(loc.Stock_Value.sum()), "wholesale_skus_below_rop": int(wh.Below_ROP_or_Stockout),
        "wholesale_lost_gp_3y": float(R["stockout_loss"].set_index("Location_Name").loc["El-Sharnouby Wholesale Desk", "Lost_GP_EGP"]),
        "forecast_wape_si_growth": float(bt.loc["si_growth", "WAPE"]), "forecast_wape_seasonal_naive": float(bt.loc["seasonal_naive", "WAPE"]), "forecast_wape_naive3": float(bt.loc["naive3", "WAPE"]),
        "replenishment_order_value": float(R["replenishment"].Order_Value_EGP.sum()), "replenishment_surplus_value": float(R["replenishment"].Surplus_Value_EGP.sum()),
        "forecast_fy27_units": float(R["forecast_12m"].Units_Forecast.sum()), "fy27_base_net": float(R["fy27_base"].Net_Sales.sum()), "fy27_base_gp": float(R["fy27_base"].Gross_Profit.sum()),
        "breakeven_roas": float(R["breakeven_roas"]), "attribution_overclaim": float(R["attribution"]["overclaim_vs_online"]),
        "roas": {p: float(r.ROAS) for p, r in mk.iterrows()}, "spend_share": {p: float(r.Spend_Share) for p, r in mk.iterrows()}, "revenue_share": {p: float(r.Revenue_Share) for p, r in mk.iterrows()},
        "line04_delay": float(lines.loc["LINE-04", "Delayed_Share"]), "other_lines_delay": float(lines.drop("LINE-04").Delayed_Share.mean()),
        "sewing_finishing_delay_share": float(bn.loc[["Sewing", "Finishing"], "Delay_Share"].sum()),
        "copq_total": float(copq.EGP.sum()), "copq_pct_gp": float(copq.Pct_of_Gross_Profit.sum()), "qc_loss": float(d["quality"].Estimated_Loss_EGP.sum()),
        "unclassified_loss_share": float(R["defect_pareto"].set_index("Defect_Type").loc["Unclassified", "Loss_Share"]),
        "size_fit_return_share": float(R["return_reasons"].groupby("Return_Reason").Lines.sum().div(R["return_reasons"].Lines.sum())["Size / fit"]),
        "pr_m03_reject": float(mach.loc["PR-M03", "Reject_Rate"]), "other_printers_reject": float(mach.drop("PR-M03").Reject_Rate.mean()), "pr_m02_downtime_h": float(mach.loc["PR-M02", "Downtime_Hours"]),
        "unplanned_downtime_share": float(R["downtime_reasons"].query("Planned == 'No'").Hours.sum() / R["downtime_reasons"].Hours.sum()),
        "sup04_otd": float(sup.loc["SUP04", "OTD"]), "sup09_otd": float(sup.loc["SUP09", "OTD"]), "other_sup_otd": float(sup.drop(["SUP04", "SUP09"]).OTD.mean()),
        "problem_supplier_spend_share": float(sup.loc[["SUP04", "SUP09"], "Spend_Share"].sum()),
        "wholesale_gm": float(ch.loc["Wholesale", "Gross_Margin"]), "wholesale_discount_r2_model": float(dg["model_r2"]), "wholesale_discount_r2_baseline": float(dg["baseline_r2"]),
        "price_response_coef": float(price.iloc[0].Coefficient), "price_response_p": float(price.iloc[0].p_value), "price_response_within_p": float(price.iloc[1].p_value),
        "mean_wholesale_discount": float(w.Discount_Pct.mean()),
        "policy_tier": {"lines_repriced_share": float(tier.loc[0.0, "Share_Lines_Repriced"]), "gm_new": float(tier.loc[0.0, "New_GM"]), "uplift_3y_0": float(tier.loc[0.0, "GP_Uplift"]), "uplift_year_0": float(tier.loc[0.0, "GP_Uplift_per_Year"]),
                        "uplift_year_10": float(tier.loc[0.1, "GP_Uplift_per_Year"]), "uplift_year_20": float(tier.loc[0.2, "GP_Uplift_per_Year"]), "breakeven_loss": float(R["breakeven_volume_loss"]["Volume-tiered cap"]),
                        "loss_making_after": int(tier.loc[0.0, "Loss_Making_Lines_After"])},
        "policy_both": {"gm_new": float(both.loc[0.0, "New_GM"]), "uplift_year_0": float(both.loc[0.0, "GP_Uplift_per_Year"]), "uplift_year_10": float(both.loc[0.1, "GP_Uplift_per_Year"]), "uplift_year_20": float(both.loc[0.2, "GP_Uplift_per_Year"]),
                        "breakeven_loss": float(R["breakeven_volume_loss"]["Tiered cap + margin floor"])},
        "value_at_stake": {r.Action: {"low": float(r.Low), "base": float(r.Base), "high": float(r.High)} for r in va.itertuples()},
        "value_at_stake_total_base": float(va.Base.sum()),
    }


def make_figures(d: dict[str, pd.DataFrame], R: dict) -> list[str]:
    F = {
        "01_revenue_margin_trend": V.revenue_margin_trend(R["monthly"]),
        "01_channel_revenue_vs_profit": V.channel_share_revenue_vs_profit(R["channel_scorecard"]),
        "01_seasonality": V.seasonality_by_group(R["seasonality"]),
        "01_discount_band_margin": V.discount_band_margin(R["discount_impact"]),
        "01_product_gp_contribution": V.product_gp_contribution(R["product_contribution"]),
        "02_rfm_segments": V.rfm_share(R["rfm_segments"]),
        "02_cohort_retention": V.cohort_heatmap(R["cohorts"]),
        "03_channel_scorecard": V.channel_scorecard_panels(R["channel_scorecard"]),
        "03_channel_growth": V.channel_growth_lines(R["channel_growth"]),
        "03_geo_demand": V.geo_stack(R["geo_demand"]),
        "04_inventory_cover": V.inventory_cover_by_location(R["inventory_diag"]),
        "04_abc_xyz": V.abc_xyz_matrix(R["inventory_diag"]),
        "04_forecast": V.forecast_vs_actual(R["history"], R["forecast_12m"], R["backtest"]),
        "04_wholesale_stockouts": V.stockout_timeline(R["stockout_month"]),
        "05_platform_roas": V.platform_roas(R["marketing_platform"], R["breakeven_roas"]),
        "05_campaign_scatter": V.campaign_scatter(R["marketing_campaigns"]),
        "05_web_funnel": V.web_funnel_panels(R["web_funnel"], R["load_time"]),
        "06_line_delay": V.line_delay_chart(R["line_scorecard"], R["line_significance"]),
        "06_bottleneck_pareto": V.bottleneck_pareto_chart(R["bottleneck"]),
        "06_overtime_delay": V.overtime_delay_chart(R["overtime"]),
        "07_defect_pareto": V.defect_pareto_chart(R["defect_pareto"]),
        "07_copq": V.copq_chart(R["copq"]),
        "07_return_reasons": V.return_reason_chart(R["return_reasons"]),
        "08_printer_priority": V.printer_priority_chart(R["machines"]),
        "08_downtime_reasons": V.downtime_reason_chart(R["downtime_reasons"]),
        "09_supplier_otd_trend": V.supplier_otd_trend_chart(R["supplier_trend"], PROBLEM_SUPPLIERS),
        "09_supplier_matrix": V.supplier_matrix_chart(R["supplier_actions"]),
        "10_value_at_stake": V.value_at_stake_chart(R["value_at_stake"]),
        "10_chain_correlation": V.chain_corr_heatmap(R["chain_corr"]),
        "11_discount_vs_quantity": V.discount_vs_quantity(R["wholesale"]),
        "11_policy_uplift": V.policy_uplift_chart(R["policy_table"]),
        "11_discount_diagnostic": V.diagnostic_importance_chart(R["discount_diagnostic"]),
    }
    return [str(save_fig(fig, name)) for name, fig in F.items()]


def save_tables(R: dict) -> None:
    for name in ["kpi_fy", "channel_scorecard", "channel_growth", "product_contribution", "discount_impact", "promo_test", "rfm_segments", "cohorts", "inventory_location",
                 "backtest_metrics", "backtest_by_horizon", "forecast_horizons", "replenishment", "rebalancing", "marketing_platform", "marketing_campaigns", "spend_response",
                 "line_scorecard", "bottleneck", "defect_pareto", "copq", "machines", "suppliers", "supplier_actions", "materials", "price_response", "policy_table", "leakage_product",
                 "leakage_customer", "value_at_stake", "product_quality_priority", "downtime_reasons", "procurement_trend"]:
        save_table(R[name], name, index=name == "cohorts")
    save_table(R["inventory_diag"].drop(columns=[]), "inventory_diagnosis")
    save_table(R["discount_diagnostic"]["importance"], "discount_model_importance")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    print("[1/4] Data pipeline")
    data_pipeline.run()
    d = A.load_all()
    print("[2/4] Analysis and models")
    R = build_results(d)
    save_tables(R)
    metrics = headline_metrics(d, R)
    (TABLES / "headline_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    if not args.skip_figures:
        print("[3/4] Figures")
        paths = make_figures(d, R)
        print(f"      {len(paths)} figures written")
    print("[4/4] Done. Net sales {:,.0f} EGP | GP {:,.0f} EGP | value at stake (base) {:,.0f} EGP/yr".format(metrics["net_sales"], metrics["gross_profit"], metrics["value_at_stake_total_base"]))


if __name__ == "__main__":
    main()
