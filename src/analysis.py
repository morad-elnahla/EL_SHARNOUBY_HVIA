"""Business analysis: KPI framework and one function group per Case Brief.

Every function takes processed dataframes and returns tidy dataframes so the same
logic feeds the notebooks, the dashboard and the final report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from utils import AS_OF_DATE, read_processed, safe_div

DATASETS = [
    "sales", "customer_360", "demand_monthly", "product_quality_returns", "inventory", "procurement",
    "production", "printing", "downtime", "quality", "marketing", "web_traffic", "customer_service",
    "products", "locations", "suppliers", "materials", "bom", "group_operations",
]
FACTORY_DAILY_CAPACITY = 5_000  # LEO Production Hub, public figure (Business_Units.csv)
WORKING_DAYS_PER_MONTH = 26


def load_all() -> dict[str, pd.DataFrame]:
    return {name: read_processed(name) for name in DATASETS}


def pct_change_safe(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.pct_change(periods=periods, fill_method=None)


# ============================================================ KPI FRAMEWORK
KPI_DEFINITIONS = pd.DataFrame(
    [
        ("Net Sales", "Sum of Net_Sales_EGP (after discount, before refunds)", "Sales_Transactions"),
        ("Gross Profit", "Net sales - estimated COGS", "Sales_Transactions"),
        ("Gross Margin %", "Gross profit / Net sales", "Sales_Transactions"),
        ("Avg Discount %", "1 - Net sales / Gross sales (value-weighted)", "Sales_Transactions"),
        ("AOV", "Net sales / invoices", "Sales_Transactions"),
        ("Refund Rate", "Refund value / Net sales", "Sales_Transactions"),
        ("Repeat Rate", "Customers with >1 order / customers with >=1 order", "Customer_360"),
        ("ROAS", "Attributed revenue / Spend (platform-reported)", "Marketing"),
        ("CAC", "Spend / attributed orders", "Marketing"),
        ("Web Conversion", "Orders / Sessions", "Web_Traffic_Daily"),
        ("Stockout Rate", "SKU-location-months with >=1 stockout day / all", "Inventory_History"),
        ("Inventory Turns", "12M units sold / average monthly closing stock", "Inventory_History"),
        ("Days of Cover", "Closing stock / avg daily sales (90D)", "Inventory_History"),
        ("Plan Attainment", "Produced / planned quantity", "Production_Orders"),
        ("Production Efficiency", "Mean Efficiency_Pct of production orders", "Production_Orders"),
        ("On-Time Delay Rate", "Share of orders completed after due date", "Production_Orders"),
        ("Downtime Hours", "Sum Downtime_Hours (planned and unplanned)", "Machine_Downtime"),
        ("Defect (Reject) Rate", "Rejected / inspected units", "Quality"),
        ("Cost of Poor Quality", "QC loss + print rework/waste + quality refunds", "Quality, Printing, Sales"),
        ("Supplier OTD", "POs received on/before promised date / all POs", "Purchase_Orders"),
        ("Supplier OTIF", "On time AND reject rate <= 2%", "Purchase_Orders"),
        ("Lead Time", "Receipt date - order date (days)", "Purchase_Orders"),
    ],
    columns=["KPI", "Definition", "Source"],
)


def kpi_by_period(sales: pd.DataFrame, by: str | list[str] = "Fiscal_Year") -> pd.DataFrame:
    g = sales.groupby(by).agg(
        Net_Sales=("Net_Sales_EGP", "sum"), Gross_Sales=("Gross_Sales_EGP", "sum"), Gross_Profit=("Gross_Profit_EGP", "sum"),
        Units=("Quantity", "sum"), Invoices=("Invoice_ID", "count"), Customers=("Customer_ID", "nunique"),
        Refunds=("Refund_Value_EGP", "sum"), Loss_Making_Invoices=("Is_Loss_Making", "sum"),
    )
    g["Gross_Margin"] = g.Gross_Profit / g.Net_Sales
    g["Avg_Discount"] = 1 - g.Net_Sales / g.Gross_Sales
    g["AOV"] = g.Net_Sales / g.Invoices
    g["Refund_Rate"] = g.Refunds / g.Net_Sales
    g["Loss_Making_Share"] = g.Loss_Making_Invoices / g.Invoices
    g["GP_per_Unit"] = g.Gross_Profit / g.Units
    return g.reset_index()


def kpi_total(sales: pd.DataFrame) -> dict:
    row = kpi_by_period(sales.assign(All="All"), "All").iloc[0].to_dict()
    row.pop("All", None)
    return row


# ======================================================= CASE 01 - EXECUTIVE
def monthly_trend(sales: pd.DataFrame) -> pd.DataFrame:
    m = sales.groupby("Month").agg(Net_Sales=("Net_Sales_EGP", "sum"), Gross_Profit=("Gross_Profit_EGP", "sum"), Units=("Quantity", "sum")).reset_index()
    m["Gross_Margin"] = m.Gross_Profit / m.Net_Sales
    m["MoM"] = pct_change_safe(m.Net_Sales)
    m["YoY"] = pct_change_safe(m.Net_Sales, 12)
    m["Rolling_12M_Net"] = m.Net_Sales.rolling(12).sum()
    return m


def seasonality_index(sales: pd.DataFrame, by_family: bool = False) -> pd.DataFrame:
    """Average of (month units / fiscal-year monthly mean), per calendar month."""
    keys = ["Fiscal_Year", "Month"] + (["Product_Family"] if by_family else [])
    m = sales.groupby(keys).Quantity.sum().reset_index()
    base_keys = ["Fiscal_Year"] + (["Product_Family"] if by_family else [])
    m["Index"] = m.Quantity / m.groupby(base_keys).Quantity.transform("mean")
    m["Calendar_Month"] = m.Month.dt.month
    return m.groupby(["Calendar_Month"] + (["Product_Family"] if by_family else []), as_index=False).Index.mean()


def product_contribution(sales: pd.DataFrame) -> pd.DataFrame:
    p = sales.groupby(["Product_ID", "Product_Name", "Product_Family"]).agg(
        Net_Sales=("Net_Sales_EGP", "sum"), Gross_Profit=("Gross_Profit_EGP", "sum"), Units=("Quantity", "sum"),
        Gross_Sales=("Gross_Sales_EGP", "sum"), Refunds=("Refund_Value_EGP", "sum"),
    ).reset_index()
    p["Gross_Margin"] = p.Gross_Profit / p.Net_Sales
    p["Avg_Discount"] = 1 - p.Net_Sales / p.Gross_Sales
    p["Net_Share"] = p.Net_Sales / p.Net_Sales.sum()
    p["GP_Share"] = p.Gross_Profit / p.Gross_Profit.sum()
    p = p.sort_values("Gross_Profit", ascending=False).reset_index(drop=True)
    p["Cum_GP_Share"] = p.GP_Share.cumsum()
    p["ABC_by_GP"] = np.select([p.Cum_GP_Share.shift(fill_value=0) < 0.80, p.Cum_GP_Share.shift(fill_value=0) < 0.95], ["A", "B"], "C")
    return p


def discount_impact(sales: pd.DataFrame) -> pd.DataFrame:
    bins = [-0.001, 0.0, 0.10, 0.20, 0.30, 0.40]
    labels = ["0%", "1-10%", "11-20%", "21-30%", "31-40%"]
    s = sales.assign(Discount_Band=pd.cut(sales.Discount_Pct, bins, labels=labels))
    g = s.groupby(["Channel", "Discount_Band"], observed=True).agg(
        Invoices=("Invoice_ID", "count"), Units=("Quantity", "sum"), Net_Sales=("Net_Sales_EGP", "sum"),
        Gross_Profit=("Gross_Profit_EGP", "sum"), Loss_Making=("Is_Loss_Making", "sum"), Return_Rate=("Is_Returned", "mean"),
    ).reset_index()
    g["Gross_Margin"] = g.Gross_Profit / g.Net_Sales
    g["Units_per_Invoice"] = g.Units / g.Invoices
    return g


def discount_leakage(sales: pd.DataFrame) -> pd.DataFrame:
    """How much of the available margin (list price - COGS) is given away as discount."""
    g = sales.groupby("Channel").agg(Gross=("Gross_Sales_EGP", "sum"), Net=("Net_Sales_EGP", "sum"), COGS=("Estimated_COGS_EGP", "sum")).reset_index()
    g["Discount_Given"] = g.Gross - g.Net
    g["Margin_Before_Discount"] = g.Gross - g.COGS
    g["Share_of_Margin_Given_Away"] = g.Discount_Given / g.Margin_Before_Discount
    return g


def promo_test(sales: pd.DataFrame) -> pd.DataFrame:
    """Jun-Jul retail/online promo vs adjacent months: does the deeper discount buy volume?"""
    d = sales[sales.Channel != "Wholesale"].copy()
    d["Year"] = d.Order_Date.dt.year
    rows = []
    for year in sorted(d.Year.unique()):
        y = d[d.Year == year]
        promo, ref = y[y.Order_Date.dt.month.isin([6, 7])], y[y.Order_Date.dt.month.isin([5, 8])]
        if promo.empty or ref.empty:
            continue
        promo_days = promo.Order_Date.nunique() or 1
        ref_days = ref.Order_Date.nunique() or 1
        rows.append(
            {
                "Year": year,
                "Discount_Promo": 1 - promo.Net_Sales_EGP.sum() / promo.Gross_Sales_EGP.sum(),
                "Discount_Ref": 1 - ref.Net_Sales_EGP.sum() / ref.Gross_Sales_EGP.sum(),
                "Units_per_Day_Promo": promo.Quantity.sum() / promo_days,
                "Units_per_Day_Ref": ref.Quantity.sum() / ref_days,
                "GP_per_Unit_Promo": promo.Gross_Profit_EGP.sum() / promo.Quantity.sum(),
                "GP_per_Unit_Ref": ref.Gross_Profit_EGP.sum() / ref.Quantity.sum(),
            }
        )
    out = pd.DataFrame(rows)
    out["Volume_Lift"] = out.Units_per_Day_Promo / out.Units_per_Day_Ref - 1
    out["GP_per_Day_Change"] = (out.Units_per_Day_Promo * out.GP_per_Unit_Promo) / (out.Units_per_Day_Ref * out.GP_per_Unit_Ref) - 1
    return out


# ================================================ CASE 02 - CUSTOMER 360
def rfm_segments(c360: pd.DataFrame, customer_type: str = "Retail") -> pd.DataFrame:
    c = c360[c360.Customer_Type == customer_type].copy()
    c["R"] = pd.qcut(-c.Recency_Days, 5, labels=False, duplicates="drop") + 1
    c["F"] = pd.qcut(c.Orders.rank(method="first"), 5, labels=False) + 1
    c["M"] = pd.qcut(c.Net_Sales_EGP.rank(method="first"), 5, labels=False) + 1
    r, f = c.R, c.F
    c["Segment"] = np.select(
        [(r >= 4) & (f >= 4), (r >= 3) & (f >= 4), (r <= 2) & (f >= 4), (r >= 4) & (f <= 2), (r >= 3) & (f == 3), (r <= 2) & (f == 3), r == 3],
        ["Champions", "Loyal", "At Risk (high frequency)", "Recent / New", "Potential Loyalists", "At Risk", "Needs Attention"],
        default="Hibernating",
    )
    return c


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    g = rfm.groupby("Segment").agg(
        Customers=("Customer_ID", "count"), Net_Sales=("Net_Sales_EGP", "sum"), Gross_Profit=("Gross_Profit_EGP", "sum"),
        Avg_Orders=("Orders", "mean"), AOV=("AOV_EGP", "mean"), Median_Recency=("Recency_Days", "median"), Opted_In=("Consent_Status", lambda x: (x == "Opted-in").mean()),
    ).reset_index()
    g["Customer_Share"] = g.Customers / g.Customers.sum()
    g["Sales_Share"] = g.Net_Sales / g.Net_Sales.sum()
    return g.sort_values("Net_Sales", ascending=False)


def repeat_metrics(c360: pd.DataFrame, by: str) -> pd.DataFrame:
    g = c360.groupby(by).agg(
        Customers=("Customer_ID", "count"), Repeat_Rate=("Is_Repeat", "mean"), Avg_Orders=("Orders", "mean"),
        AOV=("AOV_EGP", "mean"), Net_per_Customer=("Net_Sales_EGP", "mean"), GP_per_Customer=("Gross_Profit_EGP", "mean"),
    ).reset_index()
    return g.sort_values("Net_per_Customer", ascending=False)


def cohort_retention(sales: pd.DataFrame, customer_type: str = "Retail", max_offset: int = 8) -> pd.DataFrame:
    s = sales[sales.Customer_Type == customer_type][["Customer_ID", "Order_Date"]].copy()
    s["Quarter"] = s.Order_Date.dt.to_period("Q")
    first = s.groupby("Customer_ID").Quarter.min().rename("Cohort")
    s = s.join(first, on="Customer_ID")
    s["Offset"] = (s.Quarter - s.Cohort).apply(lambda x: x.n)
    active = s.groupby(["Cohort", "Offset"]).Customer_ID.nunique().unstack(fill_value=0)
    size = active[0]
    retention = active.div(size, axis=0).iloc[:, : max_offset + 1]
    # mask offsets that have not happened yet for the latest cohorts
    last_q = s.Quarter.max()
    for cohort in retention.index:
        for offset in retention.columns:
            if cohort + int(offset) > last_q:
                retention.loc[cohort, offset] = np.nan
    retention.insert(0, "Cohort_Size", size)
    return retention


def churn_pool(c360: pd.DataFrame, churn_days: int = 180) -> pd.DataFrame:
    c = c360.copy()
    c["Lapsed"] = c.Recency_Days > churn_days
    g = c.groupby("Customer_Type").agg(
        Customers=("Customer_ID", "count"), Lapsed=("Lapsed", "sum"), Lapsed_Net_Value=("Net_Sales_EGP", lambda x: x[c.loc[x.index, "Lapsed"]].sum()),
        Repeat_Lapsed=("Lapsed", lambda x: (x & c.loc[x.index, "Is_Repeat"]).sum()),
    ).reset_index()
    g["Lapse_Rate"] = g.Lapsed / g.Customers
    return g


def interpurchase_gap(sales: pd.DataFrame) -> pd.Series:
    s = sales[sales.Customer_Type == "Retail"].sort_values(["Customer_ID", "Order_Date"])
    gaps = s.groupby("Customer_ID").Order_Date.diff().dt.days.dropna()
    return gaps.describe(percentiles=[0.5, 0.75, 0.9])


def high_value_profile(rfm: pd.DataFrame, dimension: str) -> pd.DataFrame:
    top = rfm[rfm.Net_Sales_EGP >= rfm.Net_Sales_EGP.quantile(0.9)]
    a = top[dimension].value_counts(normalize=True).rename("Top_Decile_Share")
    b = rfm[dimension].value_counts(normalize=True).rename("All_Retail_Share")
    out = pd.concat([a, b], axis=1).fillna(0)
    out["Index_vs_Base"] = out.Top_Decile_Share / out.All_Retail_Share * 100
    return out.sort_values("Index_vs_Base", ascending=False).reset_index(names=dimension)


# ================================================== CASE 03 - CHANNELS
def channel_scorecard(sales: pd.DataFrame) -> pd.DataFrame:
    k = kpi_by_period(sales, "Channel")
    k["Net_Share"] = k.Net_Sales / k.Net_Sales.sum()
    k["GP_Share"] = k.Gross_Profit / k.Gross_Profit.sum()
    k["Units_per_Invoice"] = k.Units / k.Invoices
    k["GP_per_Invoice"] = k.Gross_Profit / k.Invoices
    extra = sales.groupby("Channel").agg(Return_Rate_Lines=("Is_Returned", "mean"), Quality_Return_Share=("Is_Quality_Return", "mean")).reset_index()
    return k.merge(extra, on="Channel")


def channel_growth(sales: pd.DataFrame) -> pd.DataFrame:
    k = kpi_by_period(sales, ["Channel", "Fiscal_Year"]).sort_values(["Channel", "Fiscal_Year"])
    k["Net_YoY"] = k.groupby("Channel").Net_Sales.pct_change()
    k["GP_YoY"] = k.groupby("Channel").Gross_Profit.pct_change()
    return k


def geo_demand(sales: pd.DataFrame) -> pd.DataFrame:
    g = sales.groupby(["Customer_Governorate", "Channel"]).agg(Net_Sales=("Net_Sales_EGP", "sum"), Gross_Profit=("Gross_Profit_EGP", "sum"), Units=("Quantity", "sum")).reset_index()
    g["Gross_Margin"] = g.Gross_Profit / g.Net_Sales
    return g


def return_reasons(sales: pd.DataFrame) -> pd.DataFrame:
    r = sales[sales.Is_Returned].groupby(["Channel", "Return_Reason"]).agg(Lines=("Invoice_ID", "count"), Refund=("Refund_Value_EGP", "sum")).reset_index()
    r["Share_in_Channel"] = r.Lines / r.groupby("Channel").Lines.transform("sum")
    return r


# ========================================== CASE 04 - INVENTORY & DEMAND
def unit_costs(sales: pd.DataFrame) -> pd.DataFrame:
    return sales.groupby("Product_ID").agg(Unit_COGS=("Estimated_COGS_EGP", "sum"), Units=("Quantity", "sum")).assign(Unit_COGS=lambda d: d.Unit_COGS / d.Units)[["Unit_COGS"]].reset_index()


def inventory_diagnosis(inv: pd.DataFrame, sales: pd.DataFrame, review_days: int = 30, cover_buffer_days: int = 60) -> pd.DataFrame:
    """One row per product x location with ABC/XYZ, turns, cover and a latest-month status."""
    key = ["Location_ID", "Location_Name", "Product_ID", "Product_Name"]
    last_month = inv.Month.max()
    last12 = inv[inv.Month > last_month - pd.DateOffset(months=12)]
    d = last12.groupby(key).agg(
        Units_Sold_12M=("Sold_Qty", "sum"), Avg_Stock=("Closing_Stock", "mean"), Stockout_Days_12M=("Stockout_Days_In_Month", "sum"),
        Lost_Units_12M=("Lost_Units_Est", "sum"),
    ).reset_index()
    d["Stock_Turns"] = safe_div(d.Units_Sold_12M, d.Avg_Stock)
    cv = inv.groupby(key).Sold_Qty.agg(lambda x: x.std() / x.mean() if x.mean() else np.nan).rename("Demand_CV").reset_index()
    d = d.merge(cv, on=key)
    d["XYZ"] = np.select([d.Demand_CV <= 0.5, d.Demand_CV <= 1.0], ["X", "Y"], "Z")
    last = inv[inv.Month == last_month][key + ["Closing_Stock", "Avg_Daily_Sales_90D", "Safety_Stock_Units", "Reorder_Point_Units", "Supplier_Lead_Time_Days", "Days_of_Cover", "Stockout_Days_In_Month"]]
    d = d.merge(last, on=key)
    d = d.merge(unit_costs(sales), on="Product_ID", how="left")
    value = (sales[sales.Order_Date > last_month - pd.DateOffset(months=11) - pd.offsets.MonthBegin(0)].groupby(["Location_ID", "Product_ID"]).Net_Sales_EGP.sum().rename("Net_Sales_12M").reset_index())
    d = d.merge(value, on=["Location_ID", "Product_ID"], how="left").fillna({"Net_Sales_12M": 0})
    d = d.sort_values("Net_Sales_12M", ascending=False)
    cum = d.Net_Sales_12M.cumsum() / d.Net_Sales_12M.sum()
    d["ABC"] = np.select([cum.shift(fill_value=0) < 0.80, cum.shift(fill_value=0) < 0.95], ["A", "B"], "C")
    d["Policy_Max_Stock"] = d.Avg_Daily_Sales_90D * (d.Supplier_Lead_Time_Days + review_days + cover_buffer_days) + d.Safety_Stock_Units
    d["Excess_Units"] = (d.Closing_Stock - d.Policy_Max_Stock).clip(lower=0)
    d["Excess_Value_EGP"] = d.Excess_Units * d.Unit_COGS
    d["Stock_Value_EGP"] = d.Closing_Stock * d.Unit_COGS
    d["Status"] = np.select(
        [d.Stockout_Days_In_Month > 0, d.Closing_Stock < d.Reorder_Point_Units, d.Closing_Stock > d.Policy_Max_Stock],
        ["Stockout", "Below reorder point", "Overstock"], "Healthy",
    )
    return d.reset_index(drop=True)


def inventory_by_location(diag: pd.DataFrame) -> pd.DataFrame:
    g = diag.groupby("Location_Name").agg(
        SKUs=("Product_ID", "count"), Stock_Value=("Stock_Value_EGP", "sum"), Excess_Value=("Excess_Value_EGP", "sum"),
        Median_Cover_Days=("Days_of_Cover", "median"), Median_Turns=("Stock_Turns", "median"), Stockout_Days_12M=("Stockout_Days_12M", "sum"),
        Lost_Units_12M=("Lost_Units_12M", "sum"), Overstock_SKUs=("Status", lambda x: (x == "Overstock").sum()),
        Below_ROP_or_Stockout=("Status", lambda x: x.isin(["Stockout", "Below reorder point"]).sum()),
    ).reset_index()
    g["Excess_Share"] = g.Excess_Value / g.Stock_Value
    return g


def stockout_by_month(inv: pd.DataFrame) -> pd.DataFrame:
    g = inv.groupby(["Month", "Location_Name"]).agg(Stockout_Days=("Stockout_Days_In_Month", "sum"), Stockout_SKUs=("Stockout_Flag", "sum")).reset_index()
    return g


def stockout_margin_loss(inv: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    gp_unit = sales.groupby(["Location_ID", "Product_ID"]).apply(lambda d: d.Gross_Profit_EGP.sum() / d.Quantity.sum(), include_groups=False).rename("GP_per_Unit").reset_index()
    x = inv.groupby(["Location_ID", "Location_Name", "Product_ID"]).Lost_Units_Est.sum().reset_index().merge(gp_unit, on=["Location_ID", "Product_ID"], how="left")
    x["Lost_GP_EGP"] = x.Lost_Units_Est * x.GP_per_Unit
    return x.groupby("Location_Name").agg(Lost_Units=("Lost_Units_Est", "sum"), Lost_GP_EGP=("Lost_GP_EGP", "sum")).reset_index()


# ============================================= CASE 05 - MARKETING
def marketing_by(m: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    g = m.groupby(by).agg(
        Campaigns=("Campaign_ID", "count"), Spend=("Spend_EGP", "sum"), Impressions=("Impressions", "sum"), Clicks=("Clicks", "sum"),
        Orders=("Orders_Attributed", "sum"), Revenue=("Attributed_Revenue_EGP", "sum"), Avg_Discount=("Discount_Pct", "mean"),
    ).reset_index()
    g["ROAS"] = g.Revenue / g.Spend
    g["CAC"] = g.Spend / g.Orders
    g["CTR"] = g.Clicks / g.Impressions
    g["CVR"] = g.Orders / g.Clicks
    g["CPC"] = g.Spend / g.Clicks
    g["AOV"] = g.Revenue / g.Orders
    g["Spend_Share"] = g.Spend / g.Spend.sum()
    g["Revenue_Share"] = g.Revenue / g.Revenue.sum()
    return g.sort_values("ROAS", ascending=False)


def breakeven_roas(sales: pd.DataFrame) -> float:
    """ROAS needed for a campaign to cover its own spend out of gross profit (retail+online margin)."""
    d = sales[sales.Channel != "Wholesale"]
    return d.Net_Sales_EGP.sum() / d.Gross_Profit_EGP.sum()


def attribution_reconciliation(m: pd.DataFrame, sales: pd.DataFrame) -> dict:
    attributed = m.Attributed_Revenue_EGP.sum()
    online = sales.loc[sales.Channel == "Online", "Net_Sales_EGP"].sum()
    non_wholesale = sales.loc[sales.Channel != "Wholesale", "Net_Sales_EGP"].sum()
    return {
        "attributed_revenue": attributed, "online_net_sales": online, "non_wholesale_net_sales": non_wholesale,
        "overclaim_vs_online": attributed / online, "haircut_factor_online": online / attributed,
        "haircut_factor_non_wholesale": non_wholesale / attributed,
        "attributed_orders": int(m.Orders_Attributed.sum()), "online_invoices": int((sales.Channel == "Online").sum()),
    }


def campaign_outliers(m: pd.DataFrame) -> pd.DataFrame:
    x = m.copy()
    x["log_roas"] = np.log(x.ROAS)
    x["ROAS_z_in_platform"] = x.groupby("Platform").log_roas.transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    x["Outlier"] = np.select([x.ROAS_z_in_platform >= 2, x.ROAS_z_in_platform <= -2], ["Star", "Underperformer"], "Normal")
    return x.drop(columns="log_roas")


def spend_response(m: pd.DataFrame) -> pd.DataFrame:
    """Log-log slope of attributed revenue on spend per platform (slope < 1 = diminishing returns)."""
    rows = []
    for platform, d in m.groupby("Platform"):
        res = stats.linregress(np.log(d.Spend_EGP), np.log(d.Attributed_Revenue_EGP))
        rows.append({"Platform": platform, "Campaigns": len(d), "Spend_Elasticity": res.slope, "R2": res.rvalue**2, "p_value": res.pvalue})
    return pd.DataFrame(rows)


def reallocation_scenario(
    platform_table: pd.DataFrame, marginal_efficiency: float = 0.5, haircut: float = 0.345, gross_margin: float = 0.359,
    donor_shares: dict[str, float] | None = None, receiver: str = "Google", receiver_growth_cap: float = 0.30,
) -> dict:
    """Budget-neutral shift of spend from weak platforms into Google.

    Donors lose revenue at their *average* ROAS (conservative); the receiver earns only
    `marginal_efficiency` x its average ROAS (diminishing returns). Revenue is scaled by `haircut`
    to reconcile platform-reported revenue with actual sales.
    """
    donor_shares = donor_shares or {"Influencer": 0.50, "Meta": 0.15, "TikTok": 0.15}
    t = platform_table.set_index("Platform")
    pool = sum(t.loc[p, "Spend"] * s for p, s in donor_shares.items())
    capacity = t.loc[receiver, "Spend"] * receiver_growth_cap
    shift = min(pool, capacity)
    scale = shift / pool
    lost = sum(t.loc[p, "Spend"] * s * scale * t.loc[p, "ROAS"] for p, s in donor_shares.items())
    gained = shift * t.loc[receiver, "ROAS"] * marginal_efficiency
    net_revenue = (gained - lost) * haircut
    return {"budget_shifted": shift, "revenue_lost_reported": lost, "revenue_gained_reported": gained, "net_revenue_reconciled": net_revenue, "net_gross_profit": net_revenue * gross_margin}


def web_funnel(wt: pd.DataFrame) -> pd.DataFrame:
    w = wt.copy()
    w["Month"] = w.Date.dt.to_period("M").dt.to_timestamp()
    g = w.groupby("Month").agg(
        Sessions=("Sessions", "sum"), Orders=("Orders", "sum"), Add_To_Cart=("Add_To_Cart_Rate", "mean"), Checkout=("Checkout_Rate", "mean"),
        Mobile_Pct=("Mobile_Pct", "mean"), Load_Sec=("Avg_Page_Load_Sec", "mean"),
    ).reset_index()
    g["Conversion"] = g.Orders / g.Sessions
    return g


def load_time_effect(wt: pd.DataFrame) -> pd.DataFrame:
    w = wt.copy()
    w["Load_Band"] = pd.cut(w.Avg_Page_Load_Sec, [0, 2, 2.5, 3, 3.5, 10], labels=["<2s", "2-2.5s", "2.5-3s", "3-3.5s", ">3.5s"])
    g = w.groupby("Load_Band", observed=True).agg(Days=("Date", "count"), Sessions=("Sessions", "sum"), Orders=("Orders", "sum")).reset_index()
    g["Conversion"] = g.Orders / g.Sessions
    return g


# ===================================== CASE 06 - PRODUCTION
def line_scorecard(p: pd.DataFrame) -> pd.DataFrame:
    g = p.groupby("Line_ID").agg(
        Orders=("Production_Order_ID", "count"), Planned=("Planned_Qty", "sum"), Produced=("Produced_Qty", "sum"), Good=("Good_Qty", "sum"),
        Rejected=("Rejected_Qty", "sum"), Efficiency=("Efficiency_Pct", "mean"), Avg_Delay_Days=("Delay_Days", "mean"),
        Delayed_Share=("Is_Delayed", "mean"), Overtime_Hours=("Overtime_Hours", "sum"), Avg_Throughput_Day=("Throughput_Units_Per_Day", "mean"),
        Avg_Cycle_Days=("Cycle_Days", "mean"),
    ).reset_index()
    g["Plan_Attainment"] = g.Produced / g.Planned
    g["Reject_Rate"] = g.Rejected / g.Produced
    g["Overtime_per_Order"] = g.Overtime_Hours / g.Orders
    return g


def line_delay_significance(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line, d in p.groupby("Line_ID"):
        other = p[p.Line_ID != line]
        table = [[d.Is_Delayed.sum(), (~d.Is_Delayed).sum()], [other.Is_Delayed.sum(), (~other.Is_Delayed).sum()]]
        chi2, pval, *_ = stats.chi2_contingency(table)
        rows.append({"Line_ID": line, "Delayed_Share": d.Is_Delayed.mean(), "Rest_Delayed_Share": other.Is_Delayed.mean(), "p_value": pval})
    return pd.DataFrame(rows)


def bottleneck_pareto(p: pd.DataFrame) -> pd.DataFrame:
    d = p[p.Is_Delayed].groupby("Primary_Bottleneck").agg(
        Delayed_Orders=("Production_Order_ID", "count"), Delay_Days=("Delay_Days", "sum"), Avg_Delay=("Delay_Days", "mean"),
        Avg_Efficiency=("Efficiency_Pct", "mean"), Avg_Overtime=("Overtime_Hours", "mean"),
    ).reset_index().sort_values("Delay_Days", ascending=False)
    d["Delay_Share"] = d.Delay_Days / d.Delay_Days.sum()
    d["Cum_Share"] = d.Delay_Share.cumsum()
    return d


def line_bottleneck_matrix(p: pd.DataFrame) -> pd.DataFrame:
    return p[p.Is_Delayed].pivot_table(index="Line_ID", columns="Primary_Bottleneck", values="Delay_Days", aggfunc="sum", fill_value=0)


def production_monthly(p: pd.DataFrame, go: pd.DataFrame) -> pd.DataFrame:
    m = p.groupby("Month").agg(
        Produced=("Produced_Qty", "sum"), Planned=("Planned_Qty", "sum"), Efficiency=("Efficiency_Pct", "mean"), Delay=("Delay_Days", "mean"),
        Delayed_Share=("Is_Delayed", "mean"), Overtime=("Overtime_Hours", "sum"),
    ).reset_index()
    hub = go[go.Business_Unit == "LEO PRODUCTION HUB"][["Month", "Volume", "Operational_Efficiency_Pct"]].rename(columns={"Volume": "Hub_Reported_Garments", "Operational_Efficiency_Pct": "Hub_Reported_Efficiency"})
    m = m.merge(hub, on="Month", how="left")
    capacity = FACTORY_DAILY_CAPACITY * WORKING_DAYS_PER_MONTH
    m["Capacity_Util_Reported"] = m.Hub_Reported_Garments / capacity
    m["Capacity_Util_from_Orders"] = m.Produced / capacity
    return m


def overtime_effect(p: pd.DataFrame) -> pd.DataFrame:
    d = p.copy()
    d["OT_Band"] = pd.cut(d.Overtime_Hours, [-0.1, 2, 6, 10, 15, 100], labels=["0-2h", "2-6h", "6-10h", "10-15h", ">15h"])
    return d.groupby("OT_Band", observed=True).agg(Orders=("Production_Order_ID", "count"), Delayed_Share=("Is_Delayed", "mean"), Efficiency=("Efficiency_Pct", "mean"), Avg_Delay=("Delay_Days", "mean")).reset_index()


# =========================================== CASE 07 - QUALITY
def defect_pareto(q: pd.DataFrame) -> pd.DataFrame:
    d = q[q.Rejected_Qty > 0].groupby("Defect_Type").agg(Rejected=("Rejected_Qty", "sum"), Loss_EGP=("Estimated_Loss_EGP", "sum"), Inspections=("QC_ID", "count")).reset_index().sort_values("Loss_EGP", ascending=False)
    d["Loss_Share"] = d.Loss_EGP / d.Loss_EGP.sum()
    d["Cum_Share"] = d.Loss_Share.cumsum()
    return d


def stage_defect_loss(q: pd.DataFrame) -> pd.DataFrame:
    return q[q.Rejected_Qty > 0].pivot_table(index="Stage", columns="Defect_Type", values="Estimated_Loss_EGP", aggfunc="sum", fill_value=0)


def stage_summary(q: pd.DataFrame) -> pd.DataFrame:
    g = q.groupby("Stage").agg(Inspected=("Inspected_Qty", "sum"), Rejected=("Rejected_Qty", "sum"), Loss=("Estimated_Loss_EGP", "sum")).reset_index()
    g["Reject_Rate"] = g.Rejected / g.Inspected
    return g.sort_values("Loss", ascending=False)


def root_cause_summary(q: pd.DataFrame) -> pd.DataFrame:
    g = q.groupby("Root_Cause").agg(Inspected=("Inspected_Qty", "sum"), Rejected=("Rejected_Qty", "sum"), Loss=("Estimated_Loss_EGP", "sum")).reset_index()
    g["Reject_Rate"] = g.Rejected / g.Inspected
    g["Loss_Share"] = g.Loss / g.Loss.sum()
    return g.sort_values("Loss", ascending=False)


QUALITY_RETURN_MAP = {
    "Fabric / stitching": ["Fabric Defect", "Stitch Defect", "Finishing Defect"],
    "Print issue": ["Print Cracking", "Ink Smudge", "Color Misalignment"],
}


def quality_return_link(q: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """Per product: QC reject rate of the mapped defect group vs customer return rate for the matching reason."""
    rows = []
    for reason, defects in QUALITY_RETURN_MAP.items():
        qc = q[q.Defect_Type.isin(defects)].groupby("Product_ID").Rejected_Qty.sum()
        insp = q.groupby("Product_ID").Inspected_Qty.sum()
        sold = sales.groupby("Product_ID").Invoice_ID.count()
        ret = sales[sales.Return_Reason == reason].groupby("Product_ID").Invoice_ID.count()
        d = pd.DataFrame({"QC_Reject_Rate": qc / insp, "Return_Rate": ret / sold}).dropna()
        r, p_val = stats.pearsonr(d.QC_Reject_Rate, d.Return_Rate)
        rows.append({"Return_Reason": reason, "Matched_Defects": ", ".join(defects), "Products": len(d), "Pearson_r": r, "p_value": p_val})
    return pd.DataFrame(rows)


def cost_of_poor_quality(q: pd.DataFrame, printing: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    qc_loss = q.Estimated_Loss_EGP.sum()
    print_rework = printing.Rework_Cost_EGP.sum()
    print_waste = (printing.Reject_Rate * printing.Ink_or_Consumable_Cost_EGP).sum()
    quality_refunds = sales.loc[sales.Is_Quality_Return, "Refund_Value_EGP"].sum()
    rows = [
        ("Internal failure", "QC rejects (inspection loss)", qc_loss),
        ("Internal failure", "Print rework cost", print_rework),
        ("Internal failure", "Consumables wasted on rejected prints", print_waste),
        ("External failure", "Refunds on quality-related returns", quality_refunds),
    ]
    t = pd.DataFrame(rows, columns=["Type", "Component", "EGP"])
    t["Share"] = t.EGP / t.EGP.sum()
    t["Pct_of_Gross_Profit"] = t.EGP / sales.Gross_Profit_EGP.sum()
    return t


def product_quality_ranking(pqr: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    d = pqr.copy()
    d["Priority_Score"] = d.QC_Reject_Rate.rank(pct=True) * 0.5 + d.QC_Loss_EGP.rank(pct=True) * 0.5
    return d.sort_values("Priority_Score", ascending=False).head(top)[["Product_ID", "Product_Name", "Product_Family", "QC_Reject_Rate", "QC_Loss_EGP", "Quality_Return_Rate", "Priority_Score"]]


# ====================================== CASE 08 - PRINTING & DOWNTIME
def machine_scorecard(pb: pd.DataFrame, dt: pd.DataFrame) -> pd.DataFrame:
    g = pb.groupby("Machine_ID").agg(
        Batches=("Print_Batch_ID", "count"), Printed=("Printed_Qty", "sum"), Good=("Good_Qty", "sum"), Rejected=("Rejected_Qty", "sum"),
        Reprint=("Reprint_Qty", "sum"), Consumables_EGP=("Ink_or_Consumable_Cost_EGP", "sum"), Rework_EGP=("Rework_Cost_EGP", "sum"),
        Overrun_Share=("Overrun_Flag", "mean"),
    ).reset_index()
    g["Yield"] = g.Good / g.Printed
    g["Reject_Rate"] = g.Rejected / g.Printed
    g["Reprint_Rate"] = g.Reprint / g.Printed
    g["Cost_per_Good_Unit"] = (g.Consumables_EGP + g.Rework_EGP) / g.Good
    d = dt[dt.Business_Area == "Printing"].groupby("Machine_ID").agg(
        Downtime_Events=("Downtime_ID", "count"), Downtime_Hours=("Downtime_Hours", "sum"), Unplanned_Hours=("Downtime_Hours", lambda x: x[dt.loc[x.index, "Planned"] == "No"].sum()),
    ).reset_index()
    d["Hours_per_Event"] = d.Downtime_Hours / d.Downtime_Events
    d["Unplanned_Share"] = d.Unplanned_Hours / d.Downtime_Hours
    out = g.merge(d, on="Machine_ID")
    z = lambda s: (s - s.mean()) / s.std(ddof=0)
    out["Priority_Score"] = z(out.Downtime_Hours) + z(out.Reject_Rate) + z(out.Cost_per_Good_Unit)
    return out.sort_values("Priority_Score", ascending=False)


def machine_reject_significance(pb: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for machine, d in pb.groupby("Machine_ID"):
        other = pb[pb.Machine_ID != machine]
        table = [[d.Rejected_Qty.sum(), d.Good_Qty.sum()], [other.Rejected_Qty.sum(), other.Good_Qty.sum()]]
        _, p_val, *_ = stats.chi2_contingency(table)
        rows.append({"Machine_ID": machine, "Reject_Rate": d.Rejected_Qty.sum() / d.Printed_Qty.sum(), "Rest_Reject_Rate": other.Rejected_Qty.sum() / other.Printed_Qty.sum(), "p_value": p_val})
    return pd.DataFrame(rows)


def downtime_reasons(dt: pd.DataFrame, area: str | None = None) -> pd.DataFrame:
    d = dt if area is None else dt[dt.Business_Area == area]
    g = d.groupby(["Reason", "Planned"]).agg(Events=("Downtime_ID", "count"), Hours=("Downtime_Hours", "sum")).reset_index().sort_values("Hours", ascending=False)
    g["Share"] = g.Hours / g.Hours.sum()
    return g


def downtime_by_machine_reason(dt: pd.DataFrame) -> pd.DataFrame:
    return dt[dt.Planned == "No"].pivot_table(index="Machine_ID", columns="Reason", values="Downtime_Hours", aggfunc="sum", fill_value=0)


def printing_defects_by_machine(pb: pd.DataFrame) -> pd.DataFrame:
    return pb[pb.Rejected_Qty > 0].pivot_table(index="Machine_ID", columns="Defect_Type", values="Rejected_Qty", aggfunc="sum", fill_value=0)


def machine_month_link(pb: pd.DataFrame, dt: pd.DataFrame) -> dict:
    """Does a machine-month with more downtime also show a worse yield?"""
    y = pb.groupby(["Machine_ID", "Month"]).apply(lambda d: d.Good_Qty.sum() / d.Printed_Qty.sum(), include_groups=False).rename("Yield")
    h = dt[dt.Business_Area == "Printing"].groupby(["Machine_ID", "Month"]).Downtime_Hours.sum().rename("Downtime_Hours")
    j = pd.concat([y, h], axis=1).dropna()
    r, p_val = stats.pearsonr(j.Yield, j.Downtime_Hours)
    return {"machine_months": len(j), "pearson_r": r, "p_value": p_val}


# =============================== CASE 09 - SUPPLY CHAIN & MATERIALS
def supplier_scorecard(po: pd.DataFrame) -> pd.DataFrame:
    g = po.groupby("Supplier_ID").agg(
        POs=("PO_ID", "count"), Spend=("Spend_EGP", "sum"), OTD=("On_Time", "mean"), OTIF=("OTIF", "mean"), Avg_Days_Late=("Days_Late", "mean"),
        Lead_Time=("Lead_Time_Days", "mean"), Promised_Lead_Time=("Promised_Lead_Time_Days", "mean"), Reject_Rate=("Reject_Rate", "mean"),
        Category=("Primary_Category", "first"), Contracted=("Contracted", "first"), Payment_Terms=("Payment_Terms_Days", "first"),
    ).reset_index()
    g["Spend_Share"] = g.Spend / g.Spend.sum()
    return g.sort_values("OTD")


def supplier_otd_trend(po: pd.DataFrame, freq: str = "Q") -> pd.DataFrame:
    p = po.assign(Period=po.Order_Date.dt.to_period(freq).dt.to_timestamp())
    return p.pivot_table(index="Period", columns="Supplier_ID", values="On_Time", aggfunc="mean")


def supplier_action_matrix(sc: pd.DataFrame, otd_threshold: float = 0.60) -> pd.DataFrame:
    d = sc.copy()
    median_spend = d.Spend.median()
    d["Reliability"] = np.where(d.OTD >= otd_threshold, "Reliable", "Unreliable")
    d["Spend_Tier"] = np.where(d.Spend >= median_spend, "High spend", "Lower spend")
    d["Action"] = np.select(
        [(d.OTD < otd_threshold) & (d.Contracted == "Yes"), (d.OTD < otd_threshold), d.OTD < 0.70, d.Contracted == "No"],
        ["Escalate: enforce contract SLA / penalties, qualify backup", "Replace or dual-source; put under contract", "Monitor: tighten promised dates", "Maintain; formalise contract"],
        "Maintain / strategic partner",
    )
    return d[["Supplier_ID", "Category", "Contracted", "OTD", "Avg_Days_Late", "Spend", "Spend_Share", "Reliability", "Spend_Tier", "Action"]]


def material_scorecard(po: pd.DataFrame, bom: pd.DataFrame, prod: pd.DataFrame) -> pd.DataFrame:
    g = po.groupby(["Material_ID", "Material_Name", "Material_Category"]).agg(
        POs=("PO_ID", "count"), Spend=("Spend_EGP", "sum"), Accepted=("Accepted_Qty", "sum"), OTD=("On_Time", "mean"),
        Lead_Time=("Lead_Time_Days", "mean"), Standard_Lead_Time=("Standard_Lead_Time_Days", "first"), Reject_Rate=("Reject_Rate", "mean"),
        Suppliers=("Supplier_ID", "nunique"),
    ).reset_index()
    fy = po.groupby(["Material_ID", "Fiscal_Year"]).Unit_Cost_EGP.mean().unstack()
    price_change = ((fy.iloc[:, -1] / fy.iloc[:, 0]) - 1).rename("Price_Change_FY24_to_FY26")
    g = g.merge(price_change.reset_index(), on="Material_ID")
    g["Lead_Time_Gap"] = g.Lead_Time - g.Standard_Lead_Time
    products_using = bom.groupby("Material_ID").Product_ID.nunique().rename("Products_Using")
    produced = prod.groupby("Product_ID").Produced_Qty.sum()
    required = bom.assign(Req=bom.Product_ID.map(produced) * bom.Required_Per_Unit).groupby("Material_ID").Req.sum().rename("BOM_Requirement")
    g = g.merge(products_using, on="Material_ID", how="left").merge(required, on="Material_ID", how="left")
    g["Products_Using"] = g.Products_Using.fillna(0)
    g["Coverage_vs_BOM"] = g.Accepted / g.BOM_Requirement
    g["Criticality"] = (g.Products_Using.rank(pct=True) * 0.4 + g.Standard_Lead_Time.rank(pct=True) * 0.3 + g.Spend.rank(pct=True) * 0.3)
    g["Shortage_Risk"] = g.Criticality * (1 - g.OTD) * (g.Lead_Time / g.Lead_Time.mean())
    return g.sort_values("Shortage_Risk", ascending=False)


def procurement_trend(po: pd.DataFrame) -> pd.DataFrame:
    g = po.groupby(po.Order_Date.dt.to_period("Q").dt.to_timestamp()).agg(OTD=("On_Time", "mean"), OTIF=("OTIF", "mean"), Days_Late=("Days_Late", "mean"), Spend=("Spend_EGP", "sum")).reset_index().rename(columns={"Order_Date": "Quarter"})
    return g


def otd_ex_problem_suppliers(po: pd.DataFrame, problem_ids: list[str]) -> pd.DataFrame:
    po = po.assign(Year=po.Order_Date.dt.year, Group=np.where(po.Supplier_ID.isin(problem_ids), "Problem suppliers", "Other suppliers"))
    return po.pivot_table(index="Year", columns="Group", values="On_Time", aggfunc="mean")


# ===================================== CASE 10 - INTEGRATED STORY
def integrated_monthly(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    s, inv, po, p, q, m, dt = d["sales"], d["inventory"], d["procurement"], d["production"], d["quality"], d["marketing"], d["downtime"]
    wt = web_funnel(d["web_traffic"]).set_index("Month")
    out = pd.DataFrame(
        {
            "Net_Sales": s.groupby("Month").Net_Sales_EGP.sum(), "Gross_Profit": s.groupby("Month").Gross_Profit_EGP.sum(), "Units": s.groupby("Month").Quantity.sum(),
            "Refund_Rate": s.groupby("Month").apply(lambda x: x.Refund_Value_EGP.sum() / x.Net_Sales_EGP.sum(), include_groups=False),
            "Marketing_Spend": m.groupby("Month").Spend_EGP.sum(), "Web_Conversion": wt.Conversion,
            "Stockout_Days": inv.groupby("Month").Stockout_Days_In_Month.sum(), "Supplier_OTD": po.groupby("Month").On_Time.mean(),
            "Material_Wait_Hours": dt[dt.Reason == "Material wait"].groupby("Month").Downtime_Hours.sum(),
            "Production_Delay_Days": p.groupby("Month").Delay_Days.mean(), "Production_Efficiency": p.groupby("Month").Efficiency_Pct.mean(),
            "QC_Reject_Rate": q.groupby("Month").apply(lambda x: x.Rejected_Qty.sum() / x.Inspected_Qty.sum(), include_groups=False),
        }
    )
    return out.reset_index().rename(columns={"index": "Month"})


def chain_correlations(chain: pd.DataFrame) -> pd.DataFrame:
    """Month-over-month changes (not levels) so shared trends/seasonality do not create fake links."""
    diff = chain.set_index("Month").diff().dropna()
    return diff.corr()


def lagged_link(chain: pd.DataFrame, x: str, y: str, max_lag: int = 3) -> pd.DataFrame:
    c = chain.set_index("Month")
    rows = []
    for lag in range(0, max_lag + 1):
        pair = pd.concat([c[x].shift(lag), c[y]], axis=1).dropna()
        r, p_val = stats.pearsonr(pair.iloc[:, 0].diff().dropna(), pair.iloc[:, 1].diff().dropna().loc[pair.iloc[:, 0].diff().dropna().index])
        rows.append({"x": x, "y": y, "lag_months": lag, "pearson_r_on_changes": r, "p_value": p_val})
    return pd.DataFrame(rows)
