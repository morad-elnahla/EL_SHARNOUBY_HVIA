"""Models and decision logic.

1. Demand forecast (seasonal-index model, backtested) + replenishment / rebalancing rules  -> Case 04
2. Wholesale discount-governance engine (diagnosis, price-response test, policy simulation,
   deal-desk quote check)                                                                  -> New opportunity
3. FY27 value-at-stake scenarios                                                           -> Case 10
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score

from utils import MODELS, safe_div

LOCATION_CHANNEL = {"LOC001": "Retail", "LOC002": "Online", "LOC003": "Wholesale"}
Z_95 = 1.645

# ===================================================== 1. DEMAND FORECAST


def _centered_ma(series: pd.Series) -> pd.Series:
    return series.rolling(12).mean().rolling(2).mean().shift(-6)


def seasonal_indices(hist: pd.DataFrame) -> pd.DataFrame:
    """Ratio-to-centered-moving-average seasonal index per location x season group x calendar month."""
    grouped = hist.groupby(["Location_ID", "Season_Group", "Month"]).Units.sum().reset_index()
    rows = []
    for (loc, fam), d in grouped.groupby(["Location_ID", "Season_Group"]):
        s = d.set_index("Month").Units.asfreq("MS", fill_value=0)
        ratio = (s / _centered_ma(s)).replace([np.inf, -np.inf], np.nan).dropna()
        by_month = ratio.groupby(ratio.index.month).mean().reindex(range(1, 13)).fillna(1.0)
        by_month = by_month / by_month.mean()
        rows += [(loc, fam, m, v) for m, v in by_month.items()]
    return pd.DataFrame(rows, columns=["Location_ID", "Season_Group", "Calendar_Month", "SI"])


def _group_growth(hist: pd.DataFrame) -> pd.DataFrame:
    last = hist.Month.max()
    recent = hist[hist.Month > last - pd.DateOffset(months=12)]
    prior = hist[(hist.Month <= last - pd.DateOffset(months=12)) & (hist.Month > last - pd.DateOffset(months=24))]
    a = recent.groupby(["Location_ID", "Season_Group"]).Units.sum()
    b = prior.groupby(["Location_ID", "Season_Group"]).Units.sum()
    g = (a / b - 1).replace([np.inf, -np.inf], np.nan) if len(prior) else pd.Series(0.0, index=a.index)
    return g.rename("Growth").clip(-0.3, 0.5).fillna(0).reset_index()


def forecast_from_origin(hist: pd.DataFrame, horizon: int, method: str = "si_growth", level_months: int = 6, growth_damping: float = 0.5) -> pd.DataFrame:
    """Forecast units per location x product for `horizon` months after the last month in `hist`.

    Methods: naive3 (mean of last 3 months), seasonal_naive (same month last year),
    si (deseasonalised level x seasonal index), si_growth (si plus damped season group growth).
    """
    origin = hist.Month.max()
    future = pd.date_range(origin + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    keys = hist[["Location_ID", "Product_ID", "Season_Group"]].drop_duplicates()
    series = hist.pivot_table(index="Month", columns=["Location_ID", "Product_ID"], values="Units", aggfunc="sum", fill_value=0)

    if method == "naive3":
        base = series.tail(3).mean()
        out = [pd.DataFrame({"Month": m, "Units_Forecast": base}).reset_index() for m in future]
        return pd.concat(out, ignore_index=True)
    if method == "seasonal_naive":
        frames = []
        for m in future:
            ly = m - pd.DateOffset(years=1)
            vals = series.loc[ly] if ly in series.index else series.tail(3).mean()
            frames.append(pd.DataFrame({"Month": m, "Units_Forecast": vals}).reset_index())
        return pd.concat(frames, ignore_index=True)

    si = seasonal_indices(hist)
    si_lookup = si.merge(keys[["Season_Group"]].drop_duplicates(), on="Season_Group")
    growth = _group_growth(hist) if method == "si_growth" else None
    tail = series.tail(level_months)
    tail_long = tail.stack(["Location_ID", "Product_ID"]).rename("Units").reset_index().merge(keys, on=["Location_ID", "Product_ID"])
    tail_long["Calendar_Month"] = tail_long.Month.dt.month
    tail_long = tail_long.merge(si_lookup, on=["Location_ID", "Season_Group", "Calendar_Month"])
    tail_long["Deseason"] = tail_long.Units / tail_long.SI.clip(lower=0.05)
    level = tail_long.groupby(["Location_ID", "Product_ID", "Season_Group"]).Deseason.mean().reset_index()
    frames = []
    for step, m in enumerate(future, start=1):
        f = level.copy()
        f["Month"] = m
        f["Calendar_Month"] = m.month
        f = f.merge(si_lookup, on=["Location_ID", "Season_Group", "Calendar_Month"])
        f["Units_Forecast"] = f.Deseason * f.SI
        if growth is not None:
            f = f.merge(growth, on=["Location_ID", "Season_Group"], how="left")
            f["Units_Forecast"] *= 1 + growth_damping * f.Growth.fillna(0) * step / 12
        frames.append(f[["Month", "Location_ID", "Product_ID", "Units_Forecast"]])
    return pd.concat(frames, ignore_index=True)


def build_history(inventory: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    h = inventory[["Month", "Location_ID", "Product_ID", "Sold_Qty"]].rename(columns={"Sold_Qty": "Units"})
    season = products[["Product_ID", "Catalog_Season"]].assign(
        Season_Group=lambda d: np.where(d.Catalog_Season.str.contains("Autumn|Winter"), "Autumn/Winter", "Summer"))
    return h.merge(season[["Product_ID", "Season_Group"]], on="Product_ID", how="left", validate="many_to_one")


def backtest(hist: pd.DataFrame, origins: list[str], horizon: int = 3, methods: tuple[str, ...] = ("naive3", "seasonal_naive", "si", "si_growth")) -> pd.DataFrame:
    rows = []
    for origin in origins:
        train = hist[hist.Month <= pd.Timestamp(origin)]
        test = hist[(hist.Month > pd.Timestamp(origin)) & (hist.Month <= pd.Timestamp(origin) + pd.DateOffset(months=horizon))]
        for method in methods:
            fc = forecast_from_origin(train, horizon, method)
            j = test.merge(fc, on=["Month", "Location_ID", "Product_ID"], how="inner")
            j["Horizon"] = ((j.Month.dt.year - pd.Timestamp(origin).year) * 12 + j.Month.dt.month - pd.Timestamp(origin).month)
            j["Method"], j["Origin"] = method, origin
            rows.append(j)
    return pd.concat(rows, ignore_index=True)


def backtest_metrics(bt: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    def metrics(d: pd.DataFrame) -> pd.Series:
        return pd.Series({"WAPE": (d.Units - d.Units_Forecast).abs().sum() / d.Units.sum(), "Bias": (d.Units_Forecast.sum() - d.Units.sum()) / d.Units.sum(), "Rows": len(d)})
    return bt.groupby(by).apply(metrics, include_groups=False).reset_index()


def error_quantiles(bt_best: pd.DataFrame) -> pd.DataFrame:
    d = bt_best[bt_best.Units_Forecast > 0].copy()
    d["Ratio"] = d.Units / d.Units_Forecast
    return d.groupby("Location_ID").Ratio.quantile([0.1, 0.9]).unstack().rename(columns={0.1: "P10_Ratio", 0.9: "P90_Ratio"}).reset_index()


def horizon_summary(fc: pd.DataFrame) -> pd.DataFrame:
    """30/60/90-day cumulative forecast per location x product (months 1..3 after the origin)."""
    months = sorted(fc.Month.unique())[:3]
    p = fc[fc.Month.isin(months)].pivot_table(index=["Location_ID", "Product_ID"], columns="Month", values="Units_Forecast").reset_index()
    p.columns = ["Location_ID", "Product_ID"] + [f"M{i}" for i in range(1, len(months) + 1)]
    p["Forecast_30D"], p["Forecast_60D"], p["Forecast_90D"] = p.M1, p.M1 + p.M2, p.M1 + p.M2 + p.M3
    return p


def demand_over_days(monthly: np.ndarray, days: float, start_month: pd.Timestamp) -> float:
    """Walk the monthly forecast day by day; used for lead-time demand."""
    total, remaining, month = 0.0, days, start_month
    for value in monthly:
        dim = month.days_in_month
        take = min(remaining, dim)
        total += value / dim * take
        remaining -= take
        month = month + pd.offsets.MonthBegin(1)
        if remaining <= 0:
            break
    return total


def replenishment_plan(fc: pd.DataFrame, inv_latest: pd.DataFrame, unit_cost: pd.DataFrame, location_wape: dict[str, float], review_days: int = 30) -> pd.DataFrame:
    """Reorder-point / order-up-to rules driven by the forecast and the backtested error."""
    months = sorted(fc.Month.unique())
    wide = fc.pivot_table(index=["Location_ID", "Product_ID"], columns="Month", values="Units_Forecast").reset_index()
    plan = inv_latest.merge(wide, on=["Location_ID", "Product_ID"], how="left", validate="one_to_one").merge(unit_cost, on="Product_ID", how="left")
    month_cols = [c for c in plan.columns if isinstance(c, pd.Timestamp)]
    start = months[0]

    def row_calc(r: pd.Series) -> pd.Series:
        monthly = np.array([r[c] for c in month_cols])
        lt = r.Supplier_Lead_Time_Days
        lt_demand = demand_over_days(monthly, lt, start)
        cycle_demand = demand_over_days(monthly, lt + review_days, start)
        sigma_month = location_wape.get(r.Location_ID, 0.3) * monthly[:3].mean()
        safety = Z_95 * (sigma_month / np.sqrt(30)) * np.sqrt(lt)
        d90 = demand_over_days(monthly, 90, start)
        return pd.Series({"Lead_Time_Demand": lt_demand, "Model_Safety_Stock": safety, "Model_Reorder_Point": lt_demand + safety, "Order_Up_To": cycle_demand + safety, "Demand_90D": d90})

    calc = plan.apply(row_calc, axis=1)
    plan = pd.concat([plan.drop(columns=month_cols), calc], axis=1)
    plan["Suggested_Order_Qty"] = np.where(plan.Closing_Stock <= plan.Model_Reorder_Point, np.ceil((plan.Order_Up_To - plan.Closing_Stock).clip(lower=0)), 0)
    plan["Action"] = np.select(
        [plan.Closing_Stock <= 0.5 * plan.Model_Reorder_Point, plan.Closing_Stock <= plan.Model_Reorder_Point, plan.Closing_Stock > plan.Demand_90D + plan.Model_Safety_Stock],
        ["Order now (urgent)", "Order", "Surplus: hold / rebalance"], "OK",
    )
    plan["Surplus_Units"] = (plan.Closing_Stock - (plan.Demand_90D + plan.Model_Safety_Stock)).clip(lower=0)
    plan["Deficit_Units"] = (plan.Order_Up_To - plan.Closing_Stock).clip(lower=0).where(plan.Closing_Stock <= plan.Model_Reorder_Point, 0)
    plan["Order_Value_EGP"] = plan.Suggested_Order_Qty * plan.Unit_COGS
    plan["Surplus_Value_EGP"] = plan.Surplus_Units * plan.Unit_COGS
    return plan


def rebalancing_suggestions(plan: pd.DataFrame) -> pd.DataFrame:
    """Match deficits to surpluses of the same SKU in other locations (transfer before buying)."""
    deficits = plan[plan.Deficit_Units > 0][["Product_ID", "Product_Name", "Location_ID", "Location_Name", "Deficit_Units", "Unit_COGS"]]
    surpluses = plan[plan.Surplus_Units > 0][["Product_ID", "Location_ID", "Location_Name", "Surplus_Units"]]
    rows = []
    for _, d in deficits.iterrows():
        for _, s in surpluses[(surpluses.Product_ID == d.Product_ID) & (surpluses.Location_ID != d.Location_ID)].sort_values("Surplus_Units", ascending=False).iterrows():
            qty = min(d.Deficit_Units, s.Surplus_Units)
            if qty >= 1:
                rows.append({"Product_ID": d.Product_ID, "Product_Name": d.Product_Name, "From": s.Location_Name, "To": d.Location_Name, "Transfer_Units": np.floor(qty), "Value_EGP": np.floor(qty) * d.Unit_COGS})
    out = pd.DataFrame(rows)
    return out.sort_values("Value_EGP", ascending=False).reset_index(drop=True) if not out.empty else out


# =========================== 2. WHOLESALE DISCOUNT GOVERNANCE (OPPORTUNITY)

TIER_CAPS = [(12, 23, 0.28), (24, 47, 0.30), (48, 10_000, 0.32)]  # quantity band -> max discount
MARGIN_FLOOR = 0.15


def wholesale_lines(sales: pd.DataFrame) -> pd.DataFrame:
    w = sales[sales.Channel == "Wholesale"].sort_values(["Customer_ID", "Order_Date"]).copy()
    g = w.groupby("Customer_ID")
    w["Prior_Orders"] = g.cumcount()
    w["Prior_Spend"] = g.Net_Sales_EGP.cumsum() - w.Net_Sales_EGP
    w["Prior_Mean_Discount"] = (g.Discount_Pct.cumsum() - w.Discount_Pct) / w.Prior_Orders.replace(0, np.nan)
    w["Tenure_Days"] = (w.Order_Date - w.Join_Date).dt.days
    w["Cogs_Ratio"] = w.Unit_COGS_EGP / w.List_Price_EGP
    return w


def discount_diagnostic(w: pd.DataFrame, split_date: str = "2026-01-01") -> dict:
    """Can anything we know at quote time predict the discount? (If not, discounts are discretionary.)"""
    features = ["Quantity", "List_Price_EGP", "Prior_Orders", "Prior_Spend", "Prior_Mean_Discount", "Tenure_Days", "Cogs_Ratio"]
    cats = ["Product_Family", "Customer_Governorate", "Payment_Method"]
    X = w[features + cats].copy()
    for c in cats:
        X[c] = X[c].astype("category")
    y = w.Discount_Pct
    train, test = w.Order_Date < split_date, w.Order_Date >= split_date
    model = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=200, categorical_features="from_dtype", random_state=42)
    model.fit(X[train], y[train])
    pred = model.predict(X[test])
    baseline = np.full(test.sum(), y[train].mean())
    imp = permutation_importance(model, X[test], y[test], n_repeats=5, random_state=42, scoring="r2")
    importance = pd.DataFrame({"Feature": X.columns, "R2_Drop": imp.importances_mean}).sort_values("R2_Drop", ascending=False)
    margin_fit = smf.ols("Gross_Margin_Pct ~ Discount_Pct", data=w).fit()
    joblib.dump(model, MODELS / "discount_diagnostic_hgb.joblib")
    return {
        "train_rows": int(train.sum()), "test_rows": int(test.sum()), "target": "Discount_Pct (wholesale invoice line)",
        "baseline_r2": float(r2_score(y[test], baseline)), "baseline_mae": float(mean_absolute_error(y[test], baseline)),
        "model_r2": float(r2_score(y[test], pred)), "model_mae": float(mean_absolute_error(y[test], pred)),
        "importance": importance, "margin_vs_discount_r2": float(margin_fit.rsquared), "margin_slope": float(margin_fit.params["Discount_Pct"]),
    }


def price_response_test(w: pd.DataFrame) -> pd.DataFrame:
    """Does a deeper discount buy bigger orders or faster re-ordering? Pooled and within-customer estimates."""
    w = w.copy()
    w["log_qty"] = np.log(w.Quantity)
    w["disc_pts"] = w.Discount_Pct * 100  # coefficient = effect per +1 discount point
    w["Cust_Mean_Disc"] = w.groupby("Customer_ID").disc_pts.transform("mean")
    w["Disc_Dev"] = w.disc_pts - w.Cust_Mean_Disc
    rows = []
    m1 = smf.ols("log_qty ~ disc_pts + C(Product_Family) + C(Month)", data=w).fit(cov_type="HC1")
    rows.append(("Order size: pooled (log qty per +1pt discount)", m1.params["disc_pts"], *m1.conf_int().loc["disc_pts"], m1.pvalues["disc_pts"], len(w)))
    multi = w[w.groupby("Customer_ID").Customer_ID.transform("count") > 1]
    m2 = smf.ols("log_qty ~ Disc_Dev + C(Product_Family) + C(Month)", data=multi).fit(cov_type="cluster", cov_kwds={"groups": multi.Customer_ID})
    rows.append(("Order size: within-customer deviation", m2.params["Disc_Dev"], *m2.conf_int().loc["Disc_Dev"], m2.pvalues["Disc_Dev"], len(multi)))
    w = w.sort_values(["Customer_ID", "Order_Date"])
    w["Days_To_Next_Order"] = w.groupby("Customer_ID").Order_Date.shift(-1).sub(w.Order_Date).dt.days
    nxt = w.dropna(subset=["Days_To_Next_Order"])
    m3 = smf.ols("Days_To_Next_Order ~ disc_pts + C(Product_Family)", data=nxt).fit(cov_type="cluster", cov_kwds={"groups": nxt.Customer_ID})
    rows.append(("Days to customer's next order (per +1pt discount)", m3.params["disc_pts"], *m3.conf_int().loc["disc_pts"], m3.pvalues["disc_pts"], len(nxt)))
    return pd.DataFrame(rows, columns=["Test", "Coefficient", "CI_Low", "CI_High", "p_value", "Rows"])


def discount_cap(qty: pd.Series, policy: str, cogs_ratio: pd.Series | None = None) -> pd.Series:
    tier = pd.Series(np.select([qty.between(lo, hi) for lo, hi, _ in TIER_CAPS], [cap for *_, cap in TIER_CAPS], default=0.30), index=qty.index)
    floor_cap = 1 - cogs_ratio / (1 - MARGIN_FLOOR) if cogs_ratio is not None else pd.Series(1.0, index=qty.index)
    if policy == "Flat cap 30%":
        return pd.Series(0.30, index=qty.index)
    if policy == "Volume-tiered cap":
        return tier
    if policy == "Margin floor 15%":
        return floor_cap
    if policy == "Tiered cap + margin floor":
        return np.minimum(tier, floor_cap)
    raise ValueError(policy)


POLICIES = ["Flat cap 30%", "Volume-tiered cap", "Margin floor 15%", "Tiered cap + margin floor"]


def simulate_policy(w: pd.DataFrame, policy: str, volume_loss: float = 0.0) -> dict:
    """Re-price historical wholesale lines under a cap. `volume_loss` = share of units lost on re-priced lines."""
    cap = discount_cap(w.Quantity, policy, w.Cogs_Ratio)
    new_disc = np.minimum(w.Discount_Pct, cap)
    affected = new_disc < w.Discount_Pct - 1e-9
    keep = np.where(affected, 1 - volume_loss, 1.0)
    new_net = w.Gross_Sales_EGP * (1 - new_disc) * keep
    new_cogs = w.Estimated_COGS_EGP * keep
    new_gp = new_net - new_cogs
    return {
        "Policy": policy, "Volume_Loss_On_Repriced": volume_loss, "Lines_Repriced": int(affected.sum()), "Share_Lines_Repriced": float(affected.mean()),
        "Old_Net": float(w.Net_Sales_EGP.sum()), "New_Net": float(new_net.sum()), "Old_GP": float(w.Gross_Profit_EGP.sum()), "New_GP": float(new_gp.sum()),
        "GP_Uplift": float(new_gp.sum() - w.Gross_Profit_EGP.sum()), "Old_GM": float(w.Gross_Profit_EGP.sum() / w.Net_Sales_EGP.sum()), "New_GM": float(new_gp.sum() / new_net.sum()),
        "Loss_Making_Lines_After": int((new_gp < 0).sum()),
    }


def policy_table(w: pd.DataFrame, losses: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)) -> pd.DataFrame:
    rows = [simulate_policy(w, p, v) for p in POLICIES for v in losses]
    t = pd.DataFrame(rows)
    years = (w.Order_Date.max() - w.Order_Date.min()).days / 365.25
    t["GP_Uplift_per_Year"] = t.GP_Uplift / years
    return t


def breakeven_volume_loss(w: pd.DataFrame, policy: str) -> float:
    """Volume loss on re-priced lines at which the policy stops adding gross profit."""
    cap = discount_cap(w.Quantity, policy, w.Cogs_Ratio)
    new_disc = np.minimum(w.Discount_Pct, cap)
    a = new_disc < w.Discount_Pct - 1e-9
    old_gp = w.Gross_Profit_EGP[a].sum()
    new_gp = (w.Gross_Sales_EGP[a] * (1 - new_disc[a]) - w.Estimated_COGS_EGP[a]).sum()
    return float(1 - old_gp / new_gp) if new_gp > 0 else float("nan")


def leakage_breakdown(w: pd.DataFrame, policy: str, by: str) -> pd.DataFrame:
    cap = discount_cap(w.Quantity, policy, w.Cogs_Ratio)
    excess = (w.Discount_Pct - np.minimum(w.Discount_Pct, cap)).clip(lower=0)
    d = w.assign(Leakage_EGP=excess * w.Gross_Sales_EGP)
    g = d.groupby(by).agg(Lines=("Invoice_ID", "count"), Net_Sales=("Net_Sales_EGP", "sum"), Gross_Profit=("Gross_Profit_EGP", "sum"), Leakage_EGP=("Leakage_EGP", "sum")).reset_index()
    g["Leakage_Pct_of_GP"] = g.Leakage_EGP / g.Gross_Profit
    return g.sort_values("Leakage_EGP", ascending=False)


def evaluate_quote(product_id: str, quantity: int, discount: float, products: pd.DataFrame, unit_costs: pd.DataFrame) -> dict:
    """Deal-desk check for a wholesale quote: margin, tier cap, margin-floor cap and approval route."""
    price = float(products.loc[products.Product_ID == product_id, "Public_Retail_Price_EGP"].iat[0])
    cost = float(unit_costs.loc[unit_costs.Product_ID == product_id, "Unit_COGS"].iat[0])
    qty = pd.Series([quantity])
    tier_cap = float(discount_cap(qty, "Volume-tiered cap").iat[0])
    floor_cap = float(discount_cap(qty, "Margin floor 15%", pd.Series([cost / price])).iat[0])
    counter = min(tier_cap, floor_cap, discount)
    net_unit = price * (1 - discount)
    margin = (net_unit - cost) / net_unit
    if net_unit < cost:
        status = "REJECT - below cost"
    elif discount <= tier_cap + 1e-9 and margin >= MARGIN_FLOOR:
        status = "AUTO-APPROVE"
    elif margin >= 0.08:
        status = "MANAGER APPROVAL"
    else:
        status = "DIRECTOR APPROVAL - margin under 8%"
    return {
        "Product_ID": product_id, "Quantity": quantity, "Quoted_Discount": discount, "Tier_Cap": tier_cap, "Margin_Floor_Cap": floor_cap,
        "Counter_Offer_Discount": counter, "Unit_Net_Price": net_unit, "Unit_COGS": cost, "Gross_Margin": margin,
        "Order_GP_EGP": (net_unit - cost) * quantity, "Order_GP_at_Counter_EGP": (price * (1 - counter) - cost) * quantity, "Status": status,
    }


def save_policy_artifact(w: pd.DataFrame) -> dict:
    phase1, phase2 = "Volume-tiered cap", "Tiered cap + margin floor"
    artifact = {
        "phase_1_policy": phase1, "tier_caps": [{"min_qty": lo, "max_qty": hi, "max_discount": cap} for lo, hi, cap in TIER_CAPS],
        "phase_2_policy": phase2, "margin_floor": MARGIN_FLOOR,
        "approval_rule": "within tier cap and margin >= floor: auto; margin >= 8%: manager; below 8%: director; below cost: reject",
        "historical_simulation": {p: simulate_policy(w, p, 0.0) for p in (phase1, phase2)},
        "breakeven_volume_loss": {p: breakeven_volume_loss(w, p) for p in (phase1, phase2)},
    }
    (MODELS / "deal_desk_policy.json").write_text(json.dumps(artifact, indent=2))
    return artifact


# ============================================ 3. FY27 VALUE-AT-STAKE

def fy27_base(fc12: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """Base-case FY27 revenue and gross profit: forecast units x last-12-month net price and GP per unit."""
    last12 = sales[sales.Order_Date > sales.Order_Date.max() - pd.DateOffset(months=12)]
    unit = last12.groupby(["Channel", "Product_ID"]).agg(Net=("Net_Sales_EGP", "sum"), GP=("Gross_Profit_EGP", "sum"), Units=("Quantity", "sum")).reset_index()
    unit["Net_per_Unit"], unit["GP_per_Unit"] = unit.Net / unit.Units, unit.GP / unit.Units
    f = fc12.assign(Channel=fc12.Location_ID.map(LOCATION_CHANNEL)).merge(unit[["Channel", "Product_ID", "Net_per_Unit", "GP_per_Unit"]], on=["Channel", "Product_ID"], how="left")
    f["Net_Sales"], f["Gross_Profit"] = f.Units_Forecast * f.Net_per_Unit, f.Units_Forecast * f.GP_per_Unit
    return f.groupby("Channel").agg(Units=("Units_Forecast", "sum"), Net_Sales=("Net_Sales", "sum"), Gross_Profit=("Gross_Profit", "sum")).reset_index()


def value_at_stake(sales: pd.DataFrame, w: pd.DataFrame, c: dict) -> pd.DataFrame:
    """Annual gross-profit impact per action, with low / base / high assumptions (EGP per year).

    `c` carries pre-computed inputs from the analysis layer (kept explicit so every number is auditable).
    """
    years = (w.Order_Date.max() - w.Order_Date.min()).days / 365.25
    be = breakeven_volume_loss(w, "Volume-tiered cap")
    tiered = lambda loss: simulate_policy(w, "Volume-tiered cap", loss)["GP_Uplift"] / years
    floor_extra = lambda loss: (simulate_policy(w, "Tiered cap + margin floor", loss)["GP_Uplift"] - simulate_policy(w, "Volume-tiered cap", loss)["GP_Uplift"]) / years
    rows = [
        ("1. Wholesale discount governance - Phase 1: volume-tiered cap", "Margin",
         tiered(0.20), tiered(0.10), tiered(0.0), "Volume loss on re-priced lines 20% / 10% / 0%; observed price response ~0"),
        ("1b. Wholesale discount governance - Phase 2: margin-floor exceptions", "Margin",
         floor_extra(0.20), floor_extra(0.10), floor_extra(0.0), "Incremental to Phase 1, same volume-loss assumptions"),
        ("2. Win-back of lapsed high-frequency retail customers", "Revenue retention",
         c["winback_pool_annual_gp"] * 0.05, c["winback_pool_annual_gp"] * 0.10, c["winback_pool_annual_gp"] * 0.15, "Reactivate 5% / 10% / 15% of the At-Risk (high frequency) pool"),
        ("3. Marketing reallocation (Influencer/Meta/TikTok to Google)", "Efficiency",
         c["realloc_gp_low"], c["realloc_gp_base"], c["realloc_gp_high"], "Google marginal ROAS = 25% / 50% / 100% of average; revenue reconciled to actual sales"),
        ("4. Cost of poor quality reduction", "Cost",
         c["qc_loss_annual"] * 0.10, c["qc_loss_annual"] * 0.20, c["qc_loss_annual"] * 0.30, "QC-loss reduction 10% / 20% / 30%"),
        ("5. Online size/fit return reduction", "Margin",
         c["online_sizefit_refund_annual"] * 0.15 * c["online_gm"], c["online_sizefit_refund_annual"] * 0.25 * c["online_gm"], c["online_sizefit_refund_annual"] * 0.40 * c["online_gm"], "Fewer size/fit returns 15% / 25% / 40% (size guidance, fit charts)"),
        ("6. Wholesale stockout recovery", "Revenue retention",
         c["stockout_gp_annual"] * 0.25, c["stockout_gp_annual"] * 0.50, c["stockout_gp_annual"] * 0.75, "Recover 25% / 50% / 75% of est. lost wholesale GP via forecast-based replenishment"),
    ]
    t = pd.DataFrame(rows, columns=["Action", "Lever", "Low", "Base", "High", "Assumption"])
    t.attrs["breakeven_volume_loss"] = be
    return t
