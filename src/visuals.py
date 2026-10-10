"""Static figures (PNG) used by notebooks and the final report.

Every function returns the matplotlib Figure; `save_fig` writes it to outputs/figures.
Titles state the finding, not just the metric.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from utils import CHANNEL_COLORS, PALETTE, millions, set_style

set_style()
P = PALETTE


def _pct_axis(ax, axis: str = "y", decimals: int = 0) -> None:
    fmt = mtick.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


# ------------------------------------------------------------------ Case 01
def revenue_margin_trend(monthly: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(monthly.Month, monthly.Net_Sales, width=22, color=P["primary"], alpha=0.85, label="Net sales")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(millions))
    ax.set_ylabel("Net sales (EGP)")
    ax2 = ax.twinx()
    ax2.plot(monthly.Month, monthly.Gross_Margin, color=P["accent"], lw=2.2, marker="o", ms=3, label="Gross margin")
    ax2.set_ylim(0.10, 0.20)
    _pct_axis(ax2)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax.set_title("Revenue grows with the seasons, but gross margin stays near 15%")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    return fig


def channel_share_revenue_vs_profit(scorecard: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    order = ["Wholesale", "Online", "Retail"]
    d = scorecard.set_index("Channel").loc[order]
    y = np.arange(len(order))
    ax.barh(y + 0.19, d.Net_Share, height=0.36, color=P["primary"], label="Share of net sales")
    ax.barh(y - 0.19, d.GP_Share, height=0.36, color=P["good"], label="Share of gross profit")
    for i, ch in enumerate(order):
        ax.text(d.Net_Share[ch] + 0.01, i + 0.19, f"{d.Net_Share[ch]:.0%}", va="center")
        ax.text(d.GP_Share[ch] + 0.01, i - 0.19, f"{d.GP_Share[ch]:.0%}", va="center")
    ax.set_yticks(y, order)
    _pct_axis(ax, "x")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_title("Wholesale is 85% of revenue but only 66% of gross profit")
    ax.legend(loc="lower right")
    return fig


def seasonality_by_group(si: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    names = {"Pajamas/Homewear": P["primary"], "T-Shirts": P["good"], "Kids Sets": P["warn"], "Sweatshirts": P["accent"]}
    for fam, d in si.groupby("Product_Family"):
        ax.plot(d.Calendar_Month, d.Index, marker="o", lw=2, label=fam, color=names.get(fam))
    ax.axhline(1, color=P["muted"], lw=1, ls="--")
    ax.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_ylabel("Seasonal index (1.0 = average month)")
    ax.set_title("Two opposite seasons: summer lines peak Jun-Aug, sweatshirts peak Nov-Dec")
    ax.legend(ncol=4, loc="upper center")
    return fig


def discount_band_margin(di: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.4))
    bands = ["0%", "1-10%", "11-20%", "21-30%", "31-40%"]
    for ch, color in CHANNEL_COLORS.items():
        d = di[di.Channel == ch].set_index("Discount_Band").reindex(bands)
        ax.plot(bands, d.Gross_Margin, marker="o", lw=2.2, color=color, label=ch)
    ax.axhline(0, color="black", lw=0.8)
    _pct_axis(ax)
    ax.set_ylabel("Gross margin")
    ax.set_xlabel("Discount band")
    ax.set_title("Each discount band costs margin; wholesale deals above 30% leave ~8%")
    ax.legend()
    return fig


def product_gp_contribution(pc: pd.DataFrame, top: int = 15) -> plt.Figure:
    d = pc.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.barh(d.Product_Name.str.slice(0, 38), d.Gross_Profit, color=P["primary"])
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(millions))
    for i, (gp, gm) in enumerate(zip(d.Gross_Profit, d.Gross_Margin)):
        ax.text(gp, i, f"  {gm:.0%}", va="center", fontsize=8, color=P["accent"])
    ax.set_title(f"Top {top} products by gross profit (label = margin)")
    return fig


# ------------------------------------------------------------------ Case 02
def rfm_share(seg: pd.DataFrame) -> plt.Figure:
    d = seg.sort_values("Sales_Share")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(d))
    ax.barh(y + 0.2, d.Sales_Share, height=0.38, color=P["primary"], label="Share of retail-customer sales")
    ax.barh(y - 0.2, d.Customer_Share, height=0.38, color=P["muted"], label="Share of customers")
    ax.set_yticks(y, d.Segment)
    _pct_axis(ax, "x")
    ax.set_title("'At Risk (high frequency)' holds 16% of customers and 32% of sales")
    ax.legend(loc="lower right")
    return fig


def cohort_heatmap(ret: pd.DataFrame) -> plt.Figure:
    data = ret.drop(columns="Cohort_Size")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    im = ax.imshow(data.values.astype(float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(data.shape[1]), [f"Q+{c}" for c in data.columns])
    ax.set_yticks(range(data.shape[0]), [str(i) for i in data.index])
    ax.grid(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white" if v > 0.55 else "black")
    ax.set_title("Newer cohorts re-buy far less: next-quarter retention fell from 91% to ~44%")
    fig.colorbar(im, ax=ax, format=mtick.PercentFormatter(1.0), shrink=0.8)
    return fig


# ------------------------------------------------------------------ Case 03
def channel_scorecard_panels(sc: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    order = ["Wholesale", "Retail", "Online"]
    d = sc.set_index("Channel").loc[order]
    specs = [("Gross_Margin", "Gross margin", True), ("Refund_Rate", "Refund rate (value)", True), ("GP_per_Unit", "Gross profit per unit (EGP)", False), ("Units_per_Invoice", "Basket size (units/invoice)", False)]
    for ax, (col, title, pct) in zip(axes, specs):
        ax.bar(order, d[col], color=[CHANNEL_COLORS[c] for c in order])
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        if pct:
            _pct_axis(ax)
        for i, v in enumerate(d[col]):
            ax.text(i, v, f"{v:.1%}" if pct else f"{v:,.1f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Wholesale brings volume; retail and online bring 4x the profit per unit", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def channel_growth_lines(growth: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ch, color in CHANNEL_COLORS.items():
        d = growth[growth.Channel == ch]
        axes[0].plot(d.Fiscal_Year, d.Net_Sales / d.Net_Sales.iloc[0] * 100, marker="o", lw=2.2, color=color, label=ch)
        axes[1].plot(d.Fiscal_Year, d.Gross_Profit / d.Gross_Profit.iloc[0] * 100, marker="o", lw=2.2, color=color, label=ch)
    axes[0].set_title("Net sales (FY24 = 100)")
    axes[1].set_title("Gross profit (FY24 = 100)")
    axes[0].legend()
    fig.suptitle("Online is the growth engine (+60% in two years); retail is shrinking (-5%)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def geo_stack(geo: pd.DataFrame, top: int = 10) -> plt.Figure:
    p = geo.pivot_table(index="Customer_Governorate", columns="Channel", values="Net_Sales", aggfunc="sum", fill_value=0)
    p = p.loc[p.sum(axis=1).sort_values(ascending=False).head(top).index].iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    left = np.zeros(len(p))
    for ch in ["Wholesale", "Retail", "Online"]:
        ax.barh(p.index, p[ch], left=left, color=CHANNEL_COLORS[ch], label=ch)
        left += p[ch].values
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(millions))
    ax.set_title("Alexandria and Cairo generate 47% of net sales across all channels")
    ax.legend()
    return fig


# ------------------------------------------------------------------ Case 04
def inventory_cover_by_location(diag: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    order = ["El-Sharnouby Wholesale Desk", "LEO MODA - Sidi Bishr Store", "LEO MODA Online Store"]
    short = ["Wholesale", "Retail store", "Online"]
    data = [diag[diag.Location_Name == o].Days_of_Cover.clip(upper=400) for o in order]
    bp = axes[0].boxplot(data, tick_labels=short, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [CHANNEL_COLORS["Wholesale"], CHANNEL_COLORS["Retail"], CHANNEL_COLORS["Online"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axes[0].set_ylabel("Days of cover (Sep-2026)")
    axes[0].set_title("Days of cover per SKU")
    st = diag.groupby(["Location_Name", "Status"]).size().unstack(fill_value=0).loc[order]
    colors = {"Stockout": P["accent"], "Below reorder point": P["warn"], "Healthy": P["good"], "Overstock": P["muted"]}
    bottom = np.zeros(len(order))
    for status in ["Stockout", "Below reorder point", "Healthy", "Overstock"]:
        axes[1].bar(short, st.get(status, 0), bottom=bottom, color=colors[status], label=status)
        bottom += st.get(status, pd.Series(0, index=order)).values
    axes[1].set_title("SKU status (59 SKUs per location)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Wholesale runs on ~11 days of cover; retail and online hold ~4 months", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def abc_xyz_matrix(diag: pd.DataFrame) -> plt.Figure:
    t = diag.groupby(["ABC", "XYZ"]).Net_Sales_12M.sum().unstack(fill_value=0).reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)
    n = diag.groupby(["ABC", "XYZ"]).size().unstack(fill_value=0).reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.imshow(t.values, cmap="Blues", aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(3), ["X (stable)", "Y (variable)", "Z (erratic)"])
    ax.set_yticks(range(3), ["A (80% value)", "B (next 15%)", "C (last 5%)"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{n.values[i, j]} SKUs\n{t.values[i, j] / 1e6:.1f}M", ha="center", va="center", fontsize=9, color="white" if t.values[i, j] > t.values.max() * 0.5 else "black")
    ax.set_title("ABC / XYZ: demand is seasonal, not erratic (no Z-class)")
    return fig


def forecast_vs_actual(hist: pd.DataFrame, fc: pd.DataFrame, bt: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    act = hist.groupby("Month").Units.sum()
    axes[0].plot(act.index, act.values, color=P["primary"], lw=2.2, label="Actual units")
    f = fc.groupby("Month").Units_Forecast.sum()
    axes[0].plot(f.index, f.values, color=P["accent"], lw=2.2, ls="--", marker="o", ms=3, label="Forecast (12 months)")
    axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(millions))
    axes[0].set_title("Group units: history and 12-month forecast")
    axes[0].legend()
    metrics = bt.groupby("Method").apply(lambda d: (d.Units - d.Units_Forecast).abs().sum() / d.Units.sum(), include_groups=False).reindex(["naive3", "seasonal_naive", "si", "si_growth"])
    labels = ["Naive\n(3-mo avg)", "Seasonal\nnaive", "Seasonal\nindex", "Seasonal index\n+ growth"]
    bars = axes[1].bar(labels, metrics.values, color=[P["muted"], P["muted"], P["good"], P["primary"]])
    for b, v in zip(bars, metrics.values):
        axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.0%}", ha="center", va="bottom")
    _pct_axis(axes[1])
    axes[1].set_title("Backtest error (WAPE, SKU x location, 3 origins x 3 months)")
    fig.tight_layout()
    return fig


def stockout_timeline(so: pd.DataFrame) -> plt.Figure:
    w = so[so.Location_Name.str.contains("Wholesale")]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(w.Month, w.Stockout_Days, width=22, color=P["accent"])
    ax.set_ylabel("Stockout SKU-days")
    ax.set_title("Wholesale stockouts cluster in peak months (retail/online: almost none)")
    return fig


# ------------------------------------------------------------------ Case 05
def platform_roas(pt: pd.DataFrame, breakeven: float) -> plt.Figure:
    d = pt.sort_values("ROAS")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].barh(d.Platform, d.ROAS, color=P["primary"])
    axes[0].set_xscale("log")
    axes[0].axvline(breakeven, color=P["accent"], ls="--")
    axes[0].text(breakeven * 1.05, 0.05, f"gross-profit\nbreak-even {breakeven:.1f}x", color=P["accent"], fontsize=8)
    axes[0].set_title("Reported ROAS (log scale)")
    x = np.arange(len(d))
    axes[1].bar(x - 0.2, d.Spend_Share, width=0.4, color=P["muted"], label="Share of spend")
    axes[1].bar(x + 0.2, d.Revenue_Share, width=0.4, color=P["good"], label="Share of attributed revenue")
    axes[1].set_xticks(x, d.Platform, rotation=20)
    _pct_axis(axes[1])
    axes[1].legend()
    axes[1].set_title("Budget vs. attributed return")
    fig.suptitle("Meta takes 37% of spend for 13% of revenue; Influencer does not cover its cost", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def campaign_scatter(m: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = {"Meta": "#1877F2", "Google": "#34A853", "TikTok": "#111111", "Influencer": P["accent"], "WhatsApp / CRM": P["warn"]}
    for platform, d in m.groupby("Platform"):
        ax.scatter(d.Spend_EGP, d.ROAS, s=22, alpha=0.7, color=colors.get(platform), label=platform)
    ax.set_yscale("log")
    ax.axhline(1, color="black", lw=0.8)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(millions))
    ax.set_xlabel("Campaign spend (EGP)")
    ax.set_ylabel("ROAS (log)")
    ax.set_title("ROAS is set by platform, not by campaign size: no sign of saturation")
    ax.legend(ncol=2, fontsize=8)
    return fig


def web_funnel_panels(f: pd.DataFrame, load: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(f.Month, f.Conversion, color=P["primary"], lw=2.2)
    _pct_axis(axes[0], decimals=1)
    axes[0].set_title("Monthly web conversion (orders / sessions)")
    axes[1].bar(load.Load_Band.astype(str), load.Conversion, color=[P["good"]] * 4 + [P["accent"]])
    _pct_axis(axes[1], decimals=1)
    axes[1].set_title("Conversion by page-load time")
    for i, v in enumerate(load.Conversion):
        axes[1].text(i, v, f"{v:.1%}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Conversion is stable (~2.9%) but collapses to 2.0% when pages take >3.5s", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ Case 06
def line_delay_chart(sc: pd.DataFrame, sig: pd.DataFrame) -> plt.Figure:
    d = sc.merge(sig[["Line_ID", "p_value"]], on="Line_ID")
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [P["accent"] if p < 0.05 else P["primary"] for p in d.p_value]
    ax.bar(d.Line_ID, d.Delayed_Share, color=colors)
    ax.axhline(d.Delayed_Share.mean(), color=P["muted"], ls="--")
    _pct_axis(ax)
    for i, (v, p) in enumerate(zip(d.Delayed_Share, d.p_value)):
        ax.text(i, v, f"{v:.1%}", ha="center", va="bottom")
    ax.set_title("LINE-04 is the only line with a statistically higher delay rate (p<0.01)")
    return fig


def bottleneck_pareto_chart(bp: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(bp.Primary_Bottleneck, bp.Delay_Days, color=P["primary"])
    ax.set_ylabel("Delay days")
    ax2 = ax.twinx()
    ax2.plot(bp.Primary_Bottleneck, bp.Cum_Share, color=P["accent"], marker="o")
    ax2.set_ylim(0, 1.05)
    _pct_axis(ax2)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax.set_title("Sewing and Finishing cause 68% of all delay days")
    return fig


def overtime_delay_chart(ot: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(ot.OT_Band.astype(str), ot.Delayed_Share, color=[P["good"], P["good"], P["warn"], P["accent"], P["accent"]])
    _pct_axis(ax)
    for i, v in enumerate(ot.Delayed_Share):
        ax.text(i, v, f"{v:.0%}", ha="center", va="bottom")
    ax.set_xlabel("Overtime hours per order")
    ax.set_title("Orders that need >15h overtime are late 66% of the time")
    return fig


# ------------------------------------------------------------------ Case 07
def defect_pareto_chart(dp: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(dp.Defect_Type, dp.Loss_EGP, color=[P["warn"] if d == "Unclassified" else P["primary"] for d in dp.Defect_Type])
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(millions))
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax2 = ax.twinx()
    ax2.plot(dp.Defect_Type, dp.Cum_Share, color=P["accent"], marker="o")
    ax2.set_ylim(0, 1.05)
    _pct_axis(ax2)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax.set_title("No dominant defect: losses are spread evenly, and 12% is not even classified")
    return fig


def copq_chart(copq: pd.DataFrame) -> plt.Figure:
    d = copq.sort_values("EGP")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.barh(d.Component, d.EGP, color=[P["accent"] if t == "External failure" else P["primary"] for t in d.Type])
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(millions))
    total = d.EGP.sum()
    ax.set_title(f"Estimated cost of poor quality: EGP {total / 1e6:.1f}M over 3 years ({d.Pct_of_Gross_Profit.sum():.0%} of gross profit)")
    return fig


def return_reason_chart(rr: pd.DataFrame) -> plt.Figure:
    p = rr.pivot(index="Return_Reason", columns="Channel", values="Lines").fillna(0)
    p = p.loc[p.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    left = np.zeros(len(p))
    for ch in ["Wholesale", "Retail", "Online"]:
        ax.barh(p.index, p[ch], left=left, color=CHANNEL_COLORS[ch], label=ch)
        left += p[ch].values
    ax.set_title("Size/fit is the #1 return reason (36%) - not a factory quality problem")
    ax.legend()
    return fig


# ------------------------------------------------------------------ Case 08
def printer_priority_chart(ms: pd.DataFrame) -> plt.Figure:
    d = ms.sort_values("Machine_ID")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].bar(d.Machine_ID, d.Reject_Rate, color=[P["accent"] if r == d.Reject_Rate.max() else P["primary"] for r in d.Reject_Rate])
    _pct_axis(axes[0], decimals=1)
    axes[0].set_title("Reject rate")
    axes[1].bar(d.Machine_ID, d.Downtime_Hours, color=[P["accent"] if r == d.Downtime_Hours.max() else P["primary"] for r in d.Downtime_Hours])
    axes[1].set_title("Downtime hours (3 years)")
    axes[2].bar(d.Machine_ID, d.Cost_per_Good_Unit, color=P["primary"])
    axes[2].set_title("Print cost per good unit (EGP)")
    fig.suptitle("PR-M03 rejects the most; PR-M02 loses the most hours", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def downtime_reason_chart(dr: pd.DataFrame) -> plt.Figure:
    d = dr.sort_values("Hours")
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.barh(d.Reason, d.Hours, color=[P["muted"] if p == "Yes" else P["accent"] for p in d.Planned])
    ax.set_title("78% of downtime is unplanned; mechanical faults and set-ups lead")
    return fig


# ------------------------------------------------------------------ Case 09
def supplier_otd_trend_chart(trend: pd.DataFrame, problem: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for sup in trend.columns:
        hot = sup in problem
        ax.plot(trend.index, trend[sup], lw=2.6 if hot else 1, color=P["accent"] if hot else P["muted"], alpha=1 if hot else 0.7, label=sup if hot else None)
        if hot:
            ax.text(trend.index[-1], trend[sup].iloc[-1], f" {sup}", color=P["accent"], va="center", fontsize=9)
    _pct_axis(ax)
    ax.set_ylabel("On-time delivery")
    ax.set_title("SUP04 and SUP09 collapsed from ~65% to ~15% on-time in 2025; the other 10 are steady")
    return fig


def supplier_matrix_chart(am: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    colors = np.where(am.OTD < 0.60, P["accent"], P["primary"])
    ax.scatter(am.Spend / 1e6, am.OTD, s=140, color=colors)
    for _, r in am.iterrows():
        ax.annotate(r.Supplier_ID, (r.Spend / 1e6, r.OTD), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0.60, color=P["muted"], ls="--")
    _pct_axis(ax)
    ax.set_xlabel("PO spend, 3 years (EGP M)")
    ax.set_ylabel("On-time delivery")
    ax.set_title("Supplier action matrix: two unreliable suppliers carry 16% of spend")
    return fig


# --------------------------------------------------------------- Case 10 / opp
def value_at_stake_chart(v: pd.DataFrame) -> plt.Figure:
    d = v.sort_values("Base")
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    y = np.arange(len(d))
    ax.barh(y, d.Base, color=P["primary"], label="Base")
    ax.errorbar(d.Base, y, xerr=[d.Base - d.Low, d.High - d.Base], fmt="none", ecolor=P["accent"], capsize=4, label="Low-High range")
    ax.set_yticks(y, [a[:78] for a in d.Action])
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(millions))
    ax.set_xlabel("Annual gross-profit impact (EGP)")
    ax.set_title("Value at stake per year: pricing discipline is worth more than all other levers combined")
    ax.legend(loc="lower right")
    return fig


def chain_corr_heatmap(corr: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("Month-over-month changes: links between functions are weak")
    fig.colorbar(im, ax=ax, shrink=0.7)
    return fig


def discount_vs_quantity(w: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    t = w.groupby(["Quantity", "Discount_Pct"]).size().unstack(fill_value=0)
    t = t.div(t.sum(axis=1), axis=0)
    im = axes[0].imshow(t.values, cmap="Blues", aspect="auto", vmin=0, vmax=0.4)
    axes[0].grid(False)
    axes[0].set_xticks(range(len(t.columns)), [f"{c:.0%}" for c in t.columns])
    axes[0].set_yticks(range(len(t.index)), t.index)
    axes[0].set_xlabel("Discount granted")
    axes[0].set_ylabel("Order quantity (units)")
    axes[0].set_title("Discount mix is identical at every order size")
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            axes[0].text(j, i, f"{t.values[i, j]:.0%}", ha="center", va="center", fontsize=8)
    g = w.groupby("Discount_Pct").agg(q=("Quantity", "mean"), gm=("Gross_Margin_Pct", "mean"))
    axes[1].bar([f"{d:.0%}" for d in g.index], g.q, color=P["muted"], label="Avg units per order")
    axes[1].set_ylim(0, 40)
    axes[1].set_ylabel("Avg units per order")
    ax2 = axes[1].twinx()
    ax2.plot([f"{d:.0%}" for d in g.index], g.gm, color=P["accent"], marker="o", lw=2.4, label="Gross margin")
    ax2.set_ylim(0, 0.25)
    _pct_axis(ax2)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    axes[1].set_xlabel("Discount granted")
    axes[1].set_title("Deeper discounts: same order size, much lower margin")
    fig.suptitle("Wholesale discounts are assigned with no link to volume", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def policy_uplift_chart(pt: pd.DataFrame) -> plt.Figure:
    d = pt.pivot(index="Policy", columns="Volume_Loss_On_Repriced", values="GP_Uplift_per_Year").loc[["Flat cap 30%", "Volume-tiered cap", "Margin floor 15%", "Tiered cap + margin floor"]]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(d))
    width = 0.2
    cols = [P["good"], "#7FC8BE", P["warn"], P["accent"]]
    for k, (loss, color) in enumerate(zip(d.columns, cols)):
        ax.bar(x + (k - 1.5) * width, d[loss], width=width, color=color, label=f"{loss:.0%} volume loss")
    ax.set_xticks(x, d.index)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(millions))
    ax.set_ylabel("Added gross profit per year (EGP)")
    ax.set_title("Every policy adds profit even if re-priced customers cut volume by 20%")
    ax.legend(title="Volume lost on re-priced lines", fontsize=8)
    return fig


def diagnostic_importance_chart(diag: dict) -> plt.Figure:
    imp = diag["importance"].sort_values("R2_Drop").tail(8)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    axes[0].barh(["Baseline (mean)", "Gradient boosting"], [diag["baseline_r2"], diag["model_r2"]], color=[P["muted"], P["primary"]])
    axes[0].set_xlim(-0.05, 1)
    axes[0].set_title("Held-out R2 predicting discount")
    axes[1].barh(imp.Feature, imp.R2_Drop, color=P["primary"])
    axes[1].set_title("Permutation importance (R2 drop)")
    fig.suptitle(f"Nothing known at quote time predicts the discount (R2 = {diag['model_r2']:.3f})", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig
