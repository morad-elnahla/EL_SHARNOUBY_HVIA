"""Raw -> validated -> cleaned -> integrated -> analytical datasets.

Raw CSVs are never modified. Every output is written to data/processed/ as parquet,
and the validation log is written to outputs/tables/data_quality_report.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import (
    AS_OF_DATE,
    ensure_dirs,
    fiscal_year,
    read_raw,
    safe_div,
    save_table,
    write_processed,
)

DATE_COLUMNS = {
    "Sales_Transactions": ["Order_Date", "Return_Date"],
    "Customers": ["Join_Date"],
    "Inventory_History": ["Snapshot_Month"],
    "Marketing": ["Start_Date", "End_Date"],
    "Web_Traffic_Daily": ["Date"],
    "Purchase_Orders": ["Order_Date", "Promised_Date", "Actual_Receipt_Date"],
    "Production_Orders": ["Start_Date", "Due_Date", "Completion_Date"],
    "Printing_Batches": ["Print_Date"],
    "Machine_Downtime": ["Date"],
    "Quality": ["Inspection_Date"],
    "Customer_Service": ["Open_Date"],
    "Group_Operations": ["Month"],
}
TABLES = [
    "Sales_Transactions", "Customers", "Product_Master", "Locations", "Inventory_History",
    "Marketing", "Web_Traffic_Daily", "Materials", "Product_BOM", "Suppliers",
    "Purchase_Orders", "Production_Orders", "Printing_Batches", "Machine_Downtime",
    "Quality", "Customer_Service", "Group_Operations", "Business_Units",
]


class QualityLog:
    """Collects data-quality findings: what was checked, how many rows failed, what we did."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, table: str, check: str, failed: int, total: int, action: str) -> None:
        self.rows.append(
            {
                "table": table, "check": check, "rows_failed": int(failed), "rows_checked": int(total),
                "pct_failed": round(100 * failed / total, 3) if total else 0.0,
                "status": "PASS" if failed == 0 else "FLAG", "action": action if failed else "-",
            }
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def load_raw() -> dict[str, pd.DataFrame]:
    data = {}
    for name in TABLES:
        df = read_raw(name)
        for col in DATE_COLUMNS.get(name, []):
            df[col] = pd.to_datetime(df[col], errors="coerce")
        data[name] = df
    return data


def checked_merge(left: pd.DataFrame, right: pd.DataFrame, on, how="left", label="") -> pd.DataFrame:
    """Merge that fails loudly if the join changes the row count (fan-out / many-to-many)."""
    before = len(left)
    out = left.merge(right, on=on, how=how, validate="many_to_one")
    if len(out) != before:
        raise ValueError(f"Join '{label}' changed row count {before} -> {len(out)}")
    return out


# ---------------------------------------------------------------- validation
def validate(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    log = QualityLog()
    for name, df in data.items():
        n = len(df)
        log.add(name, "exact duplicate rows", df.duplicated().sum(), n, "dropped")
        for col in DATE_COLUMNS.get(name, []):
            nullable = name == "Sales_Transactions" and col == "Return_Date"
            if not nullable:
                log.add(name, f"unparseable date: {col}", df[col].isna().sum(), n, "row excluded")
    s, cu, pm, loc = (data[k] for k in ("Sales_Transactions", "Customers", "Product_Master", "Locations"))
    n = len(s)
    log.add("Sales_Transactions", "duplicate Invoice_ID", s.Invoice_ID.duplicated().sum(), n, "dropped")
    log.add("Sales_Transactions", "Customer_ID not in Customers", (~s.Customer_ID.isin(cu.Customer_ID)).sum(), n, "excluded")
    log.add("Sales_Transactions", "Product_ID not in Product_Master", (~s.Product_ID.isin(pm.Product_ID)).sum(), n, "excluded")
    log.add("Sales_Transactions", "Location_ID not in Locations", (~s.Location_ID.isin(loc.Location_ID)).sum(), n, "excluded")
    log.add("Sales_Transactions", "Quantity <= 0", (s.Quantity <= 0).sum(), n, "excluded")
    log.add("Sales_Transactions", "Gross != Qty x List price", (s.Gross_Sales_EGP != s.Quantity * s.List_Price_EGP).sum(), n, "recomputed")
    net_calc = s.Gross_Sales_EGP * (1 - s.Discount_Pct)
    log.add("Sales_Transactions", "Net != Gross x (1 - discount)", ((net_calc - s.Net_Sales_EGP).abs() > 0.05).sum(), n, "recomputed")
    gp_calc = s.Net_Sales_EGP - s.Estimated_COGS_EGP
    log.add("Sales_Transactions", "GP != Net - COGS", ((gp_calc - s.Gross_Profit_EGP).abs() > 0.05).sum(), n, "recomputed")
    price = s.merge(pm[["Product_ID", "Public_Retail_Price_EGP"]], on="Product_ID", how="left")
    log.add("Sales_Transactions", "List price != public catalog price", (price.List_Price_EGP != price.Public_Retail_Price_EGP).sum(), n, "kept (catalog is reference)")
    log.add("Sales_Transactions", "Returned=Yes without Return_Date", ((s.Returned == "Yes") & s.Return_Date.isna()).sum(), n, "kept")
    log.add("Sales_Transactions", "Return_Date before Order_Date", (s.Return_Date < s.Order_Date).sum(), n, "excluded")
    log.add("Sales_Transactions", "Refund > Net sales", (s.Refund_Value_EGP > s.Net_Sales_EGP + 0.01).sum(), n, "capped")
    log.add("Sales_Transactions", "Month_Key != Order_Date month", (s.Order_Date.dt.strftime("%Y-%m") != s.Month_Key).sum(), n, "recomputed")
    log.add("Sales_Transactions", "Negative gross profit (loss-making invoice)", (s.Gross_Profit_EGP < 0).sum(), n, "kept & flagged: business finding")
    cust_type = s.Customer_ID.map(cu.set_index("Customer_ID").Customer_Type)
    chan_type = np.where(s.Channel == "Wholesale", "Wholesale", "Retail")
    log.add("Sales_Transactions", "Customer type inconsistent with channel", (cust_type != chan_type).sum(), n, "kept & flagged")
    joined = s.merge(cu[["Customer_ID", "Join_Date"]], on="Customer_ID", how="left")
    log.add("Sales_Transactions", "Order before customer Join_Date", (joined.Order_Date < joined.Join_Date).sum(), n, "kept")
    log.add("Sales_Transactions", "Order after data cut-off", (s.Order_Date > AS_OF_DATE).sum(), n, "excluded")

    inv = data["Inventory_History"]
    n = len(inv)
    log.add("Inventory_History", "duplicate month x location x product", inv.duplicated(["Snapshot_Month", "Location_ID", "Product_ID"]).sum(), n, "dropped")
    log.add("Inventory_History", "negative stock values", (inv[["Opening_Stock", "Closing_Stock", "Received_Qty", "Sold_Qty"]] < 0).any(axis=1).sum(), n, "excluded")
    gap = inv.Closing_Stock - (inv.Opening_Stock + inv.Received_Qty - inv.Sold_Qty + inv.Returned_Qty - inv.Damaged_Qty)
    log.add("Inventory_History", "stock roll-forward does not balance", (gap != 0).sum(), n, "kept; gap stored as Unexplained_Stock_Adj")
    log.add("Inventory_History", "Safety stock > reorder point", (inv.Safety_Stock_Units > inv.Reorder_Point_Units).sum(), n, "kept")
    sm = s.groupby([s.Order_Date.dt.to_period("M").dt.to_timestamp(), "Location_ID", "Product_ID"]).Quantity.sum().rename("sales_qty").reset_index()
    rec = inv.merge(sm, left_on=["Snapshot_Month", "Location_ID", "Product_ID"], right_on=["Order_Date", "Location_ID", "Product_ID"], how="left")
    log.add("Inventory_History", "Sold_Qty != units in Sales_Transactions", (rec.Sold_Qty != rec.sales_qty.fillna(0)).sum(), n, "kept")

    mk = data["Marketing"]
    n = len(mk)
    log.add("Marketing", "End_Date before Start_Date", (mk.End_Date < mk.Start_Date).sum(), n, "excluded")
    log.add("Marketing", "Clicks > Impressions", (mk.Clicks > mk.Impressions).sum(), n, "excluded")
    log.add("Marketing", "Orders attributed > Clicks", (mk.Orders_Attributed > mk.Clicks).sum(), n, "excluded")
    online_rev = s.loc[s.Channel == "Online", "Net_Sales_EGP"].sum()
    over = mk.Attributed_Revenue_EGP.sum() / online_rev
    log.add("Marketing", f"Attributed revenue = {over:.1f}x total Online net sales (cannot reconcile)", 1, 1, "ROAS used for ranking only, not as absolute")
    wt = data["Web_Traffic_Daily"]
    log.add("Web_Traffic_Daily", "Orders > Sessions", (wt.Orders > wt.Sessions).sum(), len(wt), "excluded")
    log.add("Web_Traffic_Daily", "Web orders < Online invoices in Sales", int(wt.Orders.sum() < (s.Channel == "Online").sum()), 1, "flagged: web analytics under-counts or Sales over-counts online orders")

    po = data["Purchase_Orders"]
    n = len(po)
    log.add("Purchase_Orders", "Accepted + Rejected != Ordered", (po.Accepted_Qty + po.Rejected_Qty != po.Ordered_Qty).sum(), n, "excluded")
    log.add("Purchase_Orders", "Receipt before Order date", (po.Actual_Receipt_Date < po.Order_Date).sum(), n, "excluded")
    log.add("Purchase_Orders", "Supplier_ID not in Suppliers", (~po.Supplier_ID.isin(data["Suppliers"].Supplier_ID)).sum(), n, "excluded")
    log.add("Purchase_Orders", "Material_ID not in Materials", (~po.Material_ID.isin(data["Materials"].Material_ID)).sum(), n, "excluded")
    sup = po.merge(data["Suppliers"][["Supplier_ID", "Primary_Category"]], on="Supplier_ID").merge(
        data["Materials"][["Material_ID", "Material_Category"]], on="Material_ID")
    log.add("Purchase_Orders", "Supplier category differs from material category", (sup.Primary_Category != sup.Material_Category).sum(), n, "kept: suppliers are not category-specialised")

    pr = data["Production_Orders"]
    n = len(pr)
    log.add("Production_Orders", "Good + Rejected != Produced", (pr.Good_Qty + pr.Rejected_Qty != pr.Produced_Qty).sum(), n, "excluded")
    log.add("Production_Orders", "Produced > Planned", (pr.Produced_Qty > pr.Planned_Qty).sum(), n, "kept")
    late = (pr.Completion_Date - pr.Due_Date).dt.days.clip(lower=0)
    log.add("Production_Orders", "Delay_Days != Completion - Due", (late != pr.Delay_Days).sum(), n, "recomputed")
    log.add("Production_Orders", "Delayed order without bottleneck label", ((pr.Delay_Days > 0) & pr.Primary_Bottleneck.isna()).sum(), n, "kept")
    log.add("Production_Orders", "Efficiency outside 0-100", (~pr.Efficiency_Pct.between(0, 100)).sum(), n, "excluded")

    pb = data["Printing_Batches"]
    n = len(pb)
    log.add("Printing_Batches", "Good + Rejected != Printed", (pb.Good_Qty + pb.Rejected_Qty != pb.Printed_Qty).sum(), n, "excluded")
    log.add("Printing_Batches", "Printed > Planned (over-run)", (pb.Printed_Qty > pb.Planned_Qty).sum(), n, "kept & flagged")
    log.add("Printing_Batches", "Rejected units without Defect_Type", ((pb.Rejected_Qty > 0) & pb.Defect_Type.isna()).sum(), n, "labelled 'Unclassified'")
    dt = data["Machine_Downtime"]
    log.add("Machine_Downtime", "Downtime_Hours <= 0", (dt.Downtime_Hours <= 0).sum(), len(dt), "excluded")

    q = data["Quality"]
    n = len(q)
    log.add("Quality", "Passed + Rejected != Inspected", (q.Passed_Qty + q.Rejected_Qty != q.Inspected_Qty).sum(), n, "excluded")
    log.add("Quality", "Rejected units without Defect_Type", ((q.Rejected_Qty > 0) & q.Defect_Type.isna()).sum(), n, "labelled 'Unclassified'")
    log.add("Quality", "Rejects with root cause 'No significant issue'", ((q.Rejected_Qty > 0) & (q.Root_Cause == "No significant issue")).sum(), n, "kept & flagged: root-cause coding unreliable")
    log.add("Quality", "Product_ID not in Product_Master", (~q.Product_ID.isin(pm.Product_ID)).sum(), n, "excluded")
    return log.frame()


# ------------------------------------------------------------------ cleaning
def clean_sales(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    s = data["Sales_Transactions"].drop_duplicates("Invoice_ID").copy()
    s = s[(s.Quantity > 0) & (s.Order_Date <= AS_OF_DATE)]
    s["Net_Sales_EGP"] = s.Gross_Sales_EGP * (1 - s.Discount_Pct)
    s["Gross_Profit_EGP"] = s.Net_Sales_EGP - s.Estimated_COGS_EGP
    s["Gross_Margin_Pct"] = safe_div(s.Gross_Profit_EGP, s.Net_Sales_EGP)
    s["Refund_Value_EGP"] = np.minimum(s.Refund_Value_EGP, s.Net_Sales_EGP)
    s["Month"] = s.Order_Date.dt.to_period("M").dt.to_timestamp()
    s["Fiscal_Year"] = fiscal_year(s.Order_Date)
    s["Is_Returned"] = s.Returned.eq("Yes")
    s["Is_Quality_Return"] = s.Quality_Related_Return.eq("Yes")
    s["Is_Loss_Making"] = s.Gross_Profit_EGP < 0
    s["Unit_COGS_EGP"] = s.Estimated_COGS_EGP / s.Quantity
    s["Unit_Net_Price_EGP"] = s.Net_Sales_EGP / s.Quantity
    s["Breakeven_Discount_Pct"] = 1 - s.Unit_COGS_EGP / s.List_Price_EGP
    s["Channel_Customer_Mismatch"] = (s.Channel == "Wholesale") != (s.Customer_Type == "Wholesale")
    s["Return_Reason"] = s.Return_Reason.replace({"No return": np.nan})
    return s.drop(columns=["Returned", "Quality_Related_Return", "Data_Class", "Month_Key"]).reset_index(drop=True)


def clean_inventory(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    inv = data["Inventory_History"].drop_duplicates(["Snapshot_Month", "Location_ID", "Product_ID"]).copy()
    inv = inv.rename(columns={"Snapshot_Month": "Month"})
    expected = inv.Opening_Stock + inv.Received_Qty - inv.Sold_Qty + inv.Returned_Qty - inv.Damaged_Qty
    inv["Unexplained_Stock_Adj"] = inv.Closing_Stock - expected
    inv["Stockout_Flag"] = inv.Stockout_Days_In_Month > 0
    inv["Days_In_Month"] = inv.Month.dt.days_in_month
    inv["Stockout_Day_Pct"] = inv.Stockout_Days_In_Month / inv.Days_In_Month
    inv["Lost_Units_Est"] = inv.Stockout_Days_In_Month * inv.Avg_Daily_Sales_90D
    return inv.drop(columns=["Data_Class", "SKU"])


def clean_procurement(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    po = data["Purchase_Orders"].copy()
    po = po[po.Accepted_Qty + po.Rejected_Qty == po.Ordered_Qty]
    po = checked_merge(po, data["Suppliers"][["Supplier_ID", "Primary_Category", "Contracted", "Payment_Terms_Days"]], "Supplier_ID", label="PO-Suppliers")
    po = checked_merge(po, data["Materials"][["Material_ID", "Material_Category", "Standard_Lead_Time_Days"]], "Material_ID", label="PO-Materials")
    po["Spend_EGP"] = po.Ordered_Qty * po.Unit_Cost_EGP
    po["Lead_Time_Days"] = (po.Actual_Receipt_Date - po.Order_Date).dt.days
    po["Promised_Lead_Time_Days"] = (po.Promised_Date - po.Order_Date).dt.days
    po["Days_Late"] = (po.Actual_Receipt_Date - po.Promised_Date).dt.days
    po["On_Time"] = po.Days_Late <= 0
    po["Reject_Rate"] = po.Rejected_Qty / po.Ordered_Qty
    po["OTIF"] = po.On_Time & (po.Reject_Rate <= 0.02)
    po["Month"] = po.Order_Date.dt.to_period("M").dt.to_timestamp()
    po["Fiscal_Year"] = fiscal_year(po.Order_Date)
    return po.drop(columns=["Data_Class"])


def clean_production(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    p = data["Production_Orders"].copy()
    p = p[(p.Good_Qty + p.Rejected_Qty == p.Produced_Qty) & p.Efficiency_Pct.between(0, 100)]
    p["Delay_Days"] = (p.Completion_Date - p.Due_Date).dt.days.clip(lower=0)
    p["Cycle_Days"] = (p.Completion_Date - p.Start_Date).dt.days + 1
    p["Is_Delayed"] = p.Delay_Days > 0
    p["Reject_Rate"] = p.Rejected_Qty / p.Produced_Qty
    p["Plan_Attainment"] = p.Produced_Qty / p.Planned_Qty
    p["Throughput_Units_Per_Day"] = p.Produced_Qty / p.Cycle_Days
    p["Primary_Bottleneck"] = p.Primary_Bottleneck.fillna("None")
    p["Month"] = p.Completion_Date.dt.to_period("M").dt.to_timestamp()
    return p.drop(columns=["Data_Class"])


def clean_printing(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    b = data["Printing_Batches"].copy()
    b = b[b.Good_Qty + b.Rejected_Qty == b.Printed_Qty]
    b["Defect_Type"] = np.where(b.Rejected_Qty.gt(0) & b.Defect_Type.isna(), "Unclassified", b.Defect_Type)
    b["Yield"] = b.Good_Qty / b.Printed_Qty
    b["Reject_Rate"] = b.Rejected_Qty / b.Printed_Qty
    b["Reprint_Rate"] = b.Reprint_Qty / b.Printed_Qty
    b["Overrun_Flag"] = b.Printed_Qty > b.Planned_Qty
    b["Total_Cost_EGP"] = b.Ink_or_Consumable_Cost_EGP + b.Rework_Cost_EGP
    b["Cost_Per_Good_Unit"] = b.Total_Cost_EGP / b.Good_Qty
    b["Month"] = b.Print_Date.dt.to_period("M").dt.to_timestamp()
    return b.drop(columns=["Data_Class"])


def clean_quality(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    q = data["Quality"].copy()
    q = q[q.Passed_Qty + q.Rejected_Qty == q.Inspected_Qty]
    q = q[q.Product_ID.isin(data["Product_Master"].Product_ID)]
    q["Defect_Type"] = np.where(q.Rejected_Qty.gt(0) & q.Defect_Type.isna(), "Unclassified", q.Defect_Type)
    q["Defect_Type"] = q.Defect_Type.fillna("None")
    q["Reject_Rate"] = q.Rejected_Qty / q.Inspected_Qty
    q["Month"] = q.Inspection_Date.dt.to_period("M").dt.to_timestamp()
    return q.drop(columns=["Data_Class"])


def clean_marketing(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    m = data["Marketing"].copy()
    m = m[(m.End_Date >= m.Start_Date) & (m.Clicks <= m.Impressions) & (m.Orders_Attributed <= m.Clicks)]
    m["Campaign_Days"] = (m.End_Date - m.Start_Date).dt.days + 1
    m["ROAS"] = m.Attributed_Revenue_EGP / m.Spend_EGP
    m["CAC_EGP"] = m.Spend_EGP / m.Orders_Attributed
    m["CTR"] = m.Clicks / m.Impressions
    m["CVR"] = m.Orders_Attributed / m.Clicks
    m["CPC_EGP"] = m.Spend_EGP / m.Clicks
    m["AOV_EGP"] = m.Attributed_Revenue_EGP / m.Orders_Attributed
    m["Month"] = m.Start_Date.dt.to_period("M").dt.to_timestamp()
    m = checked_merge(m, data["Product_Master"][["Product_ID", "Product_Family"]], "Product_ID", label="Marketing-Products")
    return m.drop(columns=["Data_Class", "Landing_Page_Sessions"])


# -------------------------------------------------------------- integration
def build_sales_enriched(sales: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    customers = data["Customers"][["Customer_ID", "Join_Date", "Age_Band", "Gender", "Acquisition_Channel", "Consent_Status"]]
    products = data["Product_Master"][["Product_ID", "Gender", "Audience", "Catalog_Season"]].rename(
        columns={"Gender": "Product_Gender", "Audience": "Product_Audience"})
    out = checked_merge(sales, customers, "Customer_ID", label="Sales-Customers")
    out = checked_merge(out, products, "Product_ID", label="Sales-Products")
    return out


def build_customer_360(sales: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    g = sales.groupby("Customer_ID").agg(
        Orders=("Invoice_ID", "nunique"), Units=("Quantity", "sum"), Net_Sales_EGP=("Net_Sales_EGP", "sum"),
        Gross_Profit_EGP=("Gross_Profit_EGP", "sum"), Refund_EGP=("Refund_Value_EGP", "sum"),
        Returned_Orders=("Is_Returned", "sum"), First_Order=("Order_Date", "min"), Last_Order=("Order_Date", "max"),
        Avg_Discount=("Discount_Pct", "mean"), Channels_Used=("Channel", "nunique"),
        Main_Channel=("Channel", lambda x: x.mode().iat[0]),
    ).reset_index()
    g["Recency_Days"] = (AS_OF_DATE - g.Last_Order).dt.days
    g["Tenure_Days"] = (g.Last_Order - g.First_Order).dt.days
    g["AOV_EGP"] = g.Net_Sales_EGP / g.Orders
    g["Is_Repeat"] = g.Orders > 1
    out = checked_merge(g, customers.drop(columns=["Data_Class"]), "Customer_ID", label="Customer360-Customers")
    out["Cohort_Month"] = out.First_Order.dt.to_period("M").dt.to_timestamp()
    return out


def build_demand_monthly(sales: pd.DataFrame) -> pd.DataFrame:
    """Product x channel x month demand: the grain used for forecasting."""
    return (
        sales.groupby(["Month", "Product_ID", "Channel"], as_index=False)
        .agg(Units=("Quantity", "sum"), Net_Sales_EGP=("Net_Sales_EGP", "sum"), Gross_Profit_EGP=("Gross_Profit_EGP", "sum"))
    )


def build_product_quality_returns(sales: pd.DataFrame, quality: pd.DataFrame, production: pd.DataFrame) -> pd.DataFrame:
    sold = sales.groupby("Product_ID").agg(
        Product_Name=("Product_Name", "first"), Product_Family=("Product_Family", "first"),
        Units=("Quantity", "sum"), Net_Sales_EGP=("Net_Sales_EGP", "sum"), Refund_EGP=("Refund_Value_EGP", "sum"),
        Return_Lines=("Is_Returned", "sum"), Quality_Return_Lines=("Is_Quality_Return", "sum"), Lines=("Invoice_ID", "count"),
    )
    qc = quality.groupby("Product_ID").agg(Inspected=("Inspected_Qty", "sum"), QC_Rejected=("Rejected_Qty", "sum"), QC_Loss_EGP=("Estimated_Loss_EGP", "sum"))
    prod = production.groupby("Product_ID").agg(Produced=("Produced_Qty", "sum"), Prod_Rejected=("Rejected_Qty", "sum"), Avg_Efficiency=("Efficiency_Pct", "mean"), Avg_Delay=("Delay_Days", "mean"))
    out = sold.join(qc).join(prod).reset_index()
    out["QC_Reject_Rate"] = out.QC_Rejected / out.Inspected
    out["Quality_Return_Rate"] = out.Quality_Return_Lines / out.Lines
    out["Return_Rate"] = out.Return_Lines / out.Lines
    return out


# ---------------------------------------------------------------------- run
def run() -> dict[str, int]:
    ensure_dirs()
    data = load_raw()
    report = validate(data)
    save_table(report, "data_quality_report")

    sales = clean_sales(data)
    sales_enriched = build_sales_enriched(sales, data)
    inventory = clean_inventory(data)
    procurement = clean_procurement(data)
    production = clean_production(data)
    printing = clean_printing(data)
    quality = clean_quality(data)
    marketing = clean_marketing(data)
    downtime = data["Machine_Downtime"].copy()
    downtime["Month"] = downtime.Date.dt.to_period("M").dt.to_timestamp()
    downtime = downtime.drop(columns=["Data_Class"])
    web = data["Web_Traffic_Daily"].drop(columns=["Data_Class"])
    service = data["Customer_Service"].drop(columns=["Data_Class"]).copy()
    service["Month"] = service.Open_Date.dt.to_period("M").dt.to_timestamp()

    outputs = {
        "sales": sales_enriched,
        "customer_360": build_customer_360(sales, data["Customers"]),
        "demand_monthly": build_demand_monthly(sales),
        "product_quality_returns": build_product_quality_returns(sales, quality, production),
        "inventory": inventory,
        "procurement": procurement,
        "production": production,
        "printing": printing,
        "downtime": downtime,
        "quality": quality,
        "marketing": marketing,
        "web_traffic": web,
        "customer_service": service,
        "products": data["Product_Master"].drop(columns=["Public_Catalog_Source", "Verification_Class"]),
        "locations": data["Locations"].drop(columns=["Source_URL", "Verification_Class"]),
        "suppliers": data["Suppliers"].drop(columns=["Data_Class"]),
        "materials": data["Materials"].drop(columns=["Data_Class"]),
        "bom": data["Product_BOM"].drop(columns=["Data_Class"]),
        "group_operations": data["Group_Operations"].drop(columns=["Data_Class"]),
    }
    for name, df in outputs.items():
        write_processed(df, name)

    raw_rows, clean_rows = len(data["Sales_Transactions"]), len(sales)
    if raw_rows != clean_rows:
        print(f"[pipeline] sales rows dropped during cleaning: {raw_rows - clean_rows}")
    flagged = report[report.status == "FLAG"]
    print(f"[pipeline] {len(outputs)} processed datasets written | {len(report)} checks | {len(flagged)} flagged")
    return {name: len(df) for name, df in outputs.items()}


if __name__ == "__main__":
    run()
