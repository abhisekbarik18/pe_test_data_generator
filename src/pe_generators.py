from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.cell import WriteOnlyCell

from . import generators as g


GL_COLUMNS = [
    "S.No", "Reference Number", "Batch Number", "Vendor", "Affiliate",
    "Deal Name", "Position", "Bank Account", "Investor Name", "Amount",
    "Debit", "Credit", "Location ID", "Fund Name",
]

SOI_COLUMNS = [
    "S.No", "Fund Name", "Fund ID", "Deal Name", "Investment ID", "Position",
    "Security Type", "Investor Name", "Investment Date", "Acquisition Date",
    "Currency", "Beginning Cost", "Additions", "Disposals", "Ending Cost",
    "Beginning Fair Value", "Unrealized Gain/Loss", "Realized Gain/Loss",
    "FX Gain/Loss", "Ending Fair Value", "Ownership %", "Shares/Units",
    "Valuation Method", "Investment Status", "Reporting Date",
]

ROLLFORWARD_COLUMNS = [
    "S.No", "Fund Name", "Fund ID", "Deal Name", "Investment ID", "Position",
    "Investor Name", "Reporting Period", "Beginning Cost", "Purchases/Additions",
    "Follow-on Investment", "Disposals", "Write-offs", "Transfers/Adjustments",
    "Ending Cost", "Beginning Fair Value", "Unrealized Gain/Loss",
    "Realized Gain/Loss", "FX Gain/Loss", "Write-up", "Write-down",
    "Ending Fair Value", "Source Transaction Reference",
]

TRANSACTION_COLUMNS = [
    "Reference Number", "Batch Number", "Transaction Type", "Fund Name", "Fund ID",
    "Deal Name", "Investment ID", "Position", "Investor Name", "Reporting Period",
    "Transaction Amount", "Cost Change", "Fair Value Change", "Realized Gain/Loss",
    "Carrying Value", "Affiliate", "Vendor", "Bank Account", "Location ID",
]


def _master_names(config_dir: Path, module: str, candidates: tuple[str, ...], count: int, seed: int) -> np.ndarray:
    path = config_dir / f"{module}.yaml"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            schema = yaml.safe_load(handle) or {}
        by_name = {column.get("name", "").lower(): column for column in schema.get("columns", [])}
        column = next((by_name[name] for name in candidates if name in by_name), None)
        if column is not None and column.get("type") == "name_pool":
            params = {key: value for key, value in column.items() if key not in {"name", "type"}}
            return g.gen_name_pool(count, g.get_rng(seed), **params)
    return g.gen_name_pool(count, g.get_rng(seed), pool="company", pool_size=25000, unique=True)


def _period_dates(reporting_date: date, count: int) -> list[date]:
    return [reporting_date - timedelta(days=92 * (count - index - 1)) for index in range(count)]


def _numbered(prefix: str, start: int, count: int, width: int = 10) -> np.ndarray:
    numbers = np.arange(start, start + count, dtype=np.int64)
    return np.char.mod(f"{prefix}%0{width}d", numbers).astype(object)


def _base_transactions(
    refs: np.ndarray,
    batches: np.ndarray,
    kind: np.ndarray,
    fund_name: np.ndarray,
    fund_id: np.ndarray,
    deal_name: np.ndarray,
    investment_id: np.ndarray,
    position: np.ndarray,
    investor: np.ndarray,
    period: np.ndarray,
    amount: np.ndarray,
    cost_change: np.ndarray,
    fv_change: np.ndarray,
    realized: np.ndarray,
    carrying: np.ndarray,
    affiliate: np.ndarray,
    vendor: np.ndarray,
    bank: np.ndarray,
    location: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame({
        "Reference Number": refs, "Batch Number": batches, "Transaction Type": kind,
        "Fund Name": fund_name, "Fund ID": fund_id, "Deal Name": deal_name,
        "Investment ID": investment_id, "Position": position, "Investor Name": investor,
        "Reporting Period": period, "Transaction Amount": amount, "Cost Change": cost_change,
        "Fair Value Change": fv_change, "Realized Gain/Loss": realized,
        "Carrying Value": carrying, "Affiliate": affiliate, "Vendor": vendor,
        "Bank Account": bank, "Location ID": location,
    }, columns=TRANSACTION_COLUMNS)


def generate_linked_datasets(
    record_count: int,
    fund_count: int,
    investment_count: int,
    period_count: int,
    reporting_date: date,
    seed: int = 42,
    base_currency: str = "USD",
    config_dir: Path | None = None,
) -> dict[str, pd.DataFrame | dict]:
    """Generate one transaction model and derive its SOI, rollforward, and GL."""
    if record_count < 1 or fund_count < 1 or investment_count < 1 or period_count < 1:
        raise ValueError("Record, fund, investment, and reporting-period counts must be at least one.")
    if record_count > 1_000_000:
        raise ValueError("SOI record count cannot exceed 1,000,000.")
    if period_count > 20:
        raise ValueError("Reporting periods cannot exceed 20.")

    rng = g.get_rng(seed)
    config_dir = config_dir or Path(__file__).resolve().parent.parent / "config"
    deal_count = investment_count
    ref_investors = _master_names(config_dir, "investors", ("investor_name", "investor name"), min(record_count, 5000), seed + 1)
    ref_vendors = _master_names(config_dir, "vendors", ("vendor_name", "vendor name"), 500, seed + 2)
    ref_affiliates = _master_names(config_dir, "affiliates", ("affiliate_name", "affiliate name"), max(fund_count, 1), seed + 3)
    fund_names = g.gen_name_pool(fund_count, rng, pool="company", pool_size=25000, unique=True)
    deal_names = g.gen_name_pool(deal_count, rng, pool="company", pool_size=25000, unique=True)
    fund_ids = _numbered("FUND-", 1, fund_count, 5)
    deal_funds = np.arange(deal_count) % fund_count
    investment_ids = _numbered("INV-", 1, record_count, 9)
    deal_indexes = rng.integers(0, deal_count, size=record_count)
    fund_indexes = deal_funds[deal_indexes]
    investor_indexes = rng.integers(0, len(ref_investors), size=record_count)
    affiliate_indexes = rng.integers(0, len(ref_affiliates), size=record_count)
    vendor_indexes = rng.integers(0, len(ref_vendors), size=record_count)
    positions = rng.choice(
        np.array(["Common Equity", "Preferred Equity", "Senior Debt", "Mezzanine Debt", "Fund Interest"], dtype=object),
        size=record_count,
        p=[0.42, 0.18, 0.2, 0.12, 0.08],
    )
    security_types = np.where(np.isin(positions, ["Senior Debt", "Mezzanine Debt"]), "Debt", "Equity")
    initial_cost = np.round(rng.uniform(250_000, 25_000_000, size=record_count), 2)
    ownership = np.round(rng.uniform(0.005, 0.35, size=record_count), 6)
    prices = rng.uniform(5, 250, size=record_count)
    shares = np.round(initial_cost / prices, 4)
    reporting_day = np.datetime64(reporting_date, "D")
    investment_dates = (reporting_day - rng.integers(365, 365 * 12, size=record_count).astype("timedelta64[D]")).astype(str)
    acquisition_dates = (investment_dates.astype("datetime64[D]") - rng.integers(0, 90, size=record_count).astype("timedelta64[D]")).astype(str)
    periods = _period_dates(reporting_date, period_count)

    running_cost = np.zeros(record_count, dtype=float)
    running_fv = np.zeros(record_count, dtype=float)
    rf_chunks: list[pd.DataFrame] = []
    tx_chunks: list[pd.DataFrame] = []
    reference_number = 1

    for period_index, period_end in enumerate(periods):
        beginning_cost = running_cost.copy()
        beginning_fv = running_fv.copy()
        count = record_count
        kind = np.empty(count, dtype=object)
        purchase = np.zeros(count)
        follow_on = np.zeros(count)
        disposals = np.zeros(count)
        writeoffs = np.zeros(count)
        adjustment = np.zeros(count)
        unrealized = np.zeros(count)
        realized = np.zeros(count)
        fx = np.zeros(count)
        write_up = np.zeros(count)
        write_down = np.zeros(count)
        carrying_value = np.zeros(count)
        transaction_amount = np.zeros(count)
        fair_value_change = np.zeros(count)

        if period_index == 0:
            kind[:] = "Investment Purchase"
            purchase[:] = initial_cost
            transaction_amount[:] = initial_cost
            fair_value_change[:] = initial_cost
        else:
            event_types = np.array([
                "Investment Purchase", "Follow-on Investment", "Investment Disposal", "Write-off",
                "Transfer/Adjustment", "Unrealized Gain", "FX Gain/Loss", "Write-up", "Write-down",
            ], dtype=object)
            event_indexes = rng.choice(len(event_types), size=count, p=[0.1, 0.16, 0.1, 0.025, 0.045, 0.36, 0.08, 0.07, 0.06])
            kind[:] = event_types[event_indexes]
            purchase_mask = kind == "Investment Purchase"
            follow_mask = kind == "Follow-on Investment"
            disposal_mask = kind == "Investment Disposal"
            writeoff_mask = kind == "Write-off"
            adjust_mask = kind == "Transfer/Adjustment"
            unrealized_mask = kind == "Unrealized Gain"
            fx_mask = kind == "FX Gain/Loss"
            writeup_mask = kind == "Write-up"
            writedown_mask = kind == "Write-down"

            purchase[purchase_mask] = np.round(np.maximum(1000, beginning_cost[purchase_mask] * rng.uniform(0.03, 0.16, purchase_mask.sum())), 2)
            follow_on[follow_mask] = np.round(np.maximum(1000, beginning_cost[follow_mask] * rng.uniform(0.03, 0.2, follow_mask.sum())), 2)
            disposal_fraction = rng.uniform(0.05, 0.3, disposal_mask.sum())
            disposals[disposal_mask] = np.round(beginning_cost[disposal_mask] * disposal_fraction, 2)
            carrying_value[disposal_mask] = np.round(beginning_fv[disposal_mask] * disposal_fraction, 2)
            realized[disposal_mask] = np.round(carrying_value[disposal_mask] * rng.uniform(-0.2, 0.25, disposal_mask.sum()), 2)
            writeoff_fraction = rng.uniform(0.02, 0.1, writeoff_mask.sum())
            writeoffs[writeoff_mask] = np.round(beginning_cost[writeoff_mask] * writeoff_fraction, 2)
            carrying_value[writeoff_mask] = np.round(beginning_fv[writeoff_mask] * writeoff_fraction, 2)
            adjustment[adjust_mask] = np.round(beginning_cost[adjust_mask] * rng.uniform(-0.04, 0.04, adjust_mask.sum()), 2)
            adjustment[adjust_mask] = np.maximum(adjustment[adjust_mask], -beginning_cost[adjust_mask] * 0.2)
            unrealized[unrealized_mask] = np.round(beginning_fv[unrealized_mask] * rng.uniform(-0.1, 0.12, unrealized_mask.sum()), 2)
            fx[fx_mask] = np.round(beginning_fv[fx_mask] * rng.uniform(-0.035, 0.035, fx_mask.sum()), 2)
            write_up[writeup_mask] = np.round(beginning_fv[writeup_mask] * rng.uniform(0.005, 0.05, writeup_mask.sum()), 2)
            write_down[writedown_mask] = np.round(beginning_fv[writedown_mask] * rng.uniform(0.005, 0.05, writedown_mask.sum()), 2)
            transaction_amount[purchase_mask] = purchase[purchase_mask]
            transaction_amount[follow_mask] = follow_on[follow_mask]
            transaction_amount[disposal_mask] = np.maximum(0, carrying_value[disposal_mask] + realized[disposal_mask])
            transaction_amount[writeoff_mask] = writeoffs[writeoff_mask]
            transaction_amount[adjust_mask] = np.abs(adjustment[adjust_mask])
            transaction_amount[unrealized_mask] = np.abs(unrealized[unrealized_mask])
            transaction_amount[fx_mask] = np.abs(fx[fx_mask])
            transaction_amount[writeup_mask] = write_up[writeup_mask]
            transaction_amount[writedown_mask] = write_down[writedown_mask]
            fair_value_change = purchase + follow_on - carrying_value + adjustment + unrealized + fx + write_up - write_down

        ending_cost = np.round(beginning_cost + purchase + follow_on - disposals - writeoffs + adjustment, 2)
        ending_fv = np.round(beginning_fv + fair_value_change, 2)
        running_cost = np.maximum(ending_cost, 0)
        running_fv = np.maximum(ending_fv, 0)
        source_refs = _numbered("PE-", reference_number, count, 10)
        batch_numbers = _numbered("BATCH-", reference_number, count, 10)
        reference_number += count
        common = {
            "Fund Name": fund_names[fund_indexes], "Fund ID": fund_ids[fund_indexes],
            "Deal Name": deal_names[deal_indexes], "Investment ID": investment_ids,
            "Position": positions, "Investor Name": ref_investors[investor_indexes],
        }
        rf_chunks.append(pd.DataFrame({
            "S.No": np.arange(period_index * count + 1, (period_index + 1) * count + 1),
            **common, "Reporting Period": period_end.isoformat(), "Beginning Cost": beginning_cost,
            "Purchases/Additions": purchase, "Follow-on Investment": follow_on,
            "Disposals": disposals, "Write-offs": writeoffs, "Transfers/Adjustments": adjustment,
            "Ending Cost": ending_cost, "Beginning Fair Value": beginning_fv,
            "Unrealized Gain/Loss": unrealized, "Realized Gain/Loss": realized,
            "FX Gain/Loss": fx, "Write-up": write_up, "Write-down": -write_down,
            "Ending Fair Value": ending_fv, "Source Transaction Reference": source_refs,
        }, columns=ROLLFORWARD_COLUMNS))
        tx_chunks.append(_base_transactions(
            source_refs, batch_numbers, kind, common["Fund Name"], common["Fund ID"],
            common["Deal Name"], common["Investment ID"], common["Position"],
            common["Investor Name"], np.full(count, period_end.isoformat(), dtype=object),
            transaction_amount, purchase + follow_on - disposals - writeoffs + adjustment,
            fair_value_change, realized, carrying_value,
            ref_affiliates[affiliate_indexes], ref_vendors[vendor_indexes],
            np.full(count, "BA-000000000001", dtype=object),
            np.full(count, "LOC-0001", dtype=object),
        ))

    # Fund-level activity is part of the same transaction model and uses the same references.
    fund_tx_chunks = []
    fund_kinds = ["Management Fee", "Fund Expense", "Interest Income", "Dividend Income", "Capital Contribution", "Investor Distribution", "Other Fund Transaction"]
    fund_event_count = fund_count * period_count * len(fund_kinds)
    fund_numbers = np.arange(fund_event_count)
    fund_indexes_for_tx = np.repeat(np.arange(fund_count), period_count * len(fund_kinds))
    period_indexes_for_tx = np.tile(np.repeat(np.arange(period_count), len(fund_kinds)), fund_count)
    kinds_for_fund = np.tile(np.array(fund_kinds, dtype=object), fund_count * period_count)
    fund_amounts = np.round(rng.uniform(5_000, 2_000_000, size=fund_event_count), 2)
    fund_refs = _numbered("PE-", reference_number, fund_event_count, 10)
    fund_batches = _numbered("BATCH-", reference_number, fund_event_count, 10)
    fund_periods = np.array([periods[int(index)].isoformat() for index in period_indexes_for_tx], dtype=object)
    fund_investors = ref_investors[rng.integers(0, len(ref_investors), size=fund_event_count)]
    fund_affiliates = ref_affiliates[fund_indexes_for_tx % len(ref_affiliates)]
    fund_vendors = ref_vendors[rng.integers(0, len(ref_vendors), size=fund_event_count)]
    fund_tx_chunks.append(_base_transactions(
        fund_refs, fund_batches, kinds_for_fund, fund_names[fund_indexes_for_tx],
        fund_ids[fund_indexes_for_tx], np.full(fund_event_count, "", dtype=object),
        np.full(fund_event_count, "", dtype=object), np.full(fund_event_count, "", dtype=object),
        fund_investors, fund_periods, fund_amounts, np.zeros(fund_event_count),
        np.zeros(fund_event_count), np.zeros(fund_event_count), fund_amounts,
        fund_affiliates, fund_vendors, np.full(fund_event_count, "BA-000000000001", dtype=object),
        np.full(fund_event_count, "LOC-0001", dtype=object),
    ))
    transactions = pd.concat([*tx_chunks, *fund_tx_chunks], ignore_index=True)
    rollforward = pd.concat(rf_chunks, ignore_index=True).sort_values(["Investment ID", "Reporting Period"], kind="stable", ignore_index=True)
    rollforward["S.No"] = np.arange(1, len(rollforward) + 1)

    latest = rollforward[rollforward["Reporting Period"] == reporting_date.isoformat()].copy()
    investment_index = latest["Investment ID"].str.removeprefix("INV-").astype(int).to_numpy() - 1
    soi = pd.DataFrame({
        "S.No": np.arange(1, len(latest) + 1), "Fund Name": latest["Fund Name"],
        "Fund ID": latest["Fund ID"], "Deal Name": latest["Deal Name"],
        "Investment ID": latest["Investment ID"], "Position": latest["Position"],
        "Security Type": security_types[investment_index], "Investor Name": latest["Investor Name"],
        "Investment Date": investment_dates[investment_index],
        "Acquisition Date": acquisition_dates[investment_index], "Currency": base_currency,
        "Beginning Cost": latest["Beginning Cost"],
        "Additions": latest["Purchases/Additions"] + latest["Follow-on Investment"],
        "Disposals": latest["Disposals"], "Ending Cost": latest["Ending Cost"],
        "Beginning Fair Value": latest["Beginning Fair Value"],
        "Unrealized Gain/Loss": latest["Unrealized Gain/Loss"],
        "Realized Gain/Loss": latest["Realized Gain/Loss"], "FX Gain/Loss": latest["FX Gain/Loss"],
        "Ending Fair Value": latest["Ending Fair Value"], "Ownership %": ownership[investment_index],
        "Shares/Units": shares[investment_index],
        "Valuation Method": rng.choice(["Market Approach", "Income Approach", "Recent Transaction", "NAV"], size=len(latest)),
        "Investment Status": np.where(latest["Ending Fair Value"].to_numpy() > 0, "Active", "Realized"),
        "Reporting Date": reporting_date.isoformat(),
    }, columns=SOI_COLUMNS)
    soi = soi.sort_values(["Fund Name", "Deal Name", "Investment ID"], kind="stable", ignore_index=True)
    soi["S.No"] = np.arange(1, len(soi) + 1)
    gl = _transactions_to_gl(transactions)
    fund_summary = soi.groupby(["Fund Name", "Fund ID", "Currency"], as_index=False).agg(
        Investments=("Investment ID", "count"), Beginning_Cost=("Beginning Cost", "sum"),
        Additions=("Additions", "sum"), Disposals=("Disposals", "sum"),
        Ending_Cost=("Ending Cost", "sum"), Ending_Fair_Value=("Ending Fair Value", "sum"),
    )
    fund_summary.columns = [column.replace("_", " ") for column in fund_summary.columns]
    validation = validate_linked_data(soi, rollforward, transactions, gl, fund_names, deal_names, ref_investors, positions, ref_affiliates)
    return {
        "soi": soi, "rollforward": rollforward, "gl": gl, "transactions": transactions,
        "fund_summary": fund_summary, "validation": validation,
        "reporting_date": reporting_date.isoformat(), "base_currency": base_currency,
    }


def _transactions_to_gl(transactions: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    def add_line(rows: pd.DataFrame, amounts: np.ndarray, side: str, vendors: np.ndarray) -> None:
        mask = amounts > 0
        if not mask.any():
            return
        selected = rows.loc[mask]
        values = amounts[mask]
        frame = selected[["Reference Number", "Batch Number", "Affiliate", "Deal Name", "Position", "Bank Account", "Investor Name", "Fund Name"]].copy()
        frame.insert(0, "S.No", 0)
        frame["Vendor"] = vendors[mask]
        frame["Amount"] = values
        frame["Debit"] = values if side == "Debit" else 0.0
        frame["Credit"] = values if side == "Credit" else 0.0
        frame["Location ID"] = selected["Location ID"].to_numpy()
        frames.append(frame[GL_COLUMNS])

    def add_journal(rows: pd.DataFrame, debit: np.ndarray, credit: np.ndarray, debit_vendor: np.ndarray, credit_vendor: np.ndarray) -> None:
        add_line(rows, debit, "Debit", debit_vendor)
        add_line(rows, credit, "Credit", credit_vendor)

    types = transactions["Transaction Type"].to_numpy()
    amounts = transactions["Transaction Amount"].to_numpy(dtype=float)
    cost = transactions["Cost Change"].to_numpy(dtype=float)
    fv = transactions["Fair Value Change"].to_numpy(dtype=float)
    carry = transactions["Carrying Value"].to_numpy(dtype=float)
    realized = transactions["Realized Gain/Loss"].to_numpy(dtype=float)

    for kind in ["Investment Purchase", "Follow-on Investment"]:
        mask = types == kind
        rows = transactions.loc[mask]
        value = amounts[mask]
        add_journal(rows, value, value, rows["Deal Name"].to_numpy(), np.full(len(rows), "Fund Cash", dtype=object))
    disposal = types == "Investment Disposal"
    rows = transactions.loc[disposal]
    if disposal.any():
        proceeds = amounts[disposal]
        carrying = carry[disposal]
        gain = realized[disposal]
        add_journal(rows, proceeds, carrying, np.full(len(rows), "Fund Cash", dtype=object), rows["Deal Name"].to_numpy())
        add_line(rows, np.maximum(-gain, 0), "Debit", np.full(len(rows), "Realized Loss", dtype=object))
        add_line(rows, np.maximum(gain, 0), "Credit", np.full(len(rows), "Realized Gain", dtype=object))
    for kind in ["Write-off", "Transfer/Adjustment", "Unrealized Gain", "FX Gain/Loss", "Write-up", "Write-down"]:
        mask = types == kind
        if not mask.any():
            continue
        rows = transactions.loc[mask]
        value = amounts[mask]
        signed = cost[mask] if kind == "Transfer/Adjustment" else fv[mask]
        positive = signed >= 0
        debit = np.where(positive, value, 0.0)
        credit = np.where(positive, 0.0, value)
        add_journal(rows, debit, credit, rows["Deal Name"].to_numpy(), np.full(len(rows), kind, dtype=object))
        add_journal(rows, credit, debit, np.full(len(rows), kind, dtype=object), rows["Deal Name"].to_numpy())

    fund_patterns = {
        "Management Fee": ("Fund Expense", "Fund Cash", "debit"),
        "Fund Expense": ("Fund Expense", "Fund Cash", "debit"),
        "Interest Income": ("Fund Cash", "Interest Income", "debit"),
        "Dividend Income": ("Fund Cash", "Dividend Income", "debit"),
        "Capital Contribution": ("Fund Cash", "Partner Capital", "debit"),
        "Investor Distribution": ("Investor Distribution", "Fund Cash", "debit"),
        "Other Fund Transaction": ("Other Fund Expense", "Fund Cash", "debit"),
    }
    for kind, (debit_account, credit_account, _) in fund_patterns.items():
        mask = types == kind
        if not mask.any():
            continue
        rows = transactions.loc[mask]
        value = amounts[mask]
        add_journal(rows, value, value, np.full(len(rows), debit_account, dtype=object), np.full(len(rows), credit_account, dtype=object))

    gl = pd.concat(frames, ignore_index=True)
    gl["S.No"] = np.arange(1, len(gl) + 1)
    gl["Amount"] = gl["Amount"].round(2)
    gl["Debit"] = gl["Debit"].round(2)
    gl["Credit"] = gl["Credit"].round(2)
    return gl[GL_COLUMNS]


def validate_linked_data(
    soi: pd.DataFrame,
    rollforward: pd.DataFrame,
    transactions: pd.DataFrame,
    gl: pd.DataFrame,
    funds: np.ndarray,
    deals: np.ndarray,
    investors: np.ndarray,
    positions: np.ndarray,
    affiliates: np.ndarray,
) -> pd.DataFrame:
    latest = rollforward.sort_values("Reporting Period").groupby("Investment ID", sort=False).tail(1).set_index("Investment ID")
    soi_index = soi.set_index("Investment ID")
    cost_diff = (soi_index["Ending Cost"] - latest.loc[soi_index.index, "Ending Cost"]).abs()
    fv_diff = (soi_index["Ending Fair Value"] - latest.loc[soi_index.index, "Ending Fair Value"]).abs()
    by_batch = gl.groupby("Batch Number", sort=False).agg(Debit=("Debit", "sum"), Credit=("Credit", "sum"))
    refs_in_gl = set(gl["Reference Number"].unique())
    all_refs_present = transactions["Reference Number"].isin(refs_in_gl).all()
    ordered = rollforward.sort_values(["Investment ID", "Reporting Period"], kind="stable")
    prev_cost = ordered.groupby("Investment ID", sort=False)["Ending Cost"].shift()
    prev_fv = ordered.groupby("Investment ID", sort=False)["Ending Fair Value"].shift()
    continuity = ((prev_cost.isna()) | np.isclose(ordered["Beginning Cost"], prev_cost, atol=0.01)) & ((prev_fv.isna()) | np.isclose(ordered["Beginning Fair Value"], prev_fv, atol=0.01))
    expected_cost = (ordered["Beginning Cost"] + ordered["Purchases/Additions"] + ordered["Follow-on Investment"] - ordered["Disposals"] - ordered["Write-offs"] + ordered["Transfers/Adjustments"])
    cost_formula = np.isclose(expected_cost, ordered["Ending Cost"], atol=0.01).all()
    transaction_fv_changes = transactions.set_index("Reference Number")["Fair Value Change"]
    expected_fv = ordered["Beginning Fair Value"] + ordered["Source Transaction Reference"].map(transaction_fv_changes)
    fv_formula = np.isclose(expected_fv, ordered["Ending Fair Value"], atol=0.01).all()
    checks = [
        ("SOI ending cost reconciles to RollForward", bool(cost_diff.le(0.01).all()), f"Maximum difference: {cost_diff.max():,.2f}"),
        ("SOI ending fair value reconciles to RollForward", bool(fv_diff.le(0.01).all()), f"Maximum difference: {fv_diff.max():,.2f}"),
        ("RollForward transactions reconcile to GL", bool(all_refs_present and rollforward["Source Transaction Reference"].isin(refs_in_gl).all()), f"{transactions['Reference Number'].nunique():,} source transactions"),
        ("GL debits equal credits by batch", bool(np.isclose(by_batch["Debit"], by_batch["Credit"], atol=0.01).all()), f"{len(by_batch):,} batches"),
        ("GL debits equal credits overall", bool(np.isclose(gl["Debit"].sum(), gl["Credit"].sum(), atol=0.01)), f"Debit {gl['Debit'].sum():,.2f}; Credit {gl['Credit'].sum():,.2f}"),
        ("Beginning balance equals previous period ending", bool(continuity.all()), f"{len(ordered):,} rollforward rows"),
        ("RollForward ending cost formula", bool(cost_formula), "Beginning + purchases + follow-ons - disposals - write-offs +/- adjustments"),
        ("RollForward ending fair-value formula", bool(fv_formula), "Beginning fair value + source transaction fair-value movement"),
        ("Fund names are valid", bool(soi["Fund Name"].isin(funds).all() and rollforward["Fund Name"].isin(funds).all() and gl["Fund Name"].isin(funds).all()), f"{len(funds):,} funds"),
        ("Deal names are valid", bool(soi["Deal Name"].isin(deals).all() and rollforward["Deal Name"].isin(deals).all() and gl.loc[gl["Deal Name"].ne(""), "Deal Name"].isin(deals).all()), f"{len(deals):,} deals; fund-level GL rows may be blank"),
        ("Investor names are valid", bool(soi["Investor Name"].isin(investors).all() and rollforward["Investor Name"].isin(investors).all() and gl["Investor Name"].isin(investors).all()), "Investor master values"),
        ("Position values are valid", bool(soi["Position"].isin(positions).all() and rollforward["Position"].isin(positions).all() and gl.loc[gl["Position"].ne(""), "Position"].isin(positions).all()), "Fund-level GL rows may be blank"),
        ("Affiliate names are valid", bool(gl["Affiliate"].isin(affiliates).all()), f"{len(affiliates):,} affiliates"),
        ("Transaction reference numbers are unique", bool(transactions["Reference Number"].is_unique), f"{len(transactions):,} transactions"),
        ("Batch numbers are unique", bool(transactions["Batch Number"].is_unique), f"{transactions['Batch Number'].nunique():,} batches"),
    ]
    return pd.DataFrame(checks, columns=["Validation", "Status", "Details"])


def export_soi_excel(data: dict) -> bytes:
    workbook = Workbook(write_only=True)
    header_fill = PatternFill("solid", fgColor="17365D")
    header_font = Font(color="FFFFFF", bold=True)
    detail = workbook.create_sheet("SOI Detail")
    detail.freeze_panes = "A5"
    detail.append(["Schedule of Investments"])
    detail.append([f"Reporting Date: {data['reporting_date']} | Base Currency: {data['base_currency']}"])
    detail.append([f"Funds: {data['soi']['Fund Name'].nunique():,} | Investments: {len(data['soi']):,}"])
    header_cells = []
    for value in SOI_COLUMNS:
        cell = WriteOnlyCell(detail, value=value)
        cell.fill = header_fill
        cell.font = header_font
        header_cells.append(cell)
    detail.append(header_cells)
    detail.auto_filter.ref = f"A4:{get_column_letter(len(SOI_COLUMNS))}{len(data['soi']) + 4}"
    for index, column in enumerate(SOI_COLUMNS, start=1):
        detail.column_dimensions[get_column_letter(index)].width = min(max(len(column) + 2, 13), 25)
    currency_columns = {"Beginning Cost", "Additions", "Disposals", "Ending Cost", "Beginning Fair Value", "Unrealized Gain/Loss", "Realized Gain/Loss", "FX Gain/Loss", "Ending Fair Value"}
    percent_columns = {"Ownership %"}
    date_columns = {"Investment Date", "Acquisition Date", "Reporting Date"}
    for row in data["soi"].itertuples(index=False, name=None):
        output = []
        for column, value in zip(SOI_COLUMNS, row):
            if column in date_columns and not pd.isna(value):
                value = pd.Timestamp(value).to_pydatetime()
            cell = WriteOnlyCell(detail, value=None if pd.isna(value) else value)
            if column in currency_columns:
                currency = data["base_currency"]
                cell.number_format = f'"{currency}" #,##0.00;[Red]("{currency}" #,##0.00);-'
            elif column == "Shares/Units":
                cell.number_format = '#,##0.0000;[Red](#,##0.0000);-'
            elif column in percent_columns:
                cell.number_format = '0.00%'
                cell.value = float(value)
            elif column in date_columns:
                cell.number_format = 'yyyy-mm-dd'
            output.append(cell)
        detail.append(output)

    summary = workbook.create_sheet("Fund Summary")
    summary.freeze_panes = "A2"
    summary_headers = []
    for value in data["fund_summary"].columns:
        cell = WriteOnlyCell(summary, value=value)
        cell.fill = header_fill
        cell.font = header_font
        summary_headers.append(cell)
    summary.append(summary_headers)
    for row in data["fund_summary"].itertuples(index=False, name=None):
        output = []
        for column, value in zip(data["fund_summary"].columns, row):
            cell = WriteOnlyCell(summary, value=value)
            if column in {"Beginning Cost", "Additions", "Disposals", "Ending Cost", "Ending Fair Value"}:
                currency = data["base_currency"]
                cell.number_format = f'"{currency}" #,##0.00;[Red]("{currency}" #,##0.00);-'
            output.append(cell)
        summary.append(output)
    summary.auto_filter.ref = f"A1:{get_column_letter(len(data['fund_summary'].columns))}{len(data['fund_summary']) + 1}"
    validation = workbook.create_sheet("Validation")
    validation.append(list(data["validation"].columns))
    for row in data["validation"].itertuples(index=False, name=None):
        validation.append(list(row))
    output = BytesIO()
    workbook.save(output)
    return output.getvalue()


def export_soi_pdf(data: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, legal
    from reportlab.pdfgen import canvas

    output = BytesIO()
    page_width, page_height = landscape(legal)
    pdf = canvas.Canvas(output, pagesize=(page_width, page_height), pageCompression=1)
    margin = 30
    columns = ["Fund Name", "Investment ID", "Deal Name", "Position", "Currency", "Beginning Cost", "Additions", "Disposals", "Ending Cost", "Ending Fair Value", "Ownership %"]
    widths = [80, 90, 95, 85, 48, 80, 70, 70, 80, 85, 62]
    top = page_height - 40
    y = top
    page_number = 0

    def draw_page_header() -> float:
        nonlocal page_number
        page_number += 1
        pdf.setFillColor(colors.HexColor("#17365D"))
        pdf.rect(0, page_height - 66, page_width, 66, fill=1, stroke=0)
        pdf.setFillColor(colors.white)
        pdf.setFont("Helvetica-Bold", 15)
        pdf.drawString(margin, page_height - 27, "Schedule of Investments")
        pdf.setFont("Helvetica", 9)
        fund_names = list(data["soi"]["Fund Name"].drop_duplicates().head(3))
        fund_label = "; ".join(fund_names)
        if data["soi"]["Fund Name"].nunique() > len(fund_names):
            fund_label += f" (+{data['soi']['Fund Name'].nunique() - len(fund_names):,} more)"
        pdf.drawString(margin, page_height - 44, f"Reporting date: {data['reporting_date']}    Base currency: {data['base_currency']}    Fund(s): {fund_label[:100]}")
        pdf.setFillColor(colors.black)
        y_value = page_height - 88
        pdf.setFillColor(colors.HexColor("#17365D"))
        pdf.rect(margin, y_value - 4, sum(widths), 19, fill=1, stroke=0)
        pdf.setFillColor(colors.white)
        pdf.setFont("Helvetica-Bold", 7)
        x = margin + 3
        for column, width in zip(columns, widths):
            pdf.drawString(x, y_value + 2, column[:18])
            x += width
        pdf.setFillColor(colors.black)
        return y_value - 12

    y = draw_page_header()
    sums = {column: 0.0 for column in ["Beginning Cost", "Additions", "Disposals", "Ending Cost", "Ending Fair Value"]}
    fund_summaries = data["fund_summary"].set_index("Fund Name").to_dict("index")
    row_count = 0

    def draw_fund_subtotal(fund_name: str) -> None:
        nonlocal y
        if y < 48:
            pdf.setFont("Helvetica", 8)
            pdf.drawRightString(page_width - margin, 22, f"Page {page_number}")
            pdf.showPage()
            y = draw_page_header()
        summary = fund_summaries[fund_name]
        pdf.setFont("Helvetica-Bold", 7)
        pdf.drawString(
            margin + 3, y,
            f"{fund_name[:28]} subtotal | Ending Cost {data['base_currency']} {summary['Ending Cost']:,.2f} | Ending Fair Value {data['base_currency']} {summary['Ending Fair Value']:,.2f}",
        )
        y -= 14

    current_fund = None
    column_indexes = [data["soi"].columns.get_loc(column) for column in columns]
    for full_row in data["soi"].itertuples(index=False, name=None):
        row = tuple(full_row[index] for index in column_indexes)
        fund_name = row[0]
        if current_fund is not None and fund_name != current_fund:
            draw_fund_subtotal(current_fund)
        if y < 48:
            pdf.setFont("Helvetica", 8)
            pdf.drawRightString(page_width - margin, 22, f"Page {page_number}")
            pdf.showPage()
            y = draw_page_header()
        values = []
        for column, value in zip(columns, row):
            if column in sums:
                sums[column] += float(value)
            if column in sums:
                values.append(f"{data['base_currency']} {float(value):,.0f}")
            elif column == "Ownership %":
                values.append(f"{float(value):.2%}")
            else:
                values.append(str(value))
        pdf.setFont("Helvetica", 7)
        x = margin + 3
        for value, width in zip(values, widths):
            pdf.drawString(x, y, value[:22])
            x += width
        y -= 12
        row_count += 1
        current_fund = fund_name
    if current_fund is not None:
        draw_fund_subtotal(current_fund)
    if y < 70:
        pdf.setFont("Helvetica", 8)
        pdf.drawRightString(page_width - margin, 22, f"Page {page_number}")
        pdf.showPage()
        y = draw_page_header()
    y -= 8
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawString(margin, y, f"Portfolio total ({row_count:,} investments)")
    for column in ["Beginning Cost", "Additions", "Disposals", "Ending Cost", "Ending Fair Value"]:
        y -= 14
        pdf.drawString(margin + 12, y, f"{column}: {data['base_currency']} {sums[column]:,.2f}")
    pdf.setFont("Helvetica", 8)
    pdf.drawRightString(page_width - margin, 22, f"Page {page_number}")
    pdf.save()
    return output.getvalue()


def dataframe_bytes(frame: pd.DataFrame, fmt: str, sheet_name: str = "Data") -> bytes:
    output = BytesIO()
    if fmt == "csv":
        frame.to_csv(output, index=False)
    elif fmt == "parquet":
        frame.to_parquet(output, index=False)
    elif fmt == "json":
        frame.to_json(output, orient="records", lines=True)
    elif fmt == "xlsx":
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            max_rows = 1_048_575
            for part, start in enumerate(range(0, len(frame), max_rows), start=1):
                frame.iloc[start:start + max_rows].to_excel(writer, index=False, sheet_name=f"{sheet_name[:25]} {part}")
    else:
        raise ValueError(f"Unsupported output format: {fmt}")
    return output.getvalue()