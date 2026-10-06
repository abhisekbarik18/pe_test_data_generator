from __future__ import annotations

from datetime import date
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import yaml

from . import generators as g
from .pe_generators import (
    GL_COLUMNS,
    dataframe_bytes,
    export_soi_excel as _export_soi_excel,
    export_soi_pdf as _export_soi_pdf,
    validate_linked_data,
)


PROFILE_FILE = "soi_profiles.yaml"


def format_number(value: float, decimals: int = 2, scale: float = 1) -> str:
    if scale <= 0:
        raise ValueError("Display scale must be greater than zero.")
    return f"{float(value) / scale:,.{decimals}f}"


def format_currency(value: float, currency: str = "USD", decimals: int = 2, scale: float = 1) -> str:
    return f"{currency} {format_number(value, decimals, scale)}"


def format_percentage(value: float, decimals: int = 2) -> str:
    return f"{float(value):.{decimals}%}"


def format_date(value: Any, output_format: str = "%Y-%m-%d") -> str:
    return pd.Timestamp(value).strftime(output_format)


def make_file_name(title: str, profile: str, reporting_date: str, extension: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", f"{title}_{profile}_{reporting_date}").strip("_").lower()
    return f"{stem}.{extension.lstrip('.').lower()}"


def report_metadata(data: dict[str, Any]) -> dict[str, Any]:
    canonical = data.get("canonical_soi", data.get("soi", pd.DataFrame()))
    return {
        "profile": data.get("profile", "PRIVATE_EQUITY"),
        "reporting_date": data.get("reporting_date", ""),
        "base_currency": data.get("base_currency", "USD"),
        "fund_count": int(canonical["Fund Name"].nunique()) if "Fund Name" in canonical else 0,
        "investment_count": len(canonical),
        "generated_from": "shared financial model",
    }


def load_soi_profiles(config_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    config_dir = config_dir or Path(__file__).resolve().parent.parent / "config"
    with (config_dir / PROFILE_FILE).open("r", encoding="utf-8") as handle:
        profiles = yaml.safe_load(handle).get("profiles", {})
    if not profiles:
        raise ValueError(f"No SOI profiles configured in {config_dir / PROFILE_FILE}.")
    return profiles


def _profile_fields(frame: pd.DataFrame, configuration: dict[str, Any]) -> pd.DataFrame:
    count = len(frame)
    rng = g.get_rng(int(configuration.get("seed", 42)))
    country = rng.choice(["United States", "United Kingdom", "Canada", "Germany", "France", "Japan", "Australia", "Singapore"], size=count, p=[0.48, 0.13, 0.1, 0.08, 0.06, 0.06, 0.05, 0.04])
    industry = rng.choice(["Technology", "Healthcare", "Industrials", "Consumer", "Financials", "Energy", "Real Estate", "Communication Services"], size=count, p=[0.2, 0.17, 0.15, 0.13, 0.13, 0.08, 0.07, 0.07])
    security_class = rng.choice(["Common Stock", "Preferred Stock", "Class A", "Class B"], size=count, p=[0.55, 0.2, 0.15, 0.1])
    security_suffix = rng.choice(["Ordinary Shares", "Class A", "Series B", "Preferred Units", "Senior Notes"], size=count)
    security_names = np.char.add(np.char.add(frame["Deal Name"].astype(str).to_numpy(dtype=str), " "), security_suffix.astype(str))
    rate = np.round(rng.uniform(0.045, 0.145, size=count), 5)
    spread = rng.integers(250, 1001, size=count)
    maturity_days = rng.integers(365, 365 * 12, size=count)
    reporting_dates = pd.to_datetime(frame["Reporting Date"])
    maturity = (reporting_dates + pd.to_timedelta(maturity_days, unit="D")).dt.strftime("%Y-%m-%d")
    fund_fair_value = frame.groupby("Fund Name", sort=False)["Ending Fair Value"].transform("sum").replace(0, np.nan)
    fund_par = frame.groupby("Fund Name", sort=False)["Shares/Units"].transform("sum").replace(0, np.nan)
    pct_net_assets = (frame["Ending Fair Value"] / fund_fair_value).fillna(0)
    par_pct = (frame["Shares/Units"] / fund_par).fillna(0)
    footnote = np.where(frame["Valuation Method"].eq("NAV"), "1", np.where(frame["Investment Status"].eq("Active"), "", "2"))
    asset_classes = rng.choice(["Private Investment Companies", "Credit Facilities", "SPVs", "Partnerships", "Money Market Investments", "Futures/Derivatives"], size=count, p=[0.4, 0.18, 0.14, 0.14, 0.08, 0.06])
    strategies = rng.choice(["Buyout", "Growth Equity", "Venture Capital", "Direct Lending", "Special Situations", "Hedged Strategies"], size=count)
    rating = rng.choice(["AAA", "AA", "A", "BBB", "BB", "B", "Unrated"], size=count, p=[0.04, 0.08, 0.16, 0.25, 0.2, 0.15, 0.12])
    reference_rate = rng.choice(["SOFR", "Prime", "EURIBOR", "SONIA", "Fixed"], size=count, p=[0.62, 0.15, 0.1, 0.08, 0.05])
    fair_value_hierarchy = np.where(frame["Position"].eq("Fund Interest"), "Level 3", np.where(frame["Security Type"].eq("Debt"), "Level 2", "Level 3"))
    formatted_shares = np.char.mod("%.4f", frame["Shares/Units"].to_numpy(dtype=float))
    return pd.DataFrame({
        "Fund Name": frame["Fund Name"], "Fund ID": frame["Fund ID"],
        "Deal Name": frame["Deal Name"], "Investment ID": frame["Investment ID"],
        "Position": frame["Position"], "Investor Name": frame["Investor Name"],
        "Investment Date": frame["Investment Date"], "Acquisition Date": frame["Acquisition Date"],
        "Currency": frame["Currency"], "Reporting Date": frame["Reporting Date"],
        "Country": country, "Geography": country, "Industry": industry,
        "Security Name": security_names, "Security Class": security_class,
        "Security Type": frame["Security Type"], "Position Description": frame["Deal Name"] + " - " + frame["Position"],
        "Identifier": frame["Investment ID"], "Shares": frame["Shares/Units"],
        "Shares/Units": frame["Shares/Units"],
        "Shares/Units or N/A": np.where(frame["Security Type"].eq("Debt"), "N/A", formatted_shares),
        "Fair Value": frame["Ending Fair Value"], "Cost": frame["Ending Cost"],
        "% Net Assets": pct_net_assets, "Notes/Footnotes": footnote,
        "Par Amount": frame["Shares/Units"] * rng.uniform(900, 1100, size=count),
        "Par %": par_pct, "Maturity/Reset Date": maturity, "Final Maturity": maturity,
        "Interest Rate": rate, "Reference Rate": reference_rate,
        "Amortized Cost": frame["Ending Cost"], "Rating": rating,
        "Investment/Portfolio Company": frame["Deal Name"],
        "Acquisition Date": frame["Acquisition Date"], "Valuation Method": frame["Valuation Method"],
        "Fair Value Hierarchy": fair_value_hierarchy, "Investment Status": frame["Investment Status"],
        "Footnote Indicators": footnote, "Investment Name": frame["Deal Name"],
        "Spread": spread, "First Acquisition Date": frame["Acquisition Date"],
        "Maturity Date": maturity, "Cost/Principal": frame["Ending Cost"],
        "PIK Indicator": rng.choice(["Yes", "No"], size=count, p=[0.18, 0.82]),
        "Commitment/Unfunded Status": rng.choice(["Funded", "Partially Unfunded", "Unfunded"], size=count, p=[0.78, 0.18, 0.04]),
        "Footnotes": footnote, "Asset Class": asset_classes, "Strategy": strategies,
        "Investment Type": asset_classes,
        "NAV/Practical Expedient": np.where(frame["Valuation Method"].eq("NAV"), "Practical Expedient", "No"),
    })


def generate_soi(
    data: dict[str, Any],
    profile: str = "PRIVATE_EQUITY",
    output_format: str = "Both",
    configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project one common investment dataset into a configured SOI profile and exports."""
    configuration = configuration or {}
    profile_key = profile.upper()
    profiles = configuration.get("profiles") or load_soi_profiles(configuration.get("config_dir"))
    if profile_key not in profiles:
        raise ValueError(f"Unknown SOI profile '{profile}'. Available profiles: {', '.join(profiles)}")
    profile_settings = profiles[profile_key]
    source = data.get("canonical_soi", data.get("soi"))
    if source is None:
        raise ValueError("The financial model must include canonical SOI investment balances.")
    enriched = _profile_fields(source, configuration)
    fields = profile_settings["fields"]
    missing = sorted(set(fields.values()) - set(enriched.columns))
    if missing:
        raise ValueError(f"SOI profile '{profile_key}' references unavailable field(s): {', '.join(missing)}")
    report = enriched[list(fields.values())].copy()
    report.columns = list(fields)
    hierarchy = configuration.get("hierarchy", profile_settings.get("hierarchy", []))
    hierarchy = [column for column in hierarchy if column in report.columns]
    if hierarchy:
        report = report.sort_values(hierarchy, kind="stable", ignore_index=True)
    result = dict(data)
    result.update({
        "canonical_soi": source,
        "soi": report,
        "soi_columns": list(report.columns),
        "profile": profile_key,
        "profile_settings": profile_settings,
        "hierarchy": hierarchy,
        "cost_fields": profile_settings.get("cost_fields", []),
        "fair_value_fields": profile_settings.get("fair_value_fields", []),
        "currency_fields": profile_settings.get("currency_fields", []),
        "percentage_fields": profile_settings.get("percentage_fields", []),
        "date_fields": profile_settings.get("date_fields", []),
        "footnote_fields": profile_settings.get("footnote_fields", []),
        "display_scale": float(configuration.get("display_scale", 1)),
    })
    formats = {"EXCEL": "xlsx", "PDF": "pdf", "BOTH": "both"}
    selected_format = formats.get(output_format.upper())
    if selected_format is None:
        raise ValueError("SOI output format must be Excel, PDF, or Both.")
    result["excel"] = _export_soi_excel(result) if selected_format in {"xlsx", "both"} else None
    result["pdf"] = _export_soi_pdf(result) if selected_format in {"pdf", "both"} else None
    return result


def generate_rollforward(data: dict[str, Any], configuration: dict[str, Any] | None = None) -> pd.DataFrame:
    del configuration
    if "rollforward" not in data:
        raise ValueError("The financial model does not contain RollForward balances.")
    return data["rollforward"]


def generate_gl(data: dict[str, Any], configuration: dict[str, Any] | None = None) -> pd.DataFrame:
    del configuration
    frame = data.get("gl")
    if frame is None:
        raise ValueError("The financial model does not contain generated GL entries.")
    if list(frame.columns) != GL_COLUMNS:
        raise ValueError("Generated GL does not match the required 14-column schema.")
    return frame


def reconcile_soi_rollforward_gl(
    soi: pd.DataFrame | dict[str, Any],
    rollforward: pd.DataFrame | None = None,
    gl: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a structured, row-level reconciliation result for shared financial data."""
    if isinstance(soi, dict):
        dataset = soi
        canonical = dataset.get("canonical_soi", dataset.get("soi"))
        rollforward = dataset.get("rollforward")
        gl = dataset.get("gl")
        transactions = dataset.get("transactions")
        model = dataset.get("financial_model", {})
        funds = model.get("Fund", canonical[["Fund ID", "Fund Name"]].drop_duplicates())["Fund Name"].drop_duplicates().to_numpy()
        deals = model.get("Deal", canonical[["Fund ID", "Fund Name", "Deal Name"]].drop_duplicates())["Deal Name"].drop_duplicates().to_numpy()
        investors = model.get("Investor", canonical[["Investor Name"]].drop_duplicates())["Investor Name"].drop_duplicates().to_numpy()
        positions = model.get("Position", canonical[["Position"]].drop_duplicates())["Position"].drop_duplicates().to_numpy()
        affiliates = dataset.get("transactions", pd.DataFrame()).get("Affiliate", pd.Series(dtype=object)).drop_duplicates().to_numpy()
        reporting_date = dataset.get("reporting_date", "")
    else:
        canonical = soi
        transactions = None
        reporting_date = ""
    if rollforward is None or gl is None:
        raise ValueError("SOI, RollForward, and GL data are all required for reconciliation.")
    if transactions is None:
        transaction_refs = rollforward["Source Transaction Reference"].drop_duplicates()
        transactions = pd.DataFrame({
            "Reference Number": transaction_refs,
            "Transaction Type": "Investment",
            "Investment ID": rollforward["Investment ID"].to_numpy(),
            "Fair Value Change": rollforward["Ending Fair Value"].to_numpy() - rollforward["Beginning Fair Value"].to_numpy(),
        })
        funds = canonical["Fund Name"].drop_duplicates().to_numpy()
        deals = canonical["Deal Name"].drop_duplicates().to_numpy()
        investors = canonical["Investor Name"].drop_duplicates().to_numpy()
        positions = canonical["Position"].drop_duplicates().to_numpy()
        affiliates = gl["Affiliate"].drop_duplicates().to_numpy()
    summary = validate_linked_data(canonical, rollforward, transactions, gl, funds, deals, investors, positions, affiliates)
    result_frames: list[pd.DataFrame] = []

    def add_check(
        name: str,
        status: np.ndarray | bool,
        fund_values: Any = "",
        investment_values: Any = "",
        dates: Any = reporting_date,
        soi_amount: Any = np.nan,
        rollforward_amount: Any = np.nan,
        gl_amount: Any = np.nan,
        variance: Any = np.nan,
        reason: Any = "",
    ) -> None:
        values = {
            "Check Name": name, "Fund": fund_values, "Investment": investment_values,
            "Reporting Date": dates, "SOI Amount": soi_amount,
            "RollForward Amount": rollforward_amount, "GL Amount": gl_amount,
            "Variance": variance, "Status": np.where(status, "PASS", "FAIL"),
            "Exception Reason": np.where(status, "", reason),
        }
        lengths = [np.asarray(value).size for value in values.values() if np.asarray(value).ndim > 0]
        row_count = max(lengths, default=1)
        frame = pd.DataFrame({
            column: np.repeat(value, row_count) if np.asarray(value).ndim == 0 else value
            for column, value in values.items()
        })
        result_frames.append(frame)

    for record in summary.itertuples(index=False):
        add_check(record.Validation, bool(record.Status), reason="" if record.Status else record.Details)

    latest = rollforward.sort_values("Reporting Period", kind="stable").groupby("Investment ID", sort=False).tail(1)
    matched = canonical[["Fund Name", "Investment ID", "Ending Cost", "Ending Fair Value"]].merge(
        latest[["Investment ID", "Ending Cost", "Ending Fair Value", "Reporting Period"]],
        on="Investment ID", suffixes=("_SOI", "_RollForward"), how="outer", indicator=True,
    )
    cost_variance = (matched["Ending Cost_SOI"] - matched["Ending Cost_RollForward"]).abs()
    add_check(
        "SOI ending cost equals RollForward", matched["_merge"].eq("both") & cost_variance.le(0.01),
        matched["Fund Name"], matched["Investment ID"], matched["Reporting Period"],
        matched["Ending Cost_SOI"], matched["Ending Cost_RollForward"], variance=cost_variance,
        reason="Missing matching investment or ending cost variance exceeds tolerance.",
    )
    fair_value_variance = (matched["Ending Fair Value_SOI"] - matched["Ending Fair Value_RollForward"]).abs()
    add_check(
        "SOI ending fair value equals RollForward", matched["_merge"].eq("both") & fair_value_variance.le(0.01),
        matched["Fund Name"], matched["Investment ID"], matched["Reporting Period"],
        matched["Ending Fair Value_SOI"], matched["Ending Fair Value_RollForward"], variance=fair_value_variance,
        reason="Missing matching investment or ending fair-value variance exceeds tolerance.",
    )

    ordered = rollforward.sort_values(["Investment ID", "Reporting Period"], kind="stable").copy()
    prior_cost = ordered.groupby("Investment ID", sort=False)["Ending Cost"].shift()
    prior_fv = ordered.groupby("Investment ID", sort=False)["Ending Fair Value"].shift()
    period_variance = np.maximum(
        (ordered["Beginning Cost"] - prior_cost).abs().fillna(0),
        (ordered["Beginning Fair Value"] - prior_fv).abs().fillna(0),
    )
    beginning_ok = prior_cost.isna() | (period_variance <= 0.01)
    add_check(
        "Prior period ending equals current beginning", beginning_ok,
        ordered["Fund Name"], ordered["Investment ID"], ordered["Reporting Period"],
        rollforward_amount=ordered["Beginning Cost"], variance=period_variance,
        reason="Current beginning cost/fair value does not equal prior ending balance.",
    )

    batch_totals = gl.groupby(["Batch Number", "Fund Name"], as_index=False, sort=False).agg(
        Debit=("Debit", "sum"), Credit=("Credit", "sum"), Investment=("Deal Name", "first"),
    )
    batch_variance = (batch_totals["Debit"] - batch_totals["Credit"]).abs()
    add_check(
        "GL debits equal credits by Batch Number", batch_variance.le(0.01),
        batch_totals["Fund Name"], batch_totals["Investment"], gl_amount=batch_totals["Credit"],
        variance=batch_variance, reason="Batch debit and credit totals differ.",
    )
    return pd.concat(result_frames, ignore_index=True)[[
        "Check Name", "Fund", "Investment", "Reporting Date", "SOI Amount",
        "RollForward Amount", "GL Amount", "Variance", "Status", "Exception Reason",
    ]]


def export_soi_excel(data: dict[str, Any], output_path: str | Path | None = None) -> bytes | Path:
    return _export_soi_excel(data, output_path=output_path)


def export_soi_pdf(data: dict[str, Any], output_path: str | Path | None = None) -> bytes | Path:
    return _export_soi_pdf(data, output_path=output_path)


def _write_frame(frame: pd.DataFrame, output_format: str, output_path: str | Path, sheet_name: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = output_format.lower().lstrip(".")
    if fmt == "csv":
        frame.to_csv(path, index=False, chunksize=100_000)
    elif fmt == "json":
        for start in range(0, len(frame), 100_000):
            frame.iloc[start:start + 100_000].to_json(path, orient="records", lines=True, mode="a" if start else "w")
    elif fmt == "parquet":
        import pyarrow as pa
        import pyarrow.parquet as parquet

        writer = None
        try:
            for start in range(0, len(frame), 100_000):
                table = pa.Table.from_pandas(frame.iloc[start:start + 100_000], preserve_index=False)
                if writer is None:
                    writer = parquet.ParquetWriter(path, table.schema)
                writer.write_table(table)
            if writer is None:
                frame.to_parquet(path, index=False)
        finally:
            if writer is not None:
                writer.close()
    elif fmt == "xlsx":
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            max_rows = 1_048_575
            for part, start in enumerate(range(0, len(frame), max_rows), start=1):
                frame.iloc[start:start + max_rows].to_excel(writer, index=False, sheet_name=f"{sheet_name[:25]} {part}")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    return path


def export_gl(
    data: pd.DataFrame | dict[str, Any],
    output_format: str = "csv",
    output_path: str | Path | None = None,
) -> bytes | Path:
    frame = generate_gl(data, {}) if isinstance(data, dict) else data
    if output_path is not None:
        if list(frame.columns) != GL_COLUMNS:
            raise ValueError("GL export requires the exact 14-column schema.")
        return _write_frame(frame, output_format, output_path, "General Ledger")
    return dataframe_bytes(frame, output_format.lower(), "General Ledger")


def export_rollforward(
    data: pd.DataFrame | dict[str, Any],
    output_format: str = "csv",
    output_path: str | Path | None = None,
) -> bytes | Path:
    frame = generate_rollforward(data, {}) if isinstance(data, dict) else data
    if output_path is not None:
        return _write_frame(frame, output_format, output_path, "RollForward")
    return dataframe_bytes(frame, output_format.lower(), "RollForward")