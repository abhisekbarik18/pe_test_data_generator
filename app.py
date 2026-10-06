"""
Streamlit front-end for the Private Equity Test Data Generator.

Run with:  streamlit run app.py
"""
import io
import logging
import time
import yaml
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import date

from src import generators as g
from src.engine import generate_dataframe
from src.pe_generators import dataframe_bytes, generate_linked_datasets
from src.financial_files import (
    generate_gl,
    generate_rollforward,
    generate_soi,
    load_soi_profiles,
    reconcile_soi_rollforward_gl,
)
from src.mappings import (
    apply_alias_mapping,
    apply_mapping,
    build_alias_profile_bundle,
    load_mapping_profiles,
    save_mapping_profile,
    validate_mapping,
)

log_path = Path(__file__).resolve().parent / "logs" / "pe_testdata_generator.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("pe_testdata_generator")
logger.setLevel(logging.INFO)
logger.propagate = False

if not any(isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == str(log_path) for handler in logger.handlers):
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

CONFIG_DIR = Path(__file__).parent / "config"
MODULES = ["investors", "vendors", "affiliates", "general_ledger", "commitments"]
SOI_PROFILES = load_soi_profiles(CONFIG_DIR)

st.set_page_config(page_title="PE Test Data Generator", layout="wide", page_icon="🧪")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "schemas" not in st.session_state:
    schemas = {}
    for m in MODULES:
        with open(CONFIG_DIR / f"{m}.yaml") as f:
            schemas[m] = f.read()
    st.session_state.schemas = schemas

if "mapping_profiles" not in st.session_state:
    st.session_state.mapping_profiles = {
        m: load_mapping_profiles(m, CONFIG_DIR) for m in MODULES
    }

if "alias_profile_bundles" not in st.session_state:
    st.session_state.alias_profile_bundles = {}
    for m in MODULES:
        try:
            with open(CONFIG_DIR / f"{m}.yaml") as handle:
                schema = yaml.safe_load(handle) or {}
            columns = [column.get("name") for column in schema.get("columns", []) if column.get("name")]
            st.session_state.alias_profile_bundles[m] = build_alias_profile_bundle(m, columns, CONFIG_DIR)
        except Exception as exc:  # pragma: no cover - UI setup guard
            logger.warning("Unable to load built-in alias bundle for %s: %s", m, exc)
            st.session_state.alias_profile_bundles[m] = {}

if "custom_modules" not in st.session_state:
    st.session_state.custom_modules = []  # list of module names added at runtime

if "last_df" not in st.session_state:
    st.session_state.last_df = None
    st.session_state.last_stats = None
    st.session_state.last_module = None
if "last_schema" not in st.session_state:
    st.session_state.last_schema = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🧪 PE Test Data Generator")
st.sidebar.caption("Private Equity domain · synthetic test data")

all_modules = MODULES + st.session_state.custom_modules
module = st.sidebar.selectbox(
    "Module",
    all_modules,
    index=0,
    format_func=lambda name: name.replace("_", " ").title(),
)
section = st.sidebar.radio("Section", ["Generate data", "Column mappings", "Linked PE data"], key="app_section")

try:
    profiles = load_mapping_profiles(module, CONFIG_DIR)
    st.session_state.mapping_profiles[module] = profiles
except (ValueError, yaml.YAMLError) as e:
    profiles = {}
    st.sidebar.error(f"Cannot load mapping profiles: {e}")

st.sidebar.markdown("---")
with st.sidebar.expander("➕ Add a new module"):
    new_mod_name = st.text_input("Module name", key="new_mod_name", placeholder="e.g. commitments")
    if st.button("Create blank module"):
        if not new_mod_name.strip():
            st.sidebar.error("Give the module a name.")
        elif new_mod_name in all_modules:
            st.sidebar.error("That module already exists.")
        else:
            blank = {
                "module": new_mod_name.strip(),
                "columns": [
                    {"name": "id", "type": "sequence_id", "prefix": new_mod_name[:3].upper(), "width": 7, "start": 1},
                    {"name": "name", "type": "name_pool", "pool": "company", "pool_size": 10000, "unique": True},
                ],
            }
            st.session_state.custom_modules.append(new_mod_name.strip())
            st.session_state.schemas[new_mod_name.strip()] = yaml.dump(blank, sort_keys=False)
            st.session_state.mapping_profiles[new_mod_name.strip()] = {}
            st.rerun()

st.sidebar.markdown("---")

if section == "Linked PE data" or (module == "general_ledger" and section == "Generate data"):
    st.title("Linked PE datasets")
    st.caption("Generate one shared investment transaction model and reconcile SOI, RollForward, and GL outputs.")

    config_cols = st.columns(4)
    with config_cols[0]:
        soi_records = st.number_input("SOI record count", min_value=1, max_value=1_000_000, value=1000, step=100)
    with config_cols[1]:
        number_of_investments = st.number_input(
            "Number of investments", min_value=1, max_value=int(soi_records),
            value=min(500, int(soi_records)), step=100,
        )
        reporting_date = st.date_input("Reporting date", value=date.today())
    with config_cols[0]:
        number_of_funds = st.number_input(
            "Number of funds", min_value=1, max_value=int(number_of_investments),
            value=min(5, int(number_of_investments)), step=1,
        )
    with config_cols[2]:
        number_of_periods = st.number_input("Number of reporting periods", min_value=1, max_value=20, value=4, step=1)
        base_currency = st.selectbox("Base currency", ["USD", "EUR", "GBP", "JPY", "CHF", "SGD"])
    with config_cols[3]:
        soi_profile = st.selectbox("SOI Template Profile", list(SOI_PROFILES))
        soi_output_format = st.selectbox("SOI Output Format", ["Excel", "PDF", "Both"])
        display_units = st.selectbox("SOI display units", ["Units", "Thousands", "Millions"])
        linked_output_format = st.selectbox("Data output format", ["csv", "parquet", "json", "xlsx"])
        linked_seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42, step=1, key="linked_pe_seed")

    generate_linked = st.button("Generate linked datasets", type="primary", use_container_width=True)
    if generate_linked:
        started = time.perf_counter()
        try:
            with st.spinner(f"Generating and reconciling {soi_records:,} SOI records..."):
                linked_data = generate_linked_datasets(
                    record_count=int(soi_records),
                    fund_count=int(number_of_funds),
                    investment_count=int(number_of_investments),
                    period_count=int(number_of_periods),
                    reporting_date=reporting_date,
                    seed=int(linked_seed),
                    base_currency=base_currency,
                    config_dir=CONFIG_DIR,
                    profile=soi_profile,
                )
                display_scale = {"Units": 1, "Thousands": 1_000, "Millions": 1_000_000}[display_units]
                linked_data = generate_soi(
                    linked_data,
                    profile=soi_profile,
                    output_format=soi_output_format,
                    configuration={"seed": int(linked_seed), "config_dir": CONFIG_DIR, "display_scale": display_scale},
                )
                reconciliation = reconcile_soi_rollforward_gl(linked_data)
                failed_checks = reconciliation.loc[reconciliation["Status"] != "PASS"]
                if not failed_checks.empty:
                    failed_names = failed_checks["Check Name"].drop_duplicates().tolist()
                    raise ValueError("Reconciliation failed: " + "; ".join(failed_names))
                soi_exports = {fmt: linked_data[key] for fmt, key in [("xlsx", "excel"), ("pdf", "pdf")] if linked_data[key] is not None}
                data_exports = {
                    "rollforward": dataframe_bytes(generate_rollforward(linked_data), linked_output_format, "RollForward"),
                    "gl": dataframe_bytes(generate_gl(linked_data), linked_output_format, "General Ledger"),
                }
                linked_data["reconciliation"] = reconciliation
            st.session_state.linked_pe_data = linked_data
            st.session_state.linked_pe_soi_exports = soi_exports
            st.session_state.linked_pe_data_exports = data_exports
            st.session_state.linked_pe_output_format = linked_output_format
            elapsed = time.perf_counter() - started
            logger.info(
                "Generated linked PE data: SOI=%d, RollForward=%d, GL=%d in %.2fs",
                len(linked_data["soi"]), len(linked_data["rollforward"]), len(linked_data["gl"]), elapsed,
            )
            st.success(f"Generated and validated SOI, RollForward, and GL in {elapsed:.2f}s.")
        except (ValueError, ImportError) as exc:
            st.error(f"Linked data generation failed: {exc}")
        except Exception as exc:
            logger.exception("Unexpected linked PE generation error")
            st.error(f"Unexpected linked data error: {exc}")

    linked_data = st.session_state.get("linked_pe_data")
    if linked_data is not None:
        st.subheader("Reconciliation")
        reconciliation = linked_data.get("reconciliation", linked_data["validation"])
        st.caption(f"{len(reconciliation):,} reconciliation results | {int((reconciliation['Status'] == 'FAIL').sum()):,} failures")
        st.dataframe(reconciliation.head(1000), hide_index=True, use_container_width=True)
        if len(reconciliation) > 1000:
            st.download_button(
                "Download complete reconciliation results",
                reconciliation.to_csv(index=False).encode("utf-8"),
                file_name="pe_reconciliation.csv", mime="text/csv",
            )
        st.subheader("SOI Preview")
        st.dataframe(linked_data["soi"].head(25), hide_index=True, use_container_width=True)
        st.subheader("Investment RollForward Preview")
        st.dataframe(linked_data["rollforward"].head(25), hide_index=True, use_container_width=True)
        st.subheader("General Ledger Preview")
        st.dataframe(linked_data["gl"].head(25), hide_index=True, use_container_width=True)
        st.subheader("Downloads")
        download_cols = st.columns(4)
        soi_exports = st.session_state.get("linked_pe_soi_exports", {})
        if "xlsx" in soi_exports:
            download_cols[0].download_button(
                "Download SOI Excel", soi_exports["xlsx"],
                file_name=f"schedule_of_investments_{linked_data['profile'].lower()}_{linked_data['reporting_date']}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        if "pdf" in soi_exports:
            download_cols[1].download_button(
                "Download SOI PDF", soi_exports["pdf"],
                file_name=f"schedule_of_investments_{linked_data['profile'].lower()}_{linked_data['reporting_date']}.pdf",
                mime="application/pdf", use_container_width=True,
            )
        fmt = st.session_state.get("linked_pe_output_format", "csv")
        mime_by_format = {
            "csv": "text/csv", "parquet": "application/octet-stream",
            "json": "application/json",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }
        exports = st.session_state.get("linked_pe_data_exports", {})
        download_cols[2].download_button(
            "Download RollForward", exports.get("rollforward", b""),
            file_name=f"investment_rollforward.{fmt}", mime=mime_by_format[fmt], use_container_width=True,
        )
        download_cols[3].download_button(
            "Download General Ledger", exports.get("gl", b""),
            file_name=f"general_ledger.{fmt}", mime=mime_by_format[fmt], use_container_width=True,
        )
    st.stop()

# ---------------------------------------------------------------------------
# Mapping profile editor
# ---------------------------------------------------------------------------
if section == "Column mappings":
    st.title(f"Column mappings: {module.replace('_', ' ').title()}")

    schema_text = st.session_state.get(f"yaml_{module}", st.session_state.schemas[module])
    try:
        schema_for_mapping = yaml.safe_load(schema_text) or {}
        schema_columns = [
            column.get("name")
            for column in schema_for_mapping.get("columns", [])
            if column.get("name")
        ]
    except yaml.YAMLError as e:
        schema_columns = []
        st.error(f"Cannot read schema columns: {e}")

    manage_options = ["(new profile)", *sorted(profiles)]
    manage_key = f"manage_mapping_profile_{module}"
    if st.session_state.get(manage_key) not in manage_options:
        st.session_state[manage_key] = "(new profile)"
    manage_profile = st.selectbox("Profile", manage_options, key=manage_key)
    initial_mapping = profiles.get(manage_profile, {})
    editor_rows = [
        {"Original column": source, "Alias": alias}
        for source, alias in initial_mapping.items()
    ]
    mapping_editor = st.data_editor(
        pd.DataFrame(editor_rows, columns=["Original column", "Alias"]),
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        key=f"mapping_editor_{module}_{manage_profile}",
    )
    profile_name = st.text_input(
        "Profile name",
        value=manage_profile if manage_profile != "(new profile)" else "",
        key=f"mapping_name_{module}_{manage_profile}",
        placeholder="e.g. downstream_import",
    )

    if st.button("Save mapping profile", key=f"save_mapping_{module}"):
        try:
            new_mapping = {}
            for _, row in mapping_editor.iterrows():
                source = "" if pd.isna(row["Original column"]) else str(row["Original column"]).strip()
                alias = "" if pd.isna(row["Alias"]) else str(row["Alias"]).strip()
                if not source and not alias:
                    continue
                if not source or not alias:
                    raise ValueError("Complete both original column and alias for every mapping row.")
                if source in new_mapping:
                    raise ValueError(f"Original column '{source}' appears more than once.")
                new_mapping[source] = alias
            save_mapping_profile(module, profile_name, new_mapping, schema_columns, CONFIG_DIR)
            st.session_state.mapping_profiles[module] = load_mapping_profiles(module, CONFIG_DIR)
            st.success(f"Saved mapping profile '{profile_name.strip()}'.")
            st.rerun()
        except ValueError as e:
            st.error(f"Cannot save mapping profile: {e}")

    st.stop()

# ---------------------------------------------------------------------------
# Main: schema editor
# ---------------------------------------------------------------------------
st.title(f"Module: {module.replace('_', ' ').title()}")

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Schema (editable)")
    st.caption(
        "Add, remove, or rename columns freely. Each column needs a `type` "
        "(sequence_id, uuid, name_pool, choice, int_range, float_range, "
        "date_range, boolean, pattern, phone, email_from) and that type's "
        "parameters. Set `unique: true` on a column to guarantee every "
        "generated value is distinct — only possible when the column's "
        "value space is large enough for the row count you request."
    )
    yaml_text = st.text_area(
        "YAML schema",
        value=st.session_state.schemas[module],
        height=480,
        key=f"yaml_{module}",
        label_visibility="collapsed",
    )
    btn_cols = st.columns([1, 1, 2])
    if btn_cols[0].button("💾 Save schema"):
        st.session_state.schemas[module] = yaml_text
        st.success("Schema saved for this session.")
    if btn_cols[1].button("↩️ Reset to default") and module in MODULES:
        with open(CONFIG_DIR / f"{module}.yaml") as f:
            st.session_state.schemas[module] = f.read()
        st.rerun()

with col_right:
    st.subheader("Column summary")
    try:
        parsed = yaml.safe_load(yaml_text)
        rows = []
        for c in parsed.get("columns", []):
            rows.append({
                "column": c.get("name"),
                "type": c.get("type"),
                "unique": "✅" if c.get("unique") else "",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=440)
    except Exception as e:
        st.error(f"YAML parse error: {e}")
        parsed = None

st.markdown("---")

# ---------------------------------------------------------------------------
# Output column mapping selection
# ---------------------------------------------------------------------------
st.subheader("Output mapping")
all_profile_names = set(profiles)
all_profile_names |= set(st.session_state.alias_profile_bundles.get(module, {}))
profile_options = ["none", *sorted(all_profile_names)]
profile_key = f"mapping_profile_{module}"
if st.session_state.get(profile_key) not in profile_options:
    st.session_state[profile_key] = "none"
selected_profile = st.selectbox("Mapping profile", profile_options, key=profile_key)
use_canonical_excel_names = st.checkbox(
    "Use canonical Excel column names",
    value=False,
    help="Rename exported columns to the standard attribute names from config/private_equity_attribute_aliases_75.json.",
)

st.caption("Built-in alias profiles: canonical_excel, business_view")

st.markdown("---")

# ---------------------------------------------------------------------------
# Generation controls
# ---------------------------------------------------------------------------
st.subheader("Generate data")

gc1, gc2, gc3, gc4 = st.columns(4)
with gc1:
    rows = st.number_input("Row count", min_value=1, max_value=5_000_000, value=1000, step=1000)
with gc2:
    seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42, step=1)
with gc3:
    out_format = st.selectbox("Output format", ["csv", "parquet", "json", "xlsx"], index=0)
with gc4:
    preview_rows = st.number_input("Preview rows", min_value=5, max_value=500, value=20, step=5)

generate_clicked = st.button("🚀 Generate", type="primary", use_container_width=True)

if generate_clicked:
    try:
        schema = yaml.safe_load(yaml_text)
    except Exception as e:
        st.error(f"Cannot generate: invalid YAML ({e})")
        schema = None

    if schema:
        try:
            selected_mapping = profiles.get(selected_profile, {}) if selected_profile != "none" else {}
            validate_mapping([column["name"] for column in schema["columns"]], selected_mapping)
            with st.spinner(f"Generating {rows:,} rows..."):
                df, gen_time = generate_dataframe(schema, int(rows), seed=int(seed), verbose=False)

            rate = rows / gen_time if gen_time > 0 else float("inf")
            st.session_state.last_df = df
            st.session_state.last_module = module
            st.session_state.last_schema = schema
            st.session_state.last_stats = {"rows": rows, "cols": len(df.columns), "gen_time": gen_time, "rate": rate}
            logger.info("Generated %s rows for module '%s' in %.2fs", rows, module, gen_time)
            st.success(f"Generated {rows:,} rows × {len(df.columns)} columns in {gen_time:.2f}s ({rate:,.0f} rows/sec)")

            if rows >= 1_000_000:
                verdict = "✅ within the 2-minute NFR" if gen_time <= 120 else "⚠️ exceeded the 2-minute NFR"
                st.info(f"1M+ row benchmark: {gen_time:.1f}s elapsed — {verdict}")

        except ValueError as e:
            st.error(f"Generation failed: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# ---------------------------------------------------------------------------
# Results: preview, uniqueness check, download
# ---------------------------------------------------------------------------
if st.session_state.last_df is not None and st.session_state.last_module == module:
    df = st.session_state.last_df
    stats = st.session_state.last_stats
    schema = st.session_state.last_schema
    active_mapping = profiles.get(selected_profile, {}) if selected_profile != "none" else {}

    try:
        display_df = apply_mapping(df, active_mapping)
        if use_canonical_excel_names:
            display_df = apply_alias_mapping(display_df, module, CONFIG_DIR)
    except ValueError as e:
        st.error(f"Cannot apply mapping profile '{selected_profile}': {e}")
        display_df = None

    if display_df is not None:
        st.markdown("### Preview")
        st.dataframe(display_df.head(int(preview_rows)), use_container_width=True)

    with st.expander("🔍 Uniqueness check (columns marked `unique: true`)"):
        unique_cols = [c["name"] for c in schema.get("columns", []) if c.get("unique")]
        if not unique_cols:
            st.caption("No columns marked unique in this schema.")
        else:
            check_rows = []
            for c in unique_cols:
                if c in df.columns:
                    nu = df[c].nunique()
                    check_rows.append({"column": c, "unique_values": nu, "total_rows": len(df), "status": "✅ OK" if nu == len(df) else "❌ DUPLICATES"})
            st.dataframe(pd.DataFrame(check_rows), hide_index=True, use_container_width=True)

    st.markdown("### Download")
    mime_map = {"csv": "text/csv", "parquet": "application/octet-stream", "json": "application/json", "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
    extra_profile_names = [name for name in profile_options if name != "none" and name != selected_profile]
    extra_profiles = st.multiselect(
        "Additional profile downloads",
        extra_profile_names,
        key=f"extra_mapping_downloads_{module}",
    )
    download_profiles = [selected_profile, *extra_profiles]

    for download_profile in download_profiles:
        if download_profile in st.session_state.alias_profile_bundles.get(module, {}):
            mapping = st.session_state.alias_profile_bundles[module][download_profile]
        else:
            mapping = profiles.get(download_profile, {}) if download_profile != "none" else {}
        try:
            output_df = apply_mapping(df, mapping)
            if use_canonical_excel_names:
                output_df = apply_alias_mapping(output_df, module, CONFIG_DIR)
            logger.info("Prepared download profile '%s' for module '%s' with %d columns", download_profile, module, len(output_df.columns))
        except ValueError as e:
            st.error(f"Cannot apply mapping profile '{download_profile}': {e}")
            continue

        buf = io.BytesIO()
        if out_format == "csv":
            buf.write(output_df.to_csv(index=False).encode("utf-8"))
        elif out_format == "parquet":
            output_df.to_parquet(buf, index=False)
        elif out_format == "json":
            buf.write(output_df.to_json(orient="records", lines=True).encode("utf-8"))
        elif out_format == "xlsx":
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                output_df.to_excel(writer, index=False)
        buf.seek(0)

        suffix = "" if download_profile == "none" else f"_{download_profile}"
        fname = f"{module}_{len(df)}rows{suffix}.{out_format}"
        st.download_button(
            label=f"⬇️ Download {fname}",
            data=buf,
            file_name=fname,
            mime=mime_map[out_format],
            use_container_width=True,
            key=f"download_{module}_{download_profile}_{out_format}",
        )
