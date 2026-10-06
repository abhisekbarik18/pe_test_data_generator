#!/usr/bin/env python3
"""
CLI for the Private Equity Test Data Generator.

Examples:
    python generate.py --module investors --rows 1000000 --output investors.csv
    python generate.py --module vendors --rows 50000 --output vendors.parquet
    python generate.py --schema config/affiliates.yaml --rows 1000000 --output affiliates.csv --seed 7
    python generate.py --module investors --rows 1000000 --output investors.csv --check-unique
"""
import argparse
import re
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.engine import load_schema, generate_dataframe, write_output
from src.mappings import apply_alias_mapping, apply_mapping, get_mapping_profile, validate_mapping
from src.pe_generators import generate_linked_datasets

CONFIG_DIR = Path(__file__).parent / "config"


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic PE-domain test data.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--module", choices=["investors", "vendors", "affiliates", "general_ledger", "commitments"],
                        help="Built-in module to generate.")
    group.add_argument("--schema", help="Path to a custom YAML schema file.")

    parser.add_argument("--rows", type=int, required=True, help="Number of rows to generate.")
    parser.add_argument("--output", required=True, help="Output file path (.csv/.parquet/.json/.xlsx).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    parser.add_argument("--format", default=None, help="Force output format (overrides file extension).")
    parser.add_argument("--mapping", action="append", default=[], metavar="PROFILE",
                        help="Mapping profile to apply; repeat to write multiple aliased outputs.")
    parser.add_argument("--canonical-names", action="store_true",
                        help="Rename output columns to the canonical field names in config/private_equity_attribute_aliases_75.json.")
    parser.add_argument("--check-unique", action="store_true",
                         help="After generation, verify every column marked unique:true has no duplicates.")

    args = parser.parse_args()
    schema_path = str(CONFIG_DIR / f"{args.module}.yaml") if args.module else args.schema

    schema = load_schema(schema_path)
    module_name = args.module or schema.get("module") or Path(schema_path).stem
    profile_specs = []
    if args.mapping:
        if len(set(args.mapping)) != len(args.mapping):
            parser.error("Each --mapping profile may only be specified once.")
        schema_columns = [column["name"] for column in schema["columns"]]
        output_path = Path(args.output)
        generated_paths = set()
        for profile in args.mapping:
            try:
                mapping = get_mapping_profile(module_name, profile, CONFIG_DIR)
                validate_mapping(schema_columns, mapping)
            except ValueError as error:
                parser.error(str(error))
            if len(args.mapping) == 1:
                profile_path = output_path
            else:
                safe_profile = re.sub(r"[^A-Za-z0-9_.-]+", "_", profile).strip("._") or "mapping"
                profile_path = output_path.with_name(f"{output_path.stem}_{safe_profile}{output_path.suffix}")
            normalized_path = str(profile_path).casefold()
            if normalized_path in generated_paths:
                parser.error("Mapping profile names produce colliding output file names.")
            generated_paths.add(normalized_path)
            profile_specs.append((profile, mapping, profile_path))

    total_start = time.perf_counter()
    if args.module == "general_ledger":
        linked = generate_linked_datasets(
            record_count=args.rows,
            fund_count=min(5, args.rows),
            investment_count=min(100, args.rows),
            period_count=1,
            reporting_date=date.today(),
            seed=args.seed,
            config_dir=CONFIG_DIR,
        )
        df = linked["gl"]
        gen_time = time.perf_counter() - total_start
    else:
        df, gen_time = generate_dataframe(schema, args.rows, seed=args.seed)
    write_time = 0.0
    output_paths = []

    if args.mapping:
        for profile, mapping, profile_path in profile_specs:
            mapped_df = apply_mapping(df, mapping)
            if args.canonical_names:
                mapped_df = apply_alias_mapping(mapped_df, module_name, CONFIG_DIR)
            write_time += write_output(mapped_df, str(profile_path), fmt=args.format)
            output_paths.append(str(profile_path))
    else:
        if args.canonical_names:
            df = apply_alias_mapping(df, module_name, CONFIG_DIR)
        write_time = write_output(df, args.output, fmt=args.format)
        output_paths.append(args.output)

    if args.check_unique:
        unique_cols = [c["name"] for c in schema["columns"] if c.get("unique")]
        print(f"\nUniqueness check ({len(unique_cols)} column(s) marked unique):")
        all_ok = True
        for c in unique_cols:
            nu = df[c].nunique()
            ok = nu == len(df)
            all_ok &= ok
            print(f"  {'OK ' if ok else 'FAIL'}  {c:35s} unique={nu:,} / {len(df):,}")
        sys.exit(0 if all_ok else 1)
    total_seconds = time.perf_counter() - total_start
    print("\nSummary:")
    print(f"  rows: {len(df)}")
    print(f"  columns: {len(df.columns)}")
    print(f"  generation_seconds: {gen_time:.2f}")
    print(f"  write_seconds: {write_time:.2f}")
    print(f"  total_seconds: {total_seconds:.2f}")
    print(f"  output_path(s): {', '.join(output_paths)}")
    if args.rows >= 1_000_000:
        verdict = "PASS" if total_seconds <= 120 else "FAIL"
        print(f"NFR check (1M rows <= 120s): {total_seconds:.2f}s -> {verdict}")


if __name__ == "__main__":
    main()
