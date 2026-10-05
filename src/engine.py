from __future__ import annotations
import time
import yaml
import pandas as pd
from pathlib import Path
from . import generators as g


def load_schema(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_dataframe(schema: dict, n: int, seed: int | None = None, verbose: bool = True) -> tuple[pd.DataFrame, float]:
    """Generate `n` rows of synthetic data for the given schema dict.
    Returns (dataframe, elapsed_seconds).
    """
    rng = g.get_rng(seed)
    columns: dict[str, "object"] = {}
    t0 = time.perf_counter()

    for col in schema["columns"]:
        name = col["name"]
        ctype = col["type"]
        params = {k: v for k, v in col.items() if k not in ("name", "type")}

        if ctype == "email_from":
            ref_col = params.pop("source_column")
            if ref_col not in columns:
                raise ValueError(
                    f"Column '{name}' of type email_from references '{ref_col}', "
                    f"which must be defined earlier in the schema."
                )
            columns[name] = g.gen_email_from_pool(n, rng, columns[ref_col], **params)
            continue

        if ctype == "conditional_amount":
            amount_col = params.pop("amount_column")
            condition_col = params.pop("condition_column")
            if amount_col not in columns or condition_col not in columns:
                raise ValueError(
                    f"Column '{name}' of type conditional_amount references '{amount_col}' "
                    f"and '{condition_col}', which must be defined earlier in the schema."
                )
            columns[name] = g.gen_conditional_amount(
                n, rng, columns[amount_col], columns[condition_col], **params
            )
            continue

        fn = g.GENERATOR_MAP.get(ctype)
        if fn is None:
            raise ValueError(f"Unknown column type '{ctype}' for column '{name}'")
        columns[name] = fn(n, rng, **params)

    df = pd.DataFrame(columns)
    elapsed = time.perf_counter() - t0

    if verbose:
        rate = n / elapsed if elapsed > 0 else float("inf")
        print(f"[engine] Generated {n:,} rows x {len(columns)} cols in {elapsed:.2f}s ({rate:,.0f} rows/sec)")

    return df, elapsed


def write_output(df: pd.DataFrame, output_path: str, fmt: str | None = None, verbose: bool = True) -> float:
    t0 = time.perf_counter()
    fmt = fmt or Path(output_path).suffix.lstrip(".").lower()

    if fmt == "csv":
        df.to_csv(output_path, index=False)
    elif fmt == "parquet":
        df.to_parquet(output_path, index=False)
    elif fmt == "json":
        df.to_json(output_path, orient="records", lines=True)
    elif fmt in ("xlsx", "excel"):
        df.to_excel(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {fmt}")

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"[engine] Wrote {output_path} ({fmt}) in {elapsed:.2f}s")
    return elapsed


def run(schema_path: str, n: int, output_path: str, seed: int | None = None, fmt: str | None = None) -> dict:
    schema = load_schema(schema_path)
    df, gen_time = generate_dataframe(schema, n, seed=seed)
    write_time = write_output(df, output_path, fmt=fmt)
    return {
        "rows": n,
        "columns": len(df.columns),
        "generation_seconds": round(gen_time, 2),
        "write_seconds": round(write_time, 2),
        "total_seconds": round(gen_time + write_time, 2),
        "output_path": output_path,
    }
