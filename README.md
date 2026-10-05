# PE Test Data Generator

Synthetic test data generator for a Private Equity domain data model.
Ships with three ready-made modules — **investors**, **vendors**,
**affiliates** — and is built so more modules (**commitments**,
**investments**, ...) can be added as plain YAML files, no code changes.

## Highlights

- **Different, fully independent column sets per module** — each module is
  a YAML schema; add/remove/rename columns freely, per module.
- **Guaranteed-unique values** where you ask for them (`unique: true` on a
  column) — IDs, names, emails, phone numbers, tax IDs, account numbers,
  amounts, etc. come out with **zero duplicates**, verified at 1,000,000
  rows (see Benchmarks below). Low-cardinality categorical fields (status,
  country, currency, type...) are intentionally left non-unique — with
  only a handful of possible values, "unique" is mathematically impossible
  once row count exceeds the number of categories, so these stay
  realistic/weighted instead.
- **NFR: 1,000,000 rows in under 2 minutes** — met with large margin (see
  below) via a fully vectorized NumPy engine. No per-row Faker calls, no
  duplicate-and-retry loops for uniqueness.
- **Two front ends**: a Streamlit web app for interactive use, and a CLI
  for scripted/CI use.

## Quick start

```bash
pip install -r requirements.txt

# Web app
streamlit run app.py

# CLI
python generate.py --module investors --rows 1000000 --output investors.csv
python generate.py --module vendors --rows 50000 --output vendors.parquet
python generate.py --module affiliates --rows 1000000 --output affiliates.csv --check-unique
```

## How uniqueness is guaranteed (no retries, no slowdowns)

A naive "generate randomly, dedupe, retry on collision" approach gets
catastrophically slow as you approach 1M rows from a limited pool. Instead:

| Column type | Technique |
|---|---|
| `sequence_id` | Unique by construction (counter). |
| `uuid` | A random permutation of row indices + a random component — unique by construction. |
| `pattern` (e.g. tax IDs, account numbers, SWIFT codes) | The pattern's digit/letter slots define an exact combinatorial space (e.g. `##-#######` = 10^9 combinations). We draw `n` unique integers from that space in one vectorized `numpy` call (`replace=False`, benchmarked at ~0.07s for 1M draws even from a 30-billion-value space) and decode each one back into the pattern's digits/letters. Exact format, zero collisions, no retries. Raises a clear error up front if the pattern can't hold `n` unique values. |
| `int_range` / `float_range` | Same technique: the numeric range is treated as a discrete space and sampled without replacement. |
| `name_pool` (company/person/address names) | Realistic names are drawn from a few-thousand-item pool, which *will* repeat at 1M rows if sampled with replacement. To stay realistic **and** unique, we append a short, randomly-ordered unique code (from a permutation of row indices) to the pooled value. |
| `email_from` | Built from the (now-unique) name column, a domain pool, and a unique per-row code — guaranteed unique regardless of name collisions. |
| `choice` (categorical) | Supports `unique: true` only when the number of listed values is >= row count (true sampling without replacement); otherwise raises a clear, explicit error rather than silently failing. |

## Benchmarks (this environment)

Measured end-to-end (generation + CSV write), all unique-flagged columns
verified to have **zero duplicates** at 1,000,000 rows:

| Module | Columns | Rows | Generation | CSV write | Total | Rows/sec |
|---|---|---|---|---|---|---|
| investors | 27 | 1,000,000 | 33.4s | 11.0s | 44.4s | ~30,000 |
| vendors | 22 | 1,000,000 | 23.9s | 8.7s | 32.5s | ~42,000 |
| affiliates | 18 | 1,000,000 | 23.3s | 7.6s | 30.9s | ~43,000 |

All well under the 120-second (2-minute) NFR, with headroom for larger
row counts or additional columns.

## Editing / adding schemas

Each module is one YAML file. Example column entries:

```yaml
columns:
  - name: investor_id
    type: sequence_id
    prefix: INV
    width: 7

  - name: investor_name
    type: name_pool
    pool: company        # company | person | first_name | last_name | street_address | city | state | country | bank_name | zip_code
    pool_size: 25000
    unique: true          # guarantees no two rows share a value

  - name: investor_type
    type: choice
    values: [Pension Fund, Endowment, Family Office]
    weights: [40, 30, 30] # optional

  - name: aum_usd_millions
    type: float_range
    min: 5
    max: 50000
    decimals: 2
    unique: true

  - name: onboarding_date
    type: date_range
    start: "2010-01-01"
    end: "2026-10-05"

  - name: tax_id
    type: pattern
    pattern: "##-#######"   # '#' = digit, '@' = uppercase letter
    unique: true

  - name: email
    type: email_from
    source_column: investor_name   # must be defined earlier in the file
```

To add a **new module** (e.g. `commitments`, `investments`): copy an
existing YAML file, change `module:` and the `columns:` list, save it as
`config/<module_name>.yaml`, and it's immediately usable via
`--module <name>` is not auto-registered for the CLI's `--module` choices
(those three are the built-ins) — use `--schema config/<name>.yaml`
instead, or add it in the Streamlit app's sidebar ("➕ Add a new module"),
which lets you build and save a new schema directly in the browser.

## Project layout

```
pe_testdata_generator/
├── app.py                 # Streamlit web app
├── generate.py             # CLI
├── requirements.txt
├── config/
│   ├── investors.yaml
│   ├── vendors.yaml
│   └── affiliates.yaml
└── src/
    ├── generators.py       # vectorized, uniqueness-aware generator functions
    └── engine.py            # schema loader + dataframe builder + writers
```
