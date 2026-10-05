"""
Vectorized generator functions for fast synthetic data creation, with
support for GUARANTEED uniqueness per column.

Uniqueness strategy (all O(n), no python-level dedupe/retry loops):
  - sequence_id / uuid: unique by construction.
  - int_range / float_range (unique=True): the numeric range is treated as
    a discrete integer space of size `space`; we draw `n` unique integers
    via rng.choice(space, n, replace=False) (numpy does this in O(n) even
    for huge `space` -- benchmarked at ~0.07s for 1M draws from a
    30-billion item space) and map them back to values. Raises a clear
    error if the requested row count exceeds the space.
  - pattern (unique=True): the pattern's digit/letter slots define a
    discrete space (e.g. "##-#######" = 10^9 combinations). We draw `n`
    unique integers from that space and decode each one via mixed-radix
    decomposition into the digit/letter slots. Guarantees exact format,
    zero collisions, no retries.
  - name_pool (unique=True): sampling realistic names with replacement
    from a pool WILL collide heavily at 1M rows (pool sizes are in the
    thousands). To guarantee uniqueness while keeping values realistic,
    we append a short unique per-row code (built from a random permutation
    of 0..n-1, so the suffix itself is unique and unpredictable) to the
    pooled base value. The pooled part can repeat; the full string cannot.
  - email_from (always effectively unique): same permutation-code
    technique folded into the local-part of the email.
  - choice (unique=True): only possible when len(values) >= n (a
    categorical field with k possible values cannot hold more than k
    unique rows). Uses true sampling without replacement; raises a clear
    error otherwise.
"""
from __future__ import annotations
import numpy as np
from faker import Faker
from datetime import datetime, timedelta

_fake = Faker()
Faker.seed(42)

_POOL_CACHE: dict[str, np.ndarray] = {}


def _build_pool(kind: str, size: int) -> np.ndarray:
    key = f"{kind}:{size}"
    if key in _POOL_CACHE:
        return _POOL_CACHE[key]

    if kind == "company":
        vals = [_fake.company() for _ in range(size)]
    elif kind == "person":
        vals = [_fake.name() for _ in range(size)]
    elif kind == "first_name":
        vals = [_fake.first_name() for _ in range(size)]
    elif kind == "last_name":
        vals = [_fake.last_name() for _ in range(size)]
    elif kind == "street_address":
        vals = [_fake.street_address() for _ in range(size)]
    elif kind == "city":
        vals = [_fake.city() for _ in range(size)]
    elif kind == "state":
        vals = [_fake.state() for _ in range(size)]
    elif kind == "country":
        vals = [_fake.country() for _ in range(size)]
    elif kind == "email_domain":
        vals = [_fake.free_email_domain() for _ in range(size)]
    elif kind == "bank_name":
        vals = [f"{_fake.last_name()} {_fake.random_element(['Bank','Trust','Capital','Financial'])}" for _ in range(size)]
    elif kind == "zip_code":
        vals = [_fake.postcode() for _ in range(size)]
    else:
        raise ValueError(f"Unknown pool kind: {kind}")

    arr = np.array(vals, dtype=object)
    _POOL_CACHE[key] = arr
    return arr


def get_rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _unique_codes(n: int, rng) -> np.ndarray:
    """A permutation of 0..n-1 as zero-padded strings -- guaranteed unique,
    guaranteed unpredictable order (not sequential-looking)."""
    width = max(len(str(n - 1)), 4)
    perm = rng.permutation(n)
    return np.array([f"{p:0{width}d}" for p in perm], dtype=object)


def _draw_ints(n: int, space: int, rng, unique: bool, field_label: str) -> np.ndarray:
    if unique:
        if space < n:
            raise ValueError(
                f"Column '{field_label}' was marked unique=True but its value space only "
                f"has {space:,} possible values, which is fewer than the {n:,} rows requested. "
                f"Widen the range / pattern, or set unique: false."
            )
        return rng.choice(space, size=n, replace=False)
    return rng.integers(0, space, size=n)


def gen_sequence_id(n: int, rng, prefix: str = "ID", width: int = 7, start: int = 1) -> np.ndarray:
    nums = np.arange(start, start + n)
    return np.array([f"{prefix}{num:0{width}d}" for num in nums], dtype=object)


def gen_uuid(n: int, rng) -> np.ndarray:
    perm = rng.permutation(n)
    rand_high = rng.integers(0, 2**63 - 1, size=n, dtype=np.int64)
    return np.array([f"{int(h):016x}-{int(p):08x}" for h, p in zip(rand_high, perm)], dtype=object)


def gen_name_pool(n: int, rng, pool: str = "company", pool_size: int = 20000,
                   unique: bool = False, suffix_sep: str = " ") -> np.ndarray:
    pool_arr = _build_pool(pool, pool_size)
    base = rng.choice(pool_arr, size=n, replace=True)
    if not unique:
        return base
    codes = _unique_codes(n, rng)
    return np.array([f"{b}{suffix_sep}{c}" for b, c in zip(base, codes)], dtype=object)


def gen_choice(n: int, rng, values: list, weights: list | None = None, unique: bool = False) -> np.ndarray:
    values_arr = np.array(values, dtype=object)
    if unique:
        if len(values_arr) < n:
            raise ValueError(
                f"A 'choice' column was marked unique=True with only {len(values_arr)} possible "
                f"values, which is fewer than the {n:,} rows requested. Categorical columns can "
                f"only be unique when rows <= number of distinct values."
            )
        return rng.choice(values_arr, size=n, replace=False)
    if weights:
        p = np.array(weights, dtype=float)
        p = p / p.sum()
        return rng.choice(values_arr, size=n, replace=True, p=p)
    return rng.choice(values_arr, size=n, replace=True)


def gen_int_range(n: int, rng, min: int = 0, max: int = 100, unique: bool = False) -> np.ndarray:
    space = max - min + 1
    ids = _draw_ints(n, space, rng, unique, "int_range")
    return ids + min


def gen_float_range(n: int, rng, min: float = 0, max: float = 100, decimals: int = 2,
                     unique: bool = False) -> np.ndarray:
    scale = 10 ** decimals
    space = int(round((max - min) * scale)) + 1
    ids = _draw_ints(n, space, rng, unique, "float_range")
    vals = min + ids / scale
    return np.round(vals, decimals)


def gen_date_range(n: int, rng, start: str = "2015-01-01", end: str = "2026-10-05",
                    fmt: str = "%Y-%m-%d", unique: bool = False) -> np.ndarray:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    total_days = (end_dt - start_dt).days
    space = max(total_days, 1) + 1
    offsets = _draw_ints(n, space, rng, unique, "date_range")
    return np.array([(start_dt + timedelta(days=int(o))).strftime(fmt) for o in offsets], dtype=object)


def gen_boolean(n: int, rng, true_weight: float = 0.5, true_label="Yes", false_label="No") -> np.ndarray:
    picks = rng.random(n) < true_weight
    return np.where(picks, true_label, false_label)


def _pattern_space(pattern: str) -> tuple[int, int, int]:
    n_hash = pattern.count("#")
    n_at = pattern.count("@")
    hash_space = 10 ** n_hash if n_hash else 1
    at_space = 26 ** n_at if n_at else 1
    return hash_space * at_space, hash_space, at_space


def gen_pattern(n: int, rng, pattern: str = "###-####", unique: bool = False) -> np.ndarray:
    n_hash = pattern.count("#")
    n_at = pattern.count("@")
    total, hash_space, at_space = _pattern_space(pattern)

    ids = _draw_ints(n, total, rng, unique, f"pattern '{pattern}'")
    hash_ids = ids % hash_space
    at_ids = ids // hash_space

    out = np.empty(n, dtype=object)
    for i in range(n):
        digit_str = str(int(hash_ids[i])).zfill(n_hash) if n_hash else ""
        val = int(at_ids[i])
        letters = []
        for _ in range(n_at):
            letters.append(chr(65 + val % 26))
            val //= 26
        letters.reverse()

        s = []
        di, li = 0, 0
        for ch in pattern:
            if ch == "#":
                s.append(digit_str[di]); di += 1
            elif ch == "@":
                s.append(letters[li]); li += 1
            else:
                s.append(ch)
        out[i] = "".join(s)
    return out


def gen_email_from_pool(n: int, rng, name_column: np.ndarray, domain_pool_size: int = 500,
                         unique: bool = True) -> np.ndarray:
    domains = _build_pool("email_domain", domain_pool_size)
    chosen_domains = rng.choice(domains, size=n, replace=True)
    codes = _unique_codes(n, rng) if unique else rng.integers(1, 999999, size=n).astype(str)

    out = np.empty(n, dtype=object)
    for i in range(n):
        base = str(name_column[i]).lower()
        base = "".join(c if c.isalnum() else "." for c in base).strip(".")
        base = ".".join([p for p in base.split(".") if p])[:30] or "user"
        out[i] = f"{base}.{codes[i]}@{chosen_domains[i]}"
    return out


def gen_conditional_amount(n: int, rng, amount_values: np.ndarray,
                           condition_values: np.ndarray, equals: str,
                           zero_value: float = 0) -> np.ndarray:
    return np.where(condition_values == equals, amount_values, zero_value)


def gen_phone(n: int, rng, pattern: str = "+1-###-###-####", unique: bool = False) -> np.ndarray:
    return gen_pattern(n, rng, pattern, unique=unique)


GENERATOR_MAP = {
    "sequence_id": gen_sequence_id,
    "uuid": gen_uuid,
    "name_pool": gen_name_pool,
    "choice": gen_choice,
    "int_range": gen_int_range,
    "float_range": gen_float_range,
    "date_range": gen_date_range,
    "boolean": gen_boolean,
    "pattern": gen_pattern,
    "phone": gen_phone,
    "email_from": gen_email_from_pool,
}
