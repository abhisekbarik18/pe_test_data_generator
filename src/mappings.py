from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger("pe_testdata_generator")


def mapping_file(module: str, config_dir: Path) -> Path:
    return config_dir / f"{module}_mappings.yaml"


def alias_catalog_file(config_dir: Path) -> Path:
    return config_dir / "private_equity_attribute_aliases_75.json"


def normalize_alias_token(value: str) -> str:
    cleaned = value.strip().lower()
    cleaned = cleaned.replace(" ", "_").replace("-", "_").replace("/", "_")
    cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def to_excel_column_name(column: str) -> str:
    if not column:
        return column
    parts = [part for part in re.split(r"[_\s-]+", str(column).strip()) if part]
    if not parts:
        return str(column)

    tokens = []
    for index, part in enumerate(parts):
        if part.lower() in {"id", "ids"}:
            tokens.append("ID")
        else:
            title = part.capitalize()
            if index == 0 and title.lower() in {"s", "no"}:
                title = "S.No" if part.lower() == "s" else part.upper()
            tokens.append(title)

    name = " ".join(tokens)
    return name.replace(" Id ", " ID ").replace(" Id", " ID").replace(" No ", " No ")


def load_alias_catalog(config_dir: Path) -> dict:
    path = alias_catalog_file(config_dir)
    if not path.exists():
        return {"modules": {}}

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Alias catalog '{path}' must contain a JSON object.")
    modules = data.get("modules", {})
    if not isinstance(modules, dict):
        raise ValueError(f"Alias catalog '{path}' must contain a 'modules' dictionary.")
    return data


def derive_excel_name_mapping(module: str, columns: list[str], config_dir: Path) -> dict[str, str]:
    catalog = load_alias_catalog(config_dir)
    module_aliases = catalog.get("modules", {}).get(module, {})
    if not isinstance(module_aliases, dict):
        return {column: column for column in columns}

    alias_candidates: dict[str, set[str]] = {}
    for canonical, aliases in module_aliases.items():
        if not isinstance(aliases, list):
            continue
        readable_name = to_excel_column_name(canonical)
        for alias in aliases:
            if not isinstance(alias, str):
                continue
            key = normalize_alias_token(alias)
            alias_candidates.setdefault(key, set()).add(readable_name)

    reverse_lookup: dict[str, str] = {}
    for alias_key, candidate_names in alias_candidates.items():
        if len(candidate_names) == 1:
            reverse_lookup[alias_key] = next(iter(candidate_names))

    mapping: dict[str, str] = {}
    for column in columns:
        lookup_key = normalize_alias_token(str(column))
        if lookup_key in reverse_lookup:
            mapping[column] = reverse_lookup[lookup_key]
        else:
            mapping[column] = to_excel_column_name(str(column))
    return mapping


def unique_column_names(columns: list[str]) -> list[str]:
    counts = Counter(columns)
    seen: set[str] = set()
    unique_columns: list[str] = []
    for name in columns:
        if counts[name] == 1:
            unique_columns.append(name)
            seen.add(name)
            continue

        base = name
        suffix = 2
        while base in seen:
            base = f"{name} ({suffix})"
            suffix += 1
        unique_columns.append(base)
        seen.add(base)
    return unique_columns


def load_mapping_profiles(module: str, config_dir: Path) -> dict[str, dict[str, str]]:
    path = mapping_file(module, config_dir)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError(f"Mapping file '{path}' must contain a 'profiles' dictionary.")

    for name, mapping in profiles.items():
        if not isinstance(mapping, dict) or not all(
            isinstance(source, str) and isinstance(alias, str)
            for source, alias in mapping.items()
        ):
            raise ValueError(f"Mapping profile '{name}' in '{path}' must be a string-to-string dictionary.")
    return profiles


def validate_mapping(columns: list[str], mapping: dict[str, str]) -> None:
    if not isinstance(mapping, dict) or not all(
        isinstance(source, str) and isinstance(alias, str) and alias.strip()
        for source, alias in mapping.items()
    ):
        raise ValueError("A mapping must contain non-empty alias names for each source column.")

    unknown = sorted(set(mapping) - set(columns))
    if unknown:
        raise ValueError(f"Mapping references column(s) not in the current schema: {', '.join(unknown)}")

    output_columns = [mapping.get(column, column) for column in columns]
    duplicates = sorted(column for column, count in Counter(output_columns).items() if count > 1)
    if duplicates:
        raise ValueError(f"Mapping creates duplicate output column name(s): {', '.join(duplicates)}")


def apply_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    validate_mapping(list(df.columns), mapping)
    return df.rename(columns=mapping)


def save_mapping_profile(
    module: str,
    profile: str,
    mapping: dict[str, str],
    columns: list[str],
    config_dir: Path,
) -> None:
    profile = profile.strip()
    if not profile:
        raise ValueError("Enter a name for the mapping profile.")
    if profile.lower() == "none":
        raise ValueError("'none' is reserved for the original column names.")

    validate_mapping(columns, mapping)
    profiles = load_mapping_profiles(module, config_dir)
    profiles[profile] = mapping
    path = mapping_file(module, config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump({"profiles": profiles}, file, sort_keys=False, allow_unicode=True)


def get_mapping_profile(module: str, profile: str, config_dir: Path) -> dict[str, str]:
    profiles = load_mapping_profiles(module, config_dir)
    if profile not in profiles:
        raise ValueError(f"Mapping profile '{profile}' was not found for module '{module}'.")
    return profiles[profile]


def build_alias_profile_bundle(module: str, columns: list[str], config_dir: Path) -> dict[str, dict[str, str]]:
    alias_mapping = derive_excel_name_mapping(module, columns, config_dir)
    canonical = {column: alias_mapping.get(column, column) for column in columns}

    business_keys = {
        "id", "name", "type", "status", "country", "city", "state", "email",
        "phone", "date", "currency", "bank", "amount", "risk", "rating"
    }
    business_view = {}
    for column in columns:
        normalized = normalize_alias_token(str(column))
        if any(token in normalized for token in business_keys):
            business_view[column] = alias_mapping.get(column, column)

    profiles: dict[str, dict[str, str]] = {
        "canonical_excel": canonical,
    }
    if business_view:
        profiles["business_view"] = business_view
    return profiles


def apply_alias_mapping(df: pd.DataFrame, module: str, config_dir: Path) -> pd.DataFrame:
    mapping = derive_excel_name_mapping(module, list(df.columns), config_dir)
    renamed = df.rename(columns=mapping)
    renamed.columns = unique_column_names(list(renamed.columns))
    logger.info("Applied alias mapping for module '%s' => %s", module, list(renamed.columns))
    return renamed
