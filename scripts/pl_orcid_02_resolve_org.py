from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process


@dataclass(frozen=True)
class OrgRecord:
    ror_id: str
    canonical_name_norm: str
    country_codes: tuple[str, ...]
    geonames_ids: tuple[str, ...]
    alias_ror: tuple[str, ...]
    acronym_ror: tuple[str, ...]
    alias_ringgold: tuple[str, ...]
    alias_isni: tuple[str, ...]


def norm_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    text = " ".join(text.split())
    return text or None


def norm_isni(value: Any) -> str | None:
    text = norm_text(value)
    if text is None:
        return None
    text = text.replace(" ", "")
    return text or None


def split_alt_names(value: Any) -> list[str]:
    text = norm_text(value)
    if text is None:
        return []
    parts = [norm_text(x) for x in text.split(";")]
    return [x for x in parts if x]


def choose_canonical_name(names: list[dict[str, Any]]) -> str | None:
    ranked: list[tuple[int, str]] = []
    for item in names:
        value = norm_text(item.get("value"))
        if value is None:
            continue
        types = set(item.get("types") or [])
        if "ror_display" in types:
            ranked.append((0, value))
        elif "label" in types:
            ranked.append((1, value))
        else:
            ranked.append((2, value))
    if not ranked:
        return None
    ranked.sort(key=lambda x: (x[0], len(x[1]), x[1]))
    return ranked[0][1]


def parse_ror_external_ids(external_ids: list[dict[str, Any]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for item in external_ids or []:
        ext_type = norm_text(item.get("type"))
        if ext_type is None:
            continue
        values = set()
        preferred = item.get("preferred")
        preferred_norm = (
            norm_text(preferred) if ext_type != "isni" else norm_isni(preferred)
        )
        if preferred_norm is not None:
            values.add(preferred_norm)
        for raw in item.get("all") or []:
            value = norm_text(raw) if ext_type != "isni" else norm_isni(raw)
            if value is not None:
                values.add(value)
        if values:
            result[ext_type] = values
    return result


def load_isni_aliases(
    isni_tsv_path: Path,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    by_isni_name: dict[str, set[str]] = {}
    by_isni_ringgold: dict[str, set[str]] = {}
    with isni_tsv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            isni = norm_isni(row.get("isni"))
            if isni is None:
                continue
            names = set()
            name = norm_text(row.get("name"))
            if name is not None:
                names.add(name)
            for alt in split_alt_names(row.get("alt_names")):
                names.add(alt)
            if names:
                by_isni_name.setdefault(isni, set()).update(names)
            ringgold = norm_text(row.get("ringgold"))
            if ringgold is not None:
                by_isni_ringgold.setdefault(isni, set()).add(ringgold)
    return by_isni_name, by_isni_ringgold


def load_ringgold_names(ids_tsv_path: Path) -> dict[str, set[str]]:
    by_ringgold: dict[str, set[str]] = {}
    with ids_tsv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ringgold_id = norm_text(row.get("ringgold_id"))
            if ringgold_id is None:
                continue
            names = set()
            name = norm_text(row.get("name"))
            if name is not None:
                names.add(name)
            locality = norm_text(row.get("locality"))
            if locality is not None and name is not None:
                names.add(f"{name} {locality}")
            if names:
                by_ringgold.setdefault(ringgold_id, set()).update(names)
    return by_ringgold


def build_org_records(
    ror_json_path: Path,
    ringgold_ids_path: Path,
    isni_tsv_path: Path,
) -> list[OrgRecord]:
    by_isni_name, by_isni_ringgold = load_isni_aliases(isni_tsv_path)
    by_ringgold_name = load_ringgold_names(ringgold_ids_path)

    with ror_json_path.open("r", encoding="utf-8") as f:
        ror_items = json.load(f)

    records: list[OrgRecord] = []

    for item in ror_items:
        ror_id = norm_text(item.get("id"))
        if ror_id is None:
            continue

        names = item.get("names") or []
        canonical_name = choose_canonical_name(names)
        if canonical_name is None:
            continue

        alias_ror: set[str] = set()
        acronym_ror: set[str] = set()

        for name_item in names:
            value = norm_text(name_item.get("value"))
            if value is None:
                continue
            types = set(name_item.get("types") or [])
            if value == canonical_name:
                continue
            if "acronym" in types:
                acronym_ror.add(value)
            else:
                alias_ror.add(value)

        external_ids = parse_ror_external_ids(item.get("external_ids") or [])

        alias_isni: set[str] = set()
        alias_ringgold: set[str] = set()

        for isni in external_ids.get("isni", set()):
            alias_isni.update(by_isni_name.get(isni, set()))
            for ringgold_id in by_isni_ringgold.get(isni, set()):
                alias_ringgold.update(by_ringgold_name.get(ringgold_id, set()))

        country_codes: set[str] = set()
        geonames_ids: set[str] = set()

        for loc in item.get("locations") or []:
            geonames_id = norm_text(loc.get("geonames_id"))
            if geonames_id is not None:
                geonames_ids.add(geonames_id)
            details = loc.get("geonames_details") or {}
            country_code = norm_text(details.get("country_code"))
            if country_code is not None:
                country_codes.add(country_code)

        records.append(
            OrgRecord(
                ror_id=ror_id,
                canonical_name_norm=canonical_name,
                country_codes=tuple(sorted(country_codes)),
                geonames_ids=tuple(sorted(geonames_ids)),
                alias_ror=tuple(sorted(alias_ror)),
                acronym_ror=tuple(sorted(acronym_ror)),
                alias_ringgold=tuple(sorted(alias_ringgold)),
                alias_isni=tuple(sorted(alias_isni)),
            )
        )

    return records


def build_bucket(records: list[OrgRecord]) -> dict[str, Any]:
    exact: dict[str, list[int]] = {}
    alias_ror: dict[str, list[int]] = {}
    acronym_ror: dict[str, list[int]] = {}
    alias_ringgold: dict[str, list[int]] = {}
    alias_isni: dict[str, list[int]] = {}

    for idx, record in enumerate(records):
        exact.setdefault(record.canonical_name_norm, []).append(idx)
        for value in record.alias_ror:
            alias_ror.setdefault(value, []).append(idx)
        for value in record.acronym_ror:
            acronym_ror.setdefault(value, []).append(idx)
        for value in record.alias_ringgold:
            alias_ringgold.setdefault(value, []).append(idx)
        for value in record.alias_isni:
            alias_isni.setdefault(value, []).append(idx)

    return {
        "records": records,
        "exact": exact,
        "alias_ror": alias_ror,
        "acronym_ror": acronym_ror,
        "alias_ringgold": alias_ringgold,
        "alias_isni": alias_isni,
        "canonical_choices": sorted(exact.keys(), key=lambda x: (len(x), x)),
        "alias_ror_choices": sorted(alias_ror.keys(), key=lambda x: (len(x), x)),
        "acronym_ror_choices": sorted(acronym_ror.keys(), key=lambda x: (len(x), x)),
        "alias_ringgold_choices": sorted(
            alias_ringgold.keys(), key=lambda x: (len(x), x)
        ),
        "alias_isni_choices": sorted(alias_isni.keys(), key=lambda x: (len(x), x)),
    }


def build_index(
    ror_json_path: Path,
    ringgold_ids_path: Path,
    isni_tsv_path: Path,
) -> dict[str, Any]:
    records = build_org_records(ror_json_path, ringgold_ids_path, isni_tsv_path)

    country_city_map: dict[tuple[str, str], list[OrgRecord]] = {}
    country_map: dict[str, list[OrgRecord]] = {}

    for record in records:
        for country_code in record.country_codes:
            country_map.setdefault(country_code, []).append(record)
            for geonames_id in record.geonames_ids:
                country_city_map.setdefault((country_code, geonames_id), []).append(
                    record
                )

    return {
        "global": build_bucket(records),
        "country": {k: build_bucket(v) for k, v in country_map.items()},
        "country_city": {k: build_bucket(v) for k, v in country_city_map.items()},
    }


def choose_record(indices: list[int], records: list[OrgRecord]) -> OrgRecord:
    candidates = [records[i] for i in indices]
    candidates.sort(
        key=lambda x: (len(x.canonical_name_norm), x.canonical_name_norm, x.ror_id)
    )
    return candidates[0]


def resolve_from_map(
    value: str,
    mapping: dict[str, list[int]],
    records: list[OrgRecord],
    method: str,
) -> tuple[str | None, str | None, str, float | None]:
    indices = mapping.get(value)
    if not indices:
        return None, None, "no_match", None
    record = choose_record(indices, records)
    return record.ror_id, record.canonical_name_norm, method, 1.0


def resolve_fuzzy_from_map(
    value: str,
    choices: list[str],
    mapping: dict[str, list[int]],
    records: list[OrgRecord],
    method: str,
) -> tuple[str | None, str | None, str, float | None]:
    if not choices:
        return None, None, "no_match", None
    match = process.extractOne(value, choices, scorer=fuzz.WRatio)
    if match is None:
        return None, None, "no_match", None
    matched_value, score, _ = match
    if score <= 0:
        return None, None, "no_match", None
    indices = mapping.get(matched_value)
    if not indices:
        return None, None, "no_match", None
    record = choose_record(indices, records)
    return record.ror_id, record.canonical_name_norm, method, float(score) / 100.0


def resolve_in_bucket(
    org_raw: str, bucket: dict[str, Any]
) -> tuple[str | None, str | None, str, float | None]:
    records = bucket["records"]

    result = resolve_from_map(org_raw, bucket["exact"], records, "exact")
    if result[2] != "no_match":
        return result

    result = resolve_from_map(org_raw, bucket["alias_ror"], records, "alias_ror")
    if result[2] != "no_match":
        return result

    result = resolve_from_map(org_raw, bucket["acronym_ror"], records, "acronym_ror")
    if result[2] != "no_match":
        return result

    result = resolve_from_map(
        org_raw, bucket["alias_ringgold"], records, "alias_ringgold"
    )
    if result[2] != "no_match":
        return result

    result = resolve_from_map(org_raw, bucket["alias_isni"], records, "alias_isni")
    if result[2] != "no_match":
        return result

    result = resolve_fuzzy_from_map(
        org_raw, bucket["canonical_choices"], bucket["exact"], records, "fuzzy"
    )
    if result[2] != "no_match":
        return result

    result = resolve_fuzzy_from_map(
        org_raw,
        bucket["alias_ror_choices"],
        bucket["alias_ror"],
        records,
        "fuzzy_alias_ror",
    )
    if result[2] != "no_match":
        return result

    result = resolve_fuzzy_from_map(
        org_raw,
        bucket["acronym_ror_choices"],
        bucket["acronym_ror"],
        records,
        "fuzzy_acronym_ror",
    )
    if result[2] != "no_match":
        return result

    result = resolve_fuzzy_from_map(
        org_raw,
        bucket["alias_ringgold_choices"],
        bucket["alias_ringgold"],
        records,
        "fuzzy_alias_ringgold",
    )
    if result[2] != "no_match":
        return result

    result = resolve_fuzzy_from_map(
        org_raw,
        bucket["alias_isni_choices"],
        bucket["alias_isni"],
        records,
        "fuzzy_alias_isni",
    )
    if result[2] != "no_match":
        return result

    return None, None, "no_match", None


def resolve_org(
    org_raw: Any,
    country_raw: Any,
    city_geonames_id_raw: Any,
    org_index: dict[str, Any],
) -> tuple[str | None, str | None, str, float | None]:
    org = norm_text(org_raw)
    country = norm_text(country_raw)
    city_geonames_id = norm_text(city_geonames_id_raw)

    if org is None:
        return None, None, "missing_org", None

    if country is not None and city_geonames_id is not None:
        bucket = org_index["country_city"].get((country, city_geonames_id))
        if bucket is not None and bucket["records"]:
            result = resolve_in_bucket(org, bucket)
            if result[2] != "no_match":
                return result

    if country is not None:
        bucket = org_index["country"].get(country)
        if bucket is not None and bucket["records"]:
            result = resolve_in_bucket(org, bucket)
            if result[2] != "no_match":
                return result

    return resolve_in_bucket(org, org_index["global"])


def resolve_part(
    input_path: Path,
    output_path: Path,
    org_index: dict[str, Any],
) -> None:
    df = pd.read_parquet(input_path).copy()

    if "org_from" in df.columns:
        df = df.rename(columns={"org_from": "org_from_raw"})
    if "org_to" in df.columns:
        df = df.rename(columns={"org_to": "org_to_raw"})

    from_results = [
        resolve_org(org, country, city_id, org_index)
        for org, country, city_id in zip(
            df["org_from_raw"],
            df["org_country_from"],
            df["org_city_from_geonames_id"],
            strict=False,
        )
    ]

    to_results = [
        resolve_org(org, country, city_id, org_index)
        for org, country, city_id in zip(
            df["org_to_raw"],
            df["org_country_to"],
            df["org_city_to_geonames_id"],
            strict=False,
        )
    ]

    df["org_from_ror_id"] = [x[0] for x in from_results]
    df["org_from_ror_name"] = [x[1] for x in from_results]
    df["org_from_ror_method"] = [x[2] for x in from_results]
    df["org_from_ror_confidence"] = [x[3] for x in from_results]

    df["org_to_ror_id"] = [x[0] for x in to_results]
    df["org_to_ror_name"] = [x[1] for x in to_results]
    df["org_to_ror_method"] = [x[2] for x in to_results]
    df["org_to_ror_confidence"] = [x[3] for x in to_results]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def iter_part_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*.parquet"))


def run_one_part(
    input_path_str: str,
    output_path_str: str,
    org_index: dict[str, Any],
) -> str:
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)
    resolve_part(
        input_path=input_path,
        output_path=output_path,
        org_index=org_index,
    )
    return input_path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/orcid/01_resolve_city"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/orcid/02_resolve_org"),
    )
    parser.add_argument(
        "--ror-json-path",
        type=Path,
        default=Path("data/ror/ror.json"),
    )
    parser.add_argument(
        "--ringgold-ids-path",
        type=Path,
        default=Path("data/ringgold/ids.tsv"),
    )
    parser.add_argument(
        "--ringgold-isni-path",
        type=Path,
        default=Path("data/ringgold/isni.tsv"),
    )
    parser.add_argument(
        "--part",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    org_index = build_index(
        ror_json_path=args.ror_json_path,
        ringgold_ids_path=args.ringgold_ids_path,
        isni_tsv_path=args.ringgold_isni_path,
    )

    if args.part is not None:
        input_files = [args.input_dir / f"{args.part}.parquet"]
    else:
        input_files = iter_part_files(args.input_dir)

    tasks = [
        (str(input_path), str(args.output_dir / input_path.name))
        for input_path in input_files
    ]

    if len(tasks) == 1:
        name = run_one_part(tasks[0][0], tasks[0][1], org_index)
        print(f"done {name}")
        return

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(run_one_part, input_path_str, output_path_str, org_index)
            for input_path_str, output_path_str in tasks
        ]
        for future in as_completed(futures):
            name = future.result()
            print(f"done {name}")


if __name__ == "__main__":
    main()
