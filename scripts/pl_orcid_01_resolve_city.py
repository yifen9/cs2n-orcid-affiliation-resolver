from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process


@dataclass(frozen=True)
class GeoRecord:
    geoname_id: str
    name_norm: str
    country_code: str
    population: int


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


def split_alternate_names(value: Any) -> list[str]:
    text = norm_text(value)
    if text is None:
        return []
    parts = [norm_text(x) for x in text.split(",")]
    return [x for x in parts if x]


def to_population(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, float) and math.isnan(value):
        return 0
    try:
        return int(value)
    except Exception:
        return 0


def choose_best(records: list[GeoRecord]) -> GeoRecord:
    return max(records, key=lambda x: (x.population, -len(x.name_norm), x.geoname_id))


def build_geonames_index(geonames_path: Path) -> dict[str, dict[str, Any]]:
    df = pd.read_parquet(geonames_path).copy()
    df["country_code"] = df["country_code"].map(norm_text)
    df["name_norm"] = df["name"].map(norm_text)
    df["ascii_name_norm"] = df["ascii_name"].map(norm_text)
    df["alternate_name_list"] = df["alternate_names"].map(split_alternate_names)
    df["population_int"] = df["population"].map(to_population)

    by_country: dict[str, dict[str, Any]] = {}

    for row in df.itertuples(index=False):
        country_code = row.country_code
        geoname_id = str(row.geoname_id)
        name_norm = row.name_norm
        ascii_name_norm = row.ascii_name_norm
        population = row.population_int

        if country_code is None or name_norm is None:
            continue

        bucket = by_country.setdefault(
            country_code,
            {
                "exact": {},
                "alternate": {},
            },
        )

        record = GeoRecord(
            geoname_id=geoname_id,
            name_norm=name_norm,
            country_code=country_code,
            population=population,
        )

        for key in {name_norm, ascii_name_norm}:
            if key is None:
                continue
            bucket["exact"].setdefault(key, []).append(record)

        for key in row.alternate_name_list:
            bucket["alternate"].setdefault(key, []).append(record)

    final_index: dict[str, dict[str, Any]] = {}

    for country_code, bucket in by_country.items():
        exact_final = {k: choose_best(v) for k, v in bucket["exact"].items()}
        alternate_final = {k: choose_best(v) for k, v in bucket["alternate"].items()}
        canonical_choices = sorted(
            {record.name_norm for record in exact_final.values()},
            key=lambda x: (len(x), x),
        )
        alternate_choices = sorted(alternate_final.keys(), key=lambda x: (len(x), x))
        final_index[country_code] = {
            "exact": exact_final,
            "alternate": alternate_final,
            "canonical_choices": canonical_choices,
            "alternate_choices": alternate_choices,
        }

    return final_index


def resolve_city(
    city_raw: Any,
    country_raw: Any,
    geonames_index: dict[str, dict[str, Any]],
) -> tuple[str | None, str | None, str, float | None]:
    city = norm_text(city_raw)
    country = norm_text(country_raw)

    if city is None:
        return None, None, "missing_city", None
    if country is None:
        return None, None, "missing_country", None

    bucket = geonames_index.get(country)
    if bucket is None:
        return None, None, "no_match", None

    exact_record = bucket["exact"].get(city)
    if exact_record is not None:
        return exact_record.geoname_id, exact_record.name_norm, "exact", 1.0

    alternate_record = bucket["alternate"].get(city)
    if alternate_record is not None:
        return alternate_record.geoname_id, alternate_record.name_norm, "alternate", 1.0

    canonical_choices = bucket["canonical_choices"]
    if canonical_choices:
        canonical_match = process.extractOne(
            city, canonical_choices, scorer=fuzz.WRatio
        )
        if canonical_match is not None:
            canonical_name_norm, canonical_score, _ = canonical_match
            if canonical_score > 0:
                record = bucket["exact"][canonical_name_norm]
                return (
                    record.geoname_id,
                    record.name_norm,
                    "fuzzy",
                    float(canonical_score) / 100.0,
                )

    alternate_choices = bucket["alternate_choices"]
    if alternate_choices:
        alternate_match = process.extractOne(
            city, alternate_choices, scorer=fuzz.WRatio
        )
        if alternate_match is not None:
            alternate_name_norm, alternate_score, _ = alternate_match
            if alternate_score > 0:
                record = bucket["alternate"][alternate_name_norm]
                return (
                    record.geoname_id,
                    record.name_norm,
                    "fuzzy_alternate",
                    float(alternate_score) / 100.0,
                )

    return None, None, "no_match", None


def resolve_part(
    input_path: Path,
    output_path: Path,
    geonames_index: dict[str, dict[str, Any]],
) -> None:
    df = pd.read_parquet(input_path).copy()

    if "org_city_from" in df.columns:
        df = df.rename(columns={"org_city_from": "org_city_from_raw"})
    if "org_city_to" in df.columns:
        df = df.rename(columns={"org_city_to": "org_city_to_raw"})

    from_results = [
        resolve_city(city, country, geonames_index)
        for city, country in zip(
            df["org_city_from_raw"], df["org_country_from"], strict=False
        )
    ]
    to_results = [
        resolve_city(city, country, geonames_index)
        for city, country in zip(
            df["org_city_to_raw"], df["org_country_to"], strict=False
        )
    ]

    df["org_city_from_geonames_id"] = [x[0] for x in from_results]
    df["org_city_from_geonames_name"] = [x[1] for x in from_results]
    df["org_city_from_geonames_method"] = [x[2] for x in from_results]
    df["org_city_from_geonames_confidence"] = [x[3] for x in from_results]

    df["org_city_to_geonames_id"] = [x[0] for x in to_results]
    df["org_city_to_geonames_name"] = [x[1] for x in to_results]
    df["org_city_to_geonames_method"] = [x[2] for x in to_results]
    df["org_city_to_geonames_confidence"] = [x[3] for x in to_results]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def iter_part_files(input_dir: Path) -> list[Path]:
    return sorted(input_dir.glob("*.parquet"))


def run_one_part(
    input_path_str: str,
    output_path_str: str,
    geonames_index: dict[str, dict[str, Any]],
) -> str:
    input_path = Path(input_path_str)
    output_path = Path(output_path_str)
    resolve_part(
        input_path=input_path,
        output_path=output_path,
        geonames_index=geonames_index,
    )
    return input_path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/orcid/00_raw/edge_aff"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/orcid/01_resolve_city"),
    )
    parser.add_argument(
        "--geonames-path",
        type=Path,
        default=Path("data/geonames/cities1000.parquet"),
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
    geonames_index = build_geonames_index(args.geonames_path)

    if args.part is not None:
        input_files = [args.input_dir / f"{args.part}.parquet"]
    else:
        input_files = iter_part_files(args.input_dir)

    tasks = [
        (str(input_path), str(args.output_dir / input_path.name))
        for input_path in input_files
    ]

    if len(tasks) == 1:
        name = run_one_part(tasks[0][0], tasks[0][1], geonames_index)
        print(f"done {name}")
        return

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                run_one_part, input_path_str, output_path_str, geonames_index
            )
            for input_path_str, output_path_str in tasks
        ]
        for future in as_completed(futures):
            name = future.result()
            print(f"done {name}")


if __name__ == "__main__":
    main()
