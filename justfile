set shell := ["bash", "-lc"]

default:
    just --list

init:
    just venv && \
    just sync

venv:
    test -d .venv || uv venv

sync:
    uv sync --all-packages

sync-lock:
    uv sync --locked --all-packages

up:
    uv lock --upgrade

add PKG:
    uv add {{PKG}}

add-dev PKG:
    uv add --dev {{PKG}}

rm PKG:
    uv remove {{PKG}}

rm-dev PKG:
    uv remove --dev {{PKG}}

fmt:
    just sync && \
    uv run ruff format .

fmt-check:
    uv run ruff format --check .

lint:
    uv run ruff check . --fix

lint-check:
    uv run ruff check .

test:
    uv run pytest

ci:
    just venv && \
    just sync-lock && \
    just fmt-check && \
    just lint-check # && \
    # just test

docs-build:
	uv run pdoc --math -o site osr_resolve

docs-serve:
    uv run pdoc --math -p 8080 osr_resolve

find-last D:
    uv run python src/osr_resolve/tools/find_last.py {{D}}

SCRIPTS_GEONAMES_DATE := "2024_03_10"
SCRIPTS_ORCID_DATE := "2025_12_17"

pl-gn-0001 S="src" D=SCRIPTS_GEONAMES_DATE U="data/pipeline/geonames/0000_raw" O="data/pipeline/geonames/0001_meta" I="data/external/geonames" C="config/pipeline/geonames/0001.yaml":
    uv run python scripts/pipeline/geonames/0001.py {{S}} {{D}} {{U}} {{O}} {{I}} {{C}}

pl-gn-0010 S="src" D=SCRIPTS_GEONAMES_DATE U="data/pipeline/geonames/0001_meta" O="data/pipeline/geonames/0010_partition" I="data/pipeline/geonames/0000_raw" C="config/pipeline/geonames/0010.yaml":
    uv run python scripts/pipeline/geonames/0010.py {{S}} {{D}} {{U}} {{O}} {{I}} {{C}}

pl-gn-0020 S="src" U="data/pipeline/geonames/0010_partition" O="data/pipeline/geonames/0020_rename" I="data/pipeline/geonames/0010_partition" C="config/pipeline/geonames/0020.yaml":
    uv run python scripts/pipeline/geonames/0020.py {{S}} "$(just find-last {{U}})" {{O}} {{I}} {{C}}

pl-gn-0030 S="src" U="data/pipeline/geonames/0020_rename" O="data/pipeline/geonames/0030_drop" I="data/pipeline/geonames/0020_rename" C="config/pipeline/geonames/0030.yaml":
    uv run python scripts/pipeline/geonames/0030.py {{S}} "$(just find-last {{U}})" {{O}} {{I}} {{C}}

pl-gn-0040 S="src" U="data/pipeline/geonames/0030_drop" O="data/pipeline/geonames/0040_alias" I="data/pipeline/geonames/0030_drop" C="config/pipeline/geonames/0040.yaml":
    uv run python scripts/pipeline/geonames/0040.py {{S}} "$(just find-last {{U}})" {{O}} {{I}} {{C}}

pl-gn-0050 S="src" U="data/pipeline/geonames/0040_alias" O="data/pipeline/geonames/0050_normalize" I="data/pipeline/geonames/0040_alias" C="config/pipeline/geonames/0050.yaml":
    uv run python scripts/pipeline/geonames/0050.py {{S}} "$(just find-last {{U}})" {{O}} {{I}} {{C}}

pl-gn-0060 S="src" U="data/pipeline/geonames/0050_normalize" O="data/pipeline/geonames/0060_index" I="data/pipeline/geonames/0050_normalize" C="config/pipeline/geonames/0060.yaml":
    uv run python scripts/pipeline/geonames/0060.py {{S}} "$(just find-last {{U}})" {{O}} {{I}} {{C}}

pl-orcid-0001 S="src" D=SCRIPTS_ORCID_DATE U="data/pipeline/orcid/0000_raw" O="data/pipeline/orcid/0001_meta" I="data/external/orcid" C="config/pipeline/orcid/0001.yaml":
    uv run python scripts/pipeline/orcid/0001.py {{S}} {{D}} {{U}} {{O}} {{I}} {{C}}

pl-orcid-0001-s S="src" D=SCRIPTS_ORCID_DATE U="samples/pipeline/orcid/65536_0/0000_raw" O="samples/pipeline/orcid/65536_0/0001_meta" I="data/external/orcid" C="config/pipeline/orcid/0001.yaml":
    uv run python scripts/pipeline/orcid/0001.py {{S}} {{D}} {{U}} {{O}} {{I}} {{C}}

pl-orcid-0010 S="src" D=SCRIPTS_ORCID_DATE U="data/pipeline/orcid/0001_meta" O="data/pipeline/orcid/0010_resolve_org_country" I="data/pipeline/orcid/0000_raw" C="config/pipeline/orcid/0010.yaml":
    uv run python scripts/pipeline/orcid/0010.py {{S}} {{D}} "$(just find-last {{U}})" {{O}} {{I}} {{C}}

pl-orcid-0010-s S="src" D=SCRIPTS_ORCID_DATE U="samples/pipeline/orcid/65536_0/0001_meta" O="samples/pipeline/orcid/65536_0/0010_resolve_org_country" I="samples/pipeline/orcid/65536_0/0000_raw" C="config/pipeline/orcid/0010.yaml":
    uv run python scripts/pipeline/orcid/0010.py {{S}} {{D}} "$(just find-last {{U}})" {{O}} {{I}} {{C}}

pl-orcid-0020 S="src" D=SCRIPTS_ORCID_DATE UO="data/pipeline/orcid/0001_meta" UG="data/pipeline/geonames/0060_index" O="data/pipeline/orcid/0020_match_org_city_exact" IO="data/pipeline/orcid/0000_raw" IG="data/pipeline/geonames/0060_index" C="config/pipeline/orcid/0020.yaml":
    uv run python scripts/pipeline/orcid/0020.py {{S}} {{D}} "$(just find-last {{UO}})" "$(just find-last {{UG}})" {{O}} {{IO}} {{IG}} {{C}}

pl-orcid-0020-s S="src" D=SCRIPTS_ORCID_DATE UO="samples/pipeline/orcid/65536_0/0001_meta" UG="data/pipeline/geonames/0060_index" O="samples/pipeline/orcid/65536_0/0020_match_org_city_exact" IO="samples/pipeline/orcid/65536_0/0000_raw" IG="data/pipeline/geonames/0060_index" C="config/pipeline/orcid/0020.yaml":
    uv run python scripts/pipeline/orcid/0020.py {{S}} {{D}} "$(just find-last {{UO}})" "$(just find-last {{UG}})" {{O}} {{IO}} {{IG}} {{C}}
