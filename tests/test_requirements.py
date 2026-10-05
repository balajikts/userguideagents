"""requirements*.txt and pyproject.toml must list the same packages."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def names(lines) -> set[str]:
    out = set()
    for line in lines:
        line = line.split("#")[0].strip()
        if line and not line.startswith("-r"):
            out.add(re.split(r"[\[<>=~!; ]", line, maxsplit=1)[0].lower())
    return out


def req(path: str) -> set[str]:
    return names((ROOT / path).read_text(encoding="utf-8").splitlines())


def test_requirements_match_pyproject():
    proj = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = proj["optional-dependencies"]
    expected = names(proj["dependencies"] + extras["ui"] + extras["evals"])
    assert req("requirements.txt") == expected
    assert names(extras["pgvector"]) == {"psycopg", "pgvector"}


def test_postgres_driver_only_in_prod_requirements():
    assert {"psycopg", "pgvector"}.isdisjoint(req("requirements.txt"))
    assert req("requirements-prod.txt") == {"psycopg", "pgvector"}
    assert "-r requirements.txt" in (ROOT / "requirements-prod.txt").read_text(encoding="utf-8")
    assert "chromadb" in req("requirements.txt")
