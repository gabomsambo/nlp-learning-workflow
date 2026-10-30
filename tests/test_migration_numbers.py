"""Hand-applied migrations must apply in one unambiguous order."""
import re
from collections import defaultdict
from pathlib import Path

MIGRATIONS = Path(__file__).resolve().parent.parent / "docs" / "migrations"


def test_migration_number_prefixes_are_unique():
    by_number = defaultdict(list)
    for path in sorted(MIGRATIONS.glob("*.sql")):
        match = re.match(r"(\d+)_", path.name)
        assert match, f"{path.name} has no numeric prefix"
        by_number[int(match.group(1))].append(path.name)
    duplicates = {n: names for n, names in by_number.items() if len(names) > 1}
    assert not duplicates, f"duplicate migration numbers: {duplicates}"
