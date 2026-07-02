"""Run database migrations.

Usage:
    python -m pagemind.migrations up
"""
import asyncio
import sys
from pathlib import Path

import psycopg

from pagemind.config import settings

# SQL files live under migrations/ at the project root.
MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "migrations"


def _split_statements(sql: str) -> list[str]:
    """Split a SQL file on semicolons, skipping blank fragments."""
    import re
    # Strip single-line comments before splitting so semicolons inside comments
    # don't produce spurious fragments.
    sql_no_comments = re.sub(r"--[^\n]*", "", sql)
    results = []
    for fragment in sql_no_comments.split(";"):
        stmt = fragment.strip()
        if stmt:
            results.append(stmt)
    return results


async def run_up() -> None:
    async with await psycopg.AsyncConnection.connect(settings.database_url, autocommit=True) as conn:
        # Bootstrap migration-tracking table.
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS _migrations (
                name       TEXT        PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)

        cur = await conn.execute("SELECT name FROM _migrations ORDER BY name")
        applied = {row[0] for row in await cur.fetchall()}

        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        if not migration_files:
            print("no migration files found in", MIGRATIONS_DIR)
            return

        for path in migration_files:
            if path.name in applied:
                print(f"  skip  {path.name}")
                continue

            print(f"  apply {path.name} …")
            statements = _split_statements(path.read_text())
            async with conn.transaction():
                for stmt in statements:
                    await conn.execute(stmt)
                await conn.execute(
                    "INSERT INTO _migrations (name) VALUES (%s)", (path.name,)
                )
            print(f"  ✓     {path.name}")

    print("migrations up to date")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "up"
    if cmd == "up":
        asyncio.run(run_up())
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)
