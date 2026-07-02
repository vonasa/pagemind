set dotenv-load := true

# Bring up Postgres (in Docker), then apply migrations.
# The embedding server runs natively — see `just embed` — so it is NOT started here.
up: _check-docker
    docker compose -f docker/compose.yaml up -d
    just _wait-healthy
    just migrate

# Start the native Infinity embedding server in the background.
# First run sets up a dedicated venv and downloads the model (~2.2 GB).
# Logs: .infinity/run.log  |  PID: .infinity/infinity.pid  |  stop via `just down`.
embed:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p .infinity
    PIDFILE=.infinity/infinity.pid
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "Embedding server already running (pid $(cat "$PIDFILE"))."
        exit 0
    fi
    nohup ./scripts/run-infinity.sh > .infinity/run.log 2>&1 &
    echo $! > "$PIDFILE"
    echo "Embedding server starting (pid $!); logs: .infinity/run.log"
    for _ in $(seq 1 120); do
        if curl -fsS http://localhost:7997/health >/dev/null 2>&1; then
            echo "Ready at http://localhost:7997"
            exit 0
        fi
        if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "Server exited during startup — see .infinity/run.log" >&2
            rm -f "$PIDFILE"
            exit 1
        fi
        sleep 2
    done
    echo "Still starting after 240s (likely first-run model download). It will come up; tail .infinity/run.log."

# Stop Postgres and the native embedding server (data volumes preserved).
down:
    #!/usr/bin/env bash
    set -uo pipefail
    PIDFILE=.infinity/infinity.pid
    if [ -f "$PIDFILE" ]; then
        PID="$(cat "$PIDFILE")"
        if kill -0 "$PID" 2>/dev/null; then
            echo "Stopping embedding server (pid $PID)…"
            kill "$PID" 2>/dev/null || true
            # Escalate to SIGKILL if it doesn't exit promptly (torch can be slow).
            for _ in $(seq 1 10); do
                kill -0 "$PID" 2>/dev/null || break
                sleep 1
            done
            kill -9 "$PID" 2>/dev/null || true
        fi
        rm -f "$PIDFILE"
    fi
    # Safety net for an untracked server bound to our venv path.
    pkill -9 -f ".infinity/venv/bin/infinity_emb" 2>/dev/null || true
    docker compose -f docker/compose.yaml down

# Apply any pending SQL migrations.
migrate:
    uv run python -m pagemind.migrations up

# Export ALL pre-compiled material (every book + summaries, entities, embeddings,
# FTS) to a single portable dump file. Requires Postgres up (`just up`).
export-db FILE="pagemind-precompiled.dump":
    #!/usr/bin/env bash
    set -euo pipefail
    if ! docker compose -f docker/compose.yaml exec -T postgres \
            pg_isready -U pagemind -d pagemind -q 2>/dev/null; then
        echo "ERROR: Postgres isn't up — run 'just up' first." >&2; exit 1
    fi
    docker compose -f docker/compose.yaml exec -T postgres \
        pg_dump -U pagemind -d pagemind -Fc --no-owner --no-privileges > "{{FILE}}"
    echo "Exported -> {{FILE}} ($(du -h "{{FILE}}" | cut -f1))"

# Import pre-compiled material from a dump produced by `just export-db`.
# Requires Postgres up. WARNING: this REPLACES the current database entirely —
# it drops & recreates the DB so the restore lands in a clean, empty schema.
import-db FILE="pagemind-precompiled.dump":
    #!/usr/bin/env bash
    set -euo pipefail
    [ -f "{{FILE}}" ] || { echo "ERROR: no such file: {{FILE}}" >&2; exit 1; }
    CO="docker compose -f docker/compose.yaml exec -T postgres"
    if ! $CO pg_isready -U pagemind -d pagemind -q 2>/dev/null; then
        echo "ERROR: Postgres isn't up — run 'just up' first." >&2; exit 1
    fi
    # Drop existing connections, then drop & recreate an empty DB. Restoring a full
    # dump over an already-migrated schema corrupts constraints ('already exists'
    # errors); a clean target avoids that and exits 0.
    $CO psql -U pagemind -d postgres -q -c \
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='pagemind' AND pid <> pg_backend_pid();" >/dev/null
    $CO dropdb -U pagemind --if-exists pagemind
    $CO createdb -U pagemind pagemind
    $CO pg_restore -U pagemind -d pagemind --no-owner --no-privileges < "{{FILE}}"
    echo "Imported pre-compiled data from {{FILE}}"

# List compiled books (id, title, status). Requires Postgres up.
books:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! docker compose -f docker/compose.yaml exec -T postgres \
            pg_isready -U pagemind -d pagemind -q 2>/dev/null; then
        echo "ERROR: Postgres isn't up — run 'just up' first." >&2; exit 1
    fi
    docker compose -f docker/compose.yaml exec -T postgres psql -U pagemind -d pagemind -c \
        "SELECT book_id, title, status FROM book_meta ORDER BY created_at;"

# Cleanly delete one book and ALL its derived data (chapters, sections, chunks,
# entities, occurrences, events, dates, FTS — all via ON DELETE CASCADE).
rm-book ID:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! [[ "{{ID}}" =~ ^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ ]]; then
        echo "ERROR: '{{ID}}' is not a book_id UUID. Run 'just books' to find it." >&2; exit 1
    fi
    if ! docker compose -f docker/compose.yaml exec -T postgres \
            pg_isready -U pagemind -d pagemind -q 2>/dev/null; then
        echo "ERROR: Postgres isn't up — run 'just up' first." >&2; exit 1
    fi
    # CTE so the statement is a SELECT — psql -tA then yields just the title (or
    # nothing), without the "DELETE N" command tag that a bare DELETE prints.
    TITLE="$(docker compose -f docker/compose.yaml exec -T postgres psql -U pagemind -d pagemind -tAc \
        "WITH d AS (DELETE FROM book_meta WHERE book_id = '{{ID}}' RETURNING title) SELECT title FROM d;")"
    if [ -z "$TITLE" ]; then
        echo "No book with id {{ID}} (already gone?). 'just books' to list." >&2; exit 1
    fi
    echo "Deleted '$TITLE' ({{ID}}) and all its derived data."

# Run the test suite.
test:
    uv run pytest -q

# Compile a book end-to-end into Postgres.
compile BOOK:
    uv run python -m pagemind add "{{BOOK}}"

# Generate the whole-book summary for ready books that lack one (already-ingested
# books compiled before the book_summary stage existed).
backfill-summaries:
    uv run python -m pagemind backfill-summaries

# Compile a book while keeping the Mac awake; sleep re-enables the instant it ends.
# `-s` prevents system sleep only on AC power; a closed lid still sleeps.
compile-caffeinate BOOK:
    caffeinate -i -s just compile "{{BOOK}}"

# Keep the Mac awake until an already-running `just compile` finishes, then re-enable
# sleep. Start the compile first (another terminal), or use `just compile-caffeinate`.
caffeinate:
    #!/usr/bin/env bash
    set -euo pipefail
    PID="$(pgrep -f 'pagemind add' | head -1 || true)"
    if [ -z "$PID" ]; then
        echo "No running compile found." >&2
        echo "  Wrap a fresh compile:  just compile-caffeinate \"book.epub\"" >&2
        echo "  Or start 'just compile <book>' first, then run 'just caffeinate'." >&2
        exit 1
    fi
    echo "Caffeinating until compile (pid $PID) finishes…"
    exec caffeinate -i -s -w "$PID"

# Build the React frontend.
build-web:
    cd web && npm install && npm run build

# Start the API server and open the app in the browser (production build).
# macOS: `open` is used to launch the default browser.
serve: build-web
    uv run uvicorn pagemind.api:app --port 8001 & \
    sleep 1 && open http://localhost:8001 && wait

# Start API + Vite dev server with hot reload (development).
dev:
    #!/usr/bin/env bash
    uv run uvicorn pagemind.api:app --port 8000 --reload &
    API_PID=$!
    cd web && npm run dev &
    WEB_PID=$!
    trap "kill $API_PID $WEB_PID 2>/dev/null" EXIT INT TERM
    sleep 2 && open http://localhost:5173
    wait

# ── internal ──────────────────────────────────────────────────────────────────

_check-docker:
    #!/usr/bin/env bash
    if ! docker info > /dev/null 2>&1; then
        echo "ERROR: Docker daemon is not running." >&2
        echo "  Colima:         colima start" >&2
        echo "  OrbStack:       open -a OrbStack" >&2
        echo "  Docker Desktop: open -a Docker" >&2
        exit 1
    fi

_wait-healthy:
    #!/usr/bin/env bash
    set -e
    echo "Waiting for Postgres to be ready..."
    for i in $(seq 1 30); do
        if docker compose -f docker/compose.yaml exec -T postgres \
               pg_isready -U pagemind -d pagemind -q 2>/dev/null; then
            echo "Postgres ready."
            exit 0
        fi
        sleep 2
    done
    echo "ERROR: timed out waiting for Postgres" >&2
    exit 1
