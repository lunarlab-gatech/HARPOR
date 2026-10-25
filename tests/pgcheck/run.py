"""Check the HARPOR migrations on a throwaway local Postgres cluster.

The cluster lives in a new temporary folder, listens on 127.0.0.1 at a free port with no Unix socket,
and is deleted at the end. It never touches another Postgres on the machine. Each database gets
supabase_stub.sql (a small stand-in for Supabase's auth, storage and Realtime), the migrations in
supabase/migrations (without the pg_cron file: plain Postgres has no pg_cron) and helpers.sql.

    python tests/pgcheck/run.py

runs what tests/test_supabase_sql.py runs and prints a summary: the check files in an old project
(with Supabase's old default grants) and in a new one (without them), the shared cases and
supabase/seed.sql. HARPOR_PG_BIN names the folder of pg_ctl, initdb and psql when they are not in the
usual places.
"""

import importlib.util
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MIGRATIONS = ROOT / "supabase" / "migrations"
SEED = ROOT / "supabase" / "seed.sql"
CHECKS = ("10_checks.sql", "20_own_checks.sql", "30_isolation_checks.sql")
BIN_DIRS = ("/opt/homebrew/opt/postgresql@15/bin", "/usr/local/opt/postgresql@15/bin", "/usr/lib/postgresql/15/bin",
            "/opt/homebrew/opt/postgresql@17/bin", "/usr/lib/postgresql/17/bin")
TOOLS = ("pg_ctl", "initdb", "psql")
CHECK_CALL = re.compile(r"\btests\.(check|expect_error)\(")

_spec = importlib.util.spec_from_file_location("supabase_cases", HERE.parent / "supabase_cases.py")
supabase_cases = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(supabase_cases)


class PsqlError(Exception):
    pass


def find_bin():
    """The folder of pg_ctl, initdb and psql of Postgres 15 or newer, or None."""
    dirs = [os.environ.get("HARPOR_PG_BIN"), *BIN_DIRS]
    if shutil.which("pg_ctl"):
        dirs.append(str(Path(shutil.which("pg_ctl")).parent))
    for d in dirs:
        if not d or not all((Path(d) / t).is_file() for t in TOOLS):
            continue
        p = subprocess.run([str(Path(d) / "pg_ctl"), "--version"], capture_output=True, text=True)
        m = re.search(r"(\d+)(?:\.\d+)?", p.stdout)
        if p.returncode == 0 and m and int(m.group(1)) >= 15:
            return Path(d)
    return None


def expected_checks(path):
    """How many checks a check file makes: one per tests.check or tests.expect_error call outside a
    comment."""
    lines = (line.split("--", 1)[0] for line in Path(path).read_text().splitlines())
    return sum(len(CHECK_CALL.findall(line)) for line in lines)


def passed(output):
    """The number of checks that printed "ok"."""
    return sum(1 for line in output.splitlines() if re.search(r"NOTICE:\s+ok", line))


def migrations():
    """The migrations in order, without the pg_cron file."""
    return [f for f in sorted(MIGRATIONS.glob("*.sql")) if not f.name.endswith("_cron.sql")]


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _env():
    # Only the cluster's own settings: no PGHOST, PGSERVICE or the like from the caller's shell.
    env = {k: v for k, v in os.environ.items() if not k.startswith("PG")}
    env.update(LC_ALL="C", LANG="C", PGCONNECT_TIMEOUT="10")
    return env


class Cluster:
    def __init__(self, root, bindir):
        self.root = Path(root)
        self.bin = Path(bindir)
        self.data = self.root / "data"
        self.log = self.root / "pg.log"
        self.port = None

    def _run(self, tool, *args, timeout=120):
        return subprocess.run([str(self.bin / tool), *args], capture_output=True, text=True, env=_env(),
                              timeout=timeout)

    def start(self):
        p = self._run("initdb", "-D", str(self.data), "-U", "postgres", "--auth=trust", "-E", "UTF8", "--no-sync")
        if p.returncode:
            raise RuntimeError(f"initdb failed: {p.stderr}")
        for _ in range(3):   # another process may take the free port first
            self.port = _free_port()
            opts = (f"-c listen_addresses=127.0.0.1 -p {self.port} -c unix_socket_directories='' -c fsync=off "
                    "-c synchronous_commit=off -c full_page_writes=off")
            p = self._run("pg_ctl", "-D", str(self.data), "-o", opts, "-l", str(self.log), "-w", "-t", "60", "start")
            if p.returncode == 0:
                return self
        raise RuntimeError(f"pg_ctl start failed: {p.stderr}\n{self.log.read_text() if self.log.exists() else ''}")

    def stop(self):
        if self.data.exists():
            self._run("pg_ctl", "-D", str(self.data), "stop", "-m", "immediate", "-w", timeout=60)
        shutil.rmtree(self.root, ignore_errors=True)

    def _cmd(self, db):
        return [str(self.bin / "psql"), "-X", "-q", "-v", "ON_ERROR_STOP=1", "-h", "127.0.0.1", "-p", str(self.port),
                "-U", "postgres", "-d", db]

    def psql(self, db, sql=None, file=None, check=True, timeout=120):
        """Run SQL text or a file; returns the NOTICEs, errors and \\echo lines in order (rows go nowhere)."""
        cmd = self._cmd(db) + ["-o", os.devnull, "-f", str(file) if file else "-"]
        p = subprocess.run(cmd, input=None if file else sql, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, env=_env(), timeout=timeout)
        if check and p.returncode:
            raise PsqlError(f"psql failed on {file or 'SQL text'} in {db}:\n{p.stdout}")
        return p.stdout

    def value(self, db, sql):
        """The first row of a query, as text."""
        p = subprocess.run(self._cmd(db) + ["-At", "-c", sql], capture_output=True, text=True, env=_env(), timeout=60)
        if p.returncode:
            raise PsqlError(p.stderr)
        return p.stdout.strip()

    def project(self, db, old_grants=False):
        """A new database with the stand-in, the migrations and the helpers. old_grants: as a project made
        before Supabase stopped granting new tables and functions to the API roles."""
        self.psql("postgres", f'create database "{db}"')
        self.psql(db, file=HERE / "supabase_stub.sql")
        if old_grants:
            self.psql(db, file=HERE / "old_grants.sql")
        for f in migrations():
            self.psql(db, file=f)
        self.psql(db, file=HERE / "helpers.sql")
        return db


def run_checks(cluster, db):
    """Run the check files in order in one database. Returns [(name, output, expected count)]; raises
    PsqlError at the first failed check."""
    return [(name, cluster.psql(db, file=HERE / name), expected_checks(HERE / name)) for name in CHECKS]


def run_case(cluster, db, case):
    """One case of the shared table, in a database where cases_setup ran. Returns (ok, output)."""
    text = cluster.psql(db, supabase_cases.case_sql(case), check=False)
    return "ERROR" not in text and passed(text) > 0, text


def cases_setup(cluster, db):
    cluster.psql(db, supabase_cases.base_sql())


def main():
    bindir = find_bin()
    if bindir is None:
        print("no Postgres 15 or newer found (set HARPOR_PG_BIN to the folder of pg_ctl, initdb and psql)")
        return 2
    cluster = Cluster(tempfile.mkdtemp(prefix="harpor_pg_"), bindir).start()
    failed = 0
    try:
        print(f"Postgres from {bindir}, port {cluster.port}; the pg_cron migration is skipped")
        for db, old in (("old_project", True), ("new_project", False)):
            cluster.project(db, old_grants=old)
            try:
                for name, text, expected in run_checks(cluster, db):
                    print(f"{db}: {name}: {passed(text)} of {expected} checks passed")
            except PsqlError as e:
                failed += 1
                print(f"{db}: FAILED\n{e}")
        cluster.project("cases")
        cases_setup(cluster, "cases")
        cases = supabase_cases.load()
        n_ok = 0
        for case in cases:
            ok, text = run_case(cluster, "cases", case)
            n_ok += ok
            if not ok:
                failed += 1
                print(f"case {case['id']}: FAILED\n{text}")
        print(f"cases: {n_ok} of {len(cases)} passed")
        cluster.project("seed")
        cluster.psql("seed", file=SEED)
        print("seed: " + cluster.value("seed", "select string_agg(email || ' approved=' || approved, ', ' order by email) "
                                       "from public.access"))
    finally:
        cluster.stop()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
