"""Create a Reimagina project with its baseline and indicators, from one of CASES.

    uv run python scripts/new_project.py

Nothing is read from the command line: pick the case in CASE below (and edit the
cases, users or areas if needed). In order, it:

1. creates the project (or reuses the one with the same name in the organization);
2. gives USERS their role on it;
3. adds the areas of interest of AREAS;
4. creates its baseline scenario (or reuses it);
5. builds the street network, uploads its H5 and sets road_network_uri
   (reimagina_street_network_write);
6. builds and writes the baseline's elements (reimagina_baseline_write);
7. computes and writes the baseline's indicators (reimagina_scenario_write).

A case with ``overlay_uri`` and ``zone_uri`` replaces the city inside the zone
with a redesign's elements and streets (``ingestion/overlay.py``): Mexicali's
master plan. Steps 5-7 run here, as the worker would, with the job folder kept in
``.cache/baseline/<city>`` so downloads and height statistics are reused (about
15 min for Mexicali). The database is REIMAGINA_POSTGRES_DSN, read from
``deploy/.env`` when it is not set; the lake needs ``az login``.
"""

import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------- settings

# Which case to create.
CASE = "mexicali_plan_maestro"

ORGANIZATION_ID = "c35c2da2-52df-4e51-95f4-adcaf15fb7a4"  # Mexicali Org
ZONES = "az://sandbox/projects/EXE/cleaned/zonas_de_interes"

CASES = {
    "mexicali": {
        "city": "mexicali",
        "project": "Mexicali",
        "description": "Mexicali tal como es hoy.",
        "baseline": "Base Mexicali",
    },
    "mexicali_plan_maestro": {
        "city": "mexicali",
        "project": "Mexicali plan maestro",
        "description": "Mexicali con el plan maestro en la zona de Condesa.",
        "baseline": "Base Mexicali con plan maestro",
        # The plan's elements and streets (blocks, lots, amenities,
        # establishments and road_segments .parquet) replace the city inside the
        # zone.
        "overlay_uri": "az://sandbox/projects/EXE/cleaned/plan_maestro",
        "zone_uri": f"{ZONES}/condesa.fgb",
    },
}

# email -> role (admin, editor or viewer).
USERS = {"ana@example.com": "admin"}

# Areas of interest: (name, polygon file), in the order the web lists them.
AREAS = [
    ("Condesa", f"{ZONES}/condesa.fgb"),
    ("Extensión Condesa", f"{ZONES}/extension_condesa.fgb"),
    ("Zona Poniente", f"{ZONES}/zona_poniente.fgb"),
]

# ------------------------------------------------------------------------------

logger = logging.getLogger("new_project")


def load_env() -> None:
    """REIMAGINA_POSTGRES_DSN from deploy/.env when it is not set."""
    if os.environ.get("REIMAGINA_POSTGRES_DSN"):
        return
    for line in (ROOT / "deploy/.env").read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "REIMAGINA_POSTGRES_DSN":
            os.environ["REIMAGINA_POSTGRES_DSN"] = value.strip().strip("'\"")


def setup(case: dict) -> tuple[str, str]:
    """Steps 1-4 in one transaction: the project's and baseline's ids."""
    import geopandas as gpd
    import psycopg
    import shapely

    from reimagina_test_port.common import lake
    from reimagina_test_port.scenario.source import resolve_dsn

    s = "reimagina_urbano"
    with psycopg.connect(resolve_dsn()) as conn, conn.cursor() as cur:
        row = cur.execute(
            f"SELECT id::text FROM {s}.projects WHERE organization_id = %s::uuid "
            "AND name = %s AND deleted_at IS NULL",
            (ORGANIZATION_ID, case["project"]),
        ).fetchone()
        if row:
            project = row[0]
            logger.info("1. Project %s exists: %s", case["project"], project)
        else:
            project = cur.execute(
                f"INSERT INTO {s}.projects (organization_id, name, short_name, "
                "description) VALUES (%s::uuid, %s, %s, %s) RETURNING id::text",
                (
                    ORGANIZATION_ID,
                    case["project"],
                    case["project"].strip().lower().replace(" ", "_"),
                    case["description"],
                ),
            ).fetchone()[0]
            logger.info("1. Project created: %s", project)
        for email, role in USERS.items():
            user = cur.execute(
                f"SELECT id FROM {s}.users WHERE email = %s AND deleted_at IS NULL",
                (email,),
            ).fetchone()
            if user is None:
                msg = f"No user with email {email}"
                raise SystemExit(msg)
            cur.execute(
                f"INSERT INTO {s}.roles (user_id, project_id, name) "
                f"VALUES (%s, %s::uuid, %s::{s}.user_role) "
                "ON CONFLICT (user_id, project_id) DO UPDATE SET "
                "name = EXCLUDED.name, deleted_at = NULL, updated_at = now()",
                (user[0], project, role),
            )
            logger.info("2. %s: %s", email, role)
        workdir = ROOT / ".cache/areas"
        for order, (name, uri) in enumerate(AREAS, start=1):
            if cur.execute(
                f"SELECT 1 FROM {s}.area_of_interest WHERE project_id = %s::uuid "
                "AND name = %s AND deleted_at IS NULL",
                (project, name),
            ).fetchone():
                logger.info("3. Area %s exists", name)
                continue
            polygons = gpd.read_file(lake.download(uri, workdir)).to_crs(4326)
            area = shapely.union_all(polygons.geometry.to_numpy())
            cur.execute(
                f'INSERT INTO {s}.area_of_interest (project_id, name, type, "order", '
                f"geometry) VALUES (%s::uuid, %s, 'analysis'::{s}.area_type, %s, "
                "ST_SetSRID(ST_GeomFromWKB(%s), 4326))",
                (project, name, order, shapely.to_wkb(area)),
            )
            logger.info("3. Area %s added", name)
        row = cur.execute(
            f"SELECT id::text FROM {s}.scenarios WHERE project_id = %s::uuid "
            "AND parent_scenario_id IS NULL AND deleted_at IS NULL",
            (project,),
        ).fetchone()
        if row:
            baseline = row[0]
            logger.info("4. Baseline exists: %s", baseline)
        else:
            baseline = cur.execute(
                f"INSERT INTO {s}.scenarios (project_id, name) "
                "VALUES (%s::uuid, %s) RETURNING id::text",
                (project, case["baseline"]),
            ).fetchone()[0]
            logger.info("4. Baseline created: %s", baseline)
    return project, baseline


def run(metric: str, parameters: dict, workdir: Path) -> None:
    """One of the plugin's metrics, here, with ``workdir`` as every node's job
    folder."""
    from lyra.sdk import LocalRunner
    from lyra.sdk.local import _LocalContext, _LocalDatasetContext

    from reimagina_test_port.plugin import create_plugin

    job = str(uuid4())

    def context(node: str, role: str):
        kind = _LocalDatasetContext if role == "dataset" else _LocalContext
        adapter = logging.LoggerAdapter(logging.getLogger(f"node.{node}"), {})
        return kind(job, node, adapter, workdir)

    runner = LocalRunner.from_definitions(
        {"reimagina-test-port": create_plugin()}, context_factory=context
    )
    start = time.time()
    outputs = runner.run(metric, {"parameters": parameters})
    for name, table in outputs.items():
        logger.info(
            "%s (%.0f s): %s",
            name,
            time.time() - start,
            table.drop(columns="geometry", errors="ignore").to_dict("records"),
        )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )
    for noisy in ("azure", "urllib3", "fsspec", "adlfs", "rasterio", "pyogrio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Tree coverage runs in worker processes: forked, as on the Linux worker.
    if sys.platform == "darwin":
        multiprocessing.set_start_method("fork", force=True)
        os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    load_env()
    case = CASES[CASE]
    logger.info("Case %s", CASE)
    project, baseline = setup(case)
    city = {
        "city": case["city"],
        **{k: case[k] for k in ("overlay_uri", "zone_uri") if k in case},
    }
    workdir = ROOT / ".cache/baseline" / case["city"]
    logger.info("5. Street network")
    run("reimagina_street_network_write", {**city, "project_id": project}, workdir)
    logger.info("6. Baseline elements")
    run(
        "reimagina_baseline_write",
        {**city, "project_id": project, "scenario_id": baseline},
        workdir,
    )
    logger.info("7. Indicators")
    run("reimagina_scenario_write", {"scenario_id": baseline}, workdir)
    logger.info("Done: project %s, baseline %s", project, baseline)


if __name__ == "__main__":
    main()
