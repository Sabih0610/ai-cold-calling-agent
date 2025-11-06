# dashboard/api_campaigns.py
import os, sys, signal, subprocess
from fastapi import APIRouter, HTTPException
import psycopg2, psycopg2.extras
import psutil

# .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://dialer:dialer@localhost:5432/dialer")
PYTHON = sys.executable

# repo root so Popen can find dialer_worker.py regardless of where uvicorn started
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

router = APIRouter(prefix="/api/campaigns", tags=["campaigns"])

def db():
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def _set_cfg(cur, key: str, value: str):
    cur.execute(
        """
        INSERT INTO config(key, value) VALUES (%s, %s)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """,
        (key, value),
    )


def _clear_cfg(cur, key: str):
    cur.execute("DELETE FROM config WHERE key=%s", (key,))


def _pid_alive(pid: int) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        p = psutil.Process(pid)
        return p.is_running() and (p.status() != psutil.STATUS_ZOMBIE)
    except psutil.Error:
        return False

@router.post("/{cid}/activate")
def activate_campaign(cid: int, selection: str = "all", every_n: int | None = None, parallel: int = 1):
    """
    selection: all | odd | even | every
    every_n:  required if selection == 'every'
    parallel: worker threads (start with 1)
    """
    if selection not in {"all", "odd", "even", "every"}:
        raise HTTPException(400, "Invalid selection")
    if selection == "every" and (not every_n or every_n < 2):
        raise HTTPException(400, "every_n must be >= 2 when selection='every'")

    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, is_active, worker_pid FROM campaigns WHERE id=%s", (cid,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "Campaign not found")

        if row["is_active"] and _pid_alive(row["worker_pid"] or 0):
            # already running – just update selection + heartbeat
            cur.execute("""
                UPDATE campaigns
                   SET selection_rule=%s,
                       selection_n=%s,
                       last_heartbeat=now()
                 WHERE id=%s
            """, (selection, every_n, cid))
            conn.commit()
            return {"ok": True, "status": "already-running", "pid": row["worker_pid"]}

        # mark active and persist selection
        cur.execute("""
            UPDATE campaigns
               SET is_active=true,
                   selection_rule=%s,
                   selection_n=%s
             WHERE id=%s
            RETURNING id
        """, (selection, every_n, cid))
        if not cur.fetchone():
            raise HTTPException(404, "Campaign not found")

        # spawn worker
        cmd = [PYTHON, "-u", os.path.join(REPO_ROOT, "dialer_worker.py"),
               "--campaign-id", str(cid),
               "--parallel", str(parallel),
               "--selection", selection]
        if selection == "every":
            cmd += ["--every", str(every_n)]

        # detached, quiet
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=REPO_ROOT
        )
        cur.execute("UPDATE campaigns SET worker_pid=%s, last_heartbeat=now() WHERE id=%s",
                    (proc.pid, cid))
        try:
            _set_cfg(cur, "ACTIVE_CAMPAIGN_ID", str(cid))
        except Exception:
            pass
        conn.commit()

    return {"ok": True, "status": "started", "pid": proc.pid}

@router.post("/{cid}/stop")
def stop_campaign(cid: int):
    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT worker_pid FROM campaigns WHERE id=%s", (cid,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "Campaign not found")
        pid = row["worker_pid"] or 0

        # flip switch; worker will exit cleanly
        cur.execute(
            "UPDATE campaigns SET is_active=false, worker_pid=NULL WHERE id=%s",
            (cid,),
        )
        try:
            cur.execute("SELECT value FROM config WHERE key='ACTIVE_CAMPAIGN_ID'")
            cfg_row = cur.fetchone()
            if cfg_row and cfg_row.get("value") == str(cid):
                _clear_cfg(cur, "ACTIVE_CAMPAIGN_ID")
        except Exception:
            pass
        conn.commit()

    if _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            try:
                psutil.Process(pid).terminate()
            except Exception:
                pass

    return {"ok": True, "status": "stopping"}
