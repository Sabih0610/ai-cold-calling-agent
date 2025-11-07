# dashboard/app.py
import os
import io
import csv
import json
from datetime import datetime
from typing import Optional, List, Tuple

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, Query
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------
# Environment / constants
# ---------------------------------------------------------------------
load_dotenv()
DB_URL = os.getenv("DATABASE_URL", "postgresql://dialer:dialer@localhost:5432/dialer")
PAGE_SIZE_DEFAULT = 50

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)

# ---------------------------------------------------------------------
# App + routers
# ---------------------------------------------------------------------
app = FastAPI(title="Dialer Dashboard")

# Import AFTER app is defined; include the API router for campaign start/stop
from .api_campaigns import router as campaigns_router  # noqa: E402
app.include_router(campaigns_router)

# Templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Static
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Recordings (served as static files)
RECORDINGS_DIR = os.path.join(REPO_ROOT, "data", "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)
app.mount("/recordings", StaticFiles(directory=RECORDINGS_DIR), name="recordings")

# ---------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------
def db():
    return psycopg2.connect(DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)

def q1(sql: str, params: Optional[Tuple] = None):
    with db() as conn, conn.cursor() as cur:
        cur.execute(sql, params or ())
        return cur.fetchone()

def qa(sql: str, params: Optional[Tuple] = None):
    with db() as conn, conn.cursor() as cur:
        cur.execute(sql, params or ())
        return cur.fetchall()

def qi(sql: str, params: Optional[Tuple] = None):
    with db() as conn, conn.cursor() as cur:
        cur.execute(sql, params or ())

def set_cfg(key: str, value: str):
    qi(
        """
        INSERT INTO config(key,value) VALUES(%s,%s)
        ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value
        """,
        (key, value),
    )

def get_cfg(key: str, default=None):
    row = q1("SELECT value FROM config WHERE key=%s", (key,))
    return row["value"] if row and row["value"] is not None else default

# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------
def _sql_bool(val: Optional[str]):
    if val is None or val == "all":
        return None
    s = str(val).lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    return None

# Whitelist sort keys to avoid SQL injection
_SORT_MAP = {
    "l.id": "l.id",
    "l.company": "l.company",
    "cl.last_called_at DESC": "cl.last_called_at DESC",
    "cl.attempts DESC": "cl.attempts DESC",
}

def _safe_sort(sort: str) -> str:
    return _SORT_MAP.get(sort, "l.id")

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    totals = q1(
        """
        SELECT
          (SELECT COUNT(*) FROM leads) AS leads_total,
          (SELECT COUNT(*) FROM leads_eligible_now) AS eligible_now,
          (SELECT COALESCE(SUM((status='COMPLETED')::int),0) FROM campaign_leads) AS completed_total
        """
    )

    # include live worker info on the home page
    campaigns = qa(
        """
        SELECT c.id, c.name, c.created_at, c.is_active, c.worker_pid, c.last_heartbeat,
               COUNT(cl.lead_id) AS leads,
               COALESCE(SUM((cl.ready)::int),0) AS ready,
               COALESCE(SUM((cl.status='COMPLETED')::int),0) AS completed
        FROM campaigns c
        LEFT JOIN campaign_leads cl ON cl.campaign_id = c.id
        GROUP BY c.id, c.name, c.created_at, c.is_active, c.worker_pid, c.last_heartbeat
        ORDER BY c.created_at DESC NULLS LAST, c.name
        """
    )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "totals": totals,
            "campaigns": campaigns,
            "dialer_enabled": get_cfg("DIALER_ENABLED", "false"),
            "active_cid": get_cfg("ACTIVE_CAMPAIGN_ID", None),
        },
    )

@app.get("/campaigns", response_class=HTMLResponse)
def list_campaigns(request: Request):
    rows = qa(
        """
        SELECT c.id, c.name, c.created_at, c.is_active, c.worker_pid, c.last_heartbeat,
               COUNT(cl.lead_id) AS leads,
               COALESCE(SUM((cl.ready)::int),0) AS ready,
               COALESCE(SUM((cl.status='COMPLETED')::int),0) AS completed
        FROM campaigns c
        LEFT JOIN campaign_leads cl ON cl.campaign_id = c.id
        GROUP BY c.id, c.name, c.created_at, c.is_active, c.worker_pid, c.last_heartbeat
        ORDER BY c.created_at DESC NULLS LAST, c.name
        """
    )
    return templates.TemplateResponse("campaigns.html", {"request": request, "rows": rows})

@app.get("/campaigns/{cid}", response_class=HTMLResponse)
def campaign_detail(cid: int, request: Request):
    c = q1("SELECT * FROM campaigns WHERE id=%s", (cid,))
    if not c:
        return HTMLResponse("Not found", status_code=404)

    counts = q1(
        """
        SELECT
          COUNT(*) AS total,
          COALESCE(SUM((status='NEW')::int),0) AS new,
          COALESCE(SUM((status='IN_PROGRESS')::int),0) AS in_progress,
          COALESCE(SUM((status='CALLBACK_DUE')::int),0) AS callback_due,
          COALESCE(SUM((status='COMPLETED')::int),0) AS completed,
          COALESCE(SUM((ready)::int),0) AS ready
        FROM campaign_leads
        WHERE campaign_id=%s
        """,
        (cid,),
    )

    active_calls = qa(
        """
        SELECT l.id AS lead_id, l.company, l.phone_e164, l.city, l.state,
               cl.last_called_at, cl.next_action_at
          FROM campaign_leads cl
          JOIN leads l ON l.id = cl.lead_id
         WHERE cl.campaign_id=%s
           AND cl.status = 'IN_PROGRESS'
         ORDER BY cl.last_called_at DESC NULLS LAST, l.id
        """,
        (cid,),
    )

    return templates.TemplateResponse(
        "campaign_detail.html",
        {
            "request": request,
            "c": c,
            "counts": counts,
            "active_calls": active_calls,
            "dialer_enabled": get_cfg("DIALER_ENABLED", "false"),
            "active_cid": get_cfg("ACTIVE_CAMPAIGN_ID", None),
        },
    )

# ---------- Leads table (filters + paging) ----------
@app.get("/campaigns/{cid}/leads", response_class=HTMLResponse)
def campaign_leads(
    cid: int,
    request: Request,
    status: str = Query("all"),          # NEW / IN_PROGRESS / CALLBACK_DUE / COMPLETED / all / READY
    ready: str = Query("all"),           # true / false / all
    q: Optional[str] = Query(None),      # search (company/email/phone)
    page: int = Query(1, ge=1),
    size: int = Query(PAGE_SIZE_DEFAULT, ge=5, le=200),
    sort: str = Query("l.id"),           # whitelisted in _safe_sort()
):
    where: List[str] = ["cl.campaign_id=%s"]
    params: List = [cid]

    if status and status != "all":
        if status == "READY":
            where.append("cl.ready=true AND (cl.next_action_at IS NULL OR cl.next_action_at <= NOW())")
        else:
            where.append("cl.status=%s")
            params.append(status)

    rb = _sql_bool(ready)
    if rb is not None:
        where.append("cl.ready=%s")
        params.append(rb)

    if q:
        like = f"%{q}%"
        where.append("(l.company ILIKE %s OR l.email ILIKE %s OR l.phone_e164 ILIKE %s)")
        params.extend([like, like, like])

    where_sql = " AND ".join(where)
    order_sql = _safe_sort(sort)

    total = q1(
        f"SELECT COUNT(*) AS n FROM campaign_leads cl JOIN leads l ON l.id=cl.lead_id WHERE {where_sql}",
        params,
    )["n"]

    rows = qa(
        f"""
        SELECT l.id AS lead_id, l.company, l.email, l.phone_e164, l.city, l.state,
               cl.status, cl.ready, cl.attempts, cl.last_called_at, cl.next_action_at,
               last.disposition, last.duration_sec, last.recording_url, last.started_at AS last_started_at
        FROM campaign_leads cl
        JOIN leads l ON l.id = cl.lead_id
        LEFT JOIN LATERAL (
           SELECT ca.disposition, ca.duration_sec, ca.recording_url, ca.started_at
           FROM call_attempts ca
           WHERE ca.lead_id = l.id AND ca.campaign_id = cl.campaign_id
           ORDER BY ca.started_at DESC
           LIMIT 1
        ) last ON true
        WHERE {where_sql}
        ORDER BY {order_sql} NULLS LAST
        LIMIT %s OFFSET %s
        """,
        params + [size, (page - 1) * size],
    )

    return templates.TemplateResponse(
        "campaign_leads.html",
        {
            "request": request,
            "cid": cid,
            "rows": rows,
            "page": page,
            "size": size,
            "total": total,
            "status": status,
            "ready": ready,
            "q": q,
            "sort": sort,
        },
    )

# ---------- Toggle a single lead's ready flag ----------
@app.post("/api/leads/{cid}/{lead_id}/toggle_ready")
def toggle_lead_ready(cid: int, lead_id: int):
    qi(
        """
        UPDATE campaign_leads
           SET ready = NOT ready,
               next_action_at = CASE WHEN ready THEN next_action_at ELSE NOW() END
         WHERE campaign_id=%s AND lead_id=%s
        """,
        (cid, lead_id),
    )
    return RedirectResponse(f"/campaigns/{cid}", status_code=303)

# ---------- Calls page (filters + paging) ----------
@app.get("/calls", response_class=HTMLResponse)
def calls(
    request: Request,
    campaign_id: Optional[int] = Query(None),
    disposition: str = Query("all"),
    q: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    size: int = Query(PAGE_SIZE_DEFAULT, ge=5, le=200),
):
    where: List[str] = ["1=1"]
    params: List = []

    if campaign_id:
        where.append("ca.campaign_id=%s")
        params.append(campaign_id)
    if disposition and disposition != "all":
        where.append("ca.disposition=%s")
        params.append(disposition)
    if q:
        like = f"%{q}%"
        where.append("(l.company ILIKE %s OR l.phone_e164 ILIKE %s)")
        params.extend([like, like])
    if date_from:
        where.append("ca.started_at >= %s")
        params.append(date_from)
    if date_to:
        where.append("ca.started_at < %s")
        params.append(date_to)

    where_sql = " AND ".join(where)
    total = q1(
        f"SELECT COUNT(*) AS n FROM call_attempts ca JOIN leads l ON l.id=ca.lead_id WHERE {where_sql}",
        params,
    )["n"]

    rows = qa(
        f"""
        SELECT ca.*, l.company, l.phone_e164
          FROM call_attempts ca
          JOIN leads l ON l.id = ca.lead_id
         WHERE {where_sql}
         ORDER BY ca.started_at DESC
         LIMIT %s OFFSET %s
        """,
        params + [size, (page - 1) * size],
    )

    return templates.TemplateResponse(
        "calls.html",
        {
            "request": request,
            "rows": rows,
            "campaign_id": campaign_id,
            "disposition": disposition,
            "q": q,
            "date_from": date_from,
            "date_to": date_to,
            "page": page,
            "size": size,
            "total": total,
        },
    )


def _load_transcript(basename: str) -> list[dict]:
    if not basename:
        return []
    path = os.path.join(RECORDINGS_DIR, f"{basename}.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


@app.get("/conversations", response_class=HTMLResponse)
def conversations(
    request: Request,
    campaign_id: Optional[int] = Query(None),
    lead_id: Optional[int] = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    where = ["ca.recording_url IS NOT NULL"]
    params: List = []
    if campaign_id:
        where.append("ca.campaign_id=%s")
        params.append(campaign_id)
    if lead_id:
        where.append("ca.lead_id=%s")
        params.append(lead_id)
    where_sql = " AND ".join(where) if where else "1=1"

    rows = qa(
        f"""
        SELECT ca.id,
               ca.lead_id,
               ca.campaign_id,
               ca.started_at,
               ca.disposition,
               ca.duration_sec,
               ca.recording_url,
               l.company,
               l.contact,
               l.phone_e164,
               c.name AS campaign_name
          FROM call_attempts ca
          JOIN leads l ON l.id = ca.lead_id
          JOIN campaigns c ON c.id = ca.campaign_id
         WHERE {where_sql}
         ORDER BY ca.started_at DESC
         LIMIT %s
        """,
        params + [limit],
    )

    entries = []
    for row in rows:
        rec_url = row.get("recording_url") or ""
        basename = ""
        if rec_url:
            basename = os.path.splitext(os.path.basename(rec_url))[0]
        transcript = _load_transcript(basename)
        entries.append(
            {
                "campaign_name": row.get("campaign_name"),
                "lead_id": row.get("lead_id"),
                "company": row.get("company"),
                "contact": row.get("contact"),
                "phone": row.get("phone_e164"),
                "started_at": row.get("started_at"),
                "disposition": row.get("disposition"),
                "duration": row.get("duration_sec"),
                "recording_url": rec_url,
                "transcript": transcript,
                "rec_basename": basename,
            }
        )

    return templates.TemplateResponse(
        "conversations.html",
        {
            "request": request,
            "entries": entries,
            "campaign_id": campaign_id,
            "lead_id": lead_id,
            "limit": limit,
        },
    )

# ---------- Dialer toggle (global enable/disable) ----------
@app.post("/api/dialer/toggle")
def api_toggle_dialer(enabled: str = Form(...)):
    set_cfg("DIALER_ENABLED", "true" if enabled == "true" else "false")
    if enabled != "true":
        qi("UPDATE campaign_leads SET status='CALLBACK_DUE' WHERE status='IN_PROGRESS'")
    return RedirectResponse("/", status_code=303)

# ---------- Campaign selection rules (server-side apply) ----------
@app.post("/api/campaigns/{cid}/select")
def api_select_rules(
    cid: int,
    rule: str = Form(...),
    every_n: Optional[int] = Form(None),
    offset_start: Optional[int] = Form(None),
):
    # reset all ready
    qi("UPDATE campaign_leads SET ready=false WHERE campaign_id=%s", (cid,))

    if rule == "all":
        qi("UPDATE campaign_leads SET ready=true WHERE campaign_id=%s", (cid,))
    elif rule == "odd":
        qi("UPDATE campaign_leads SET ready=((lead_id % 2)=1) WHERE campaign_id=%s", (cid,))
    elif rule == "even":
        qi("UPDATE campaign_leads SET ready=((lead_id % 2)=0) WHERE campaign_id=%s", (cid,))
    elif rule == "every_n" and every_n and every_n > 1:
        qi(
            """
            WITH t AS (
              SELECT lead_id, row_number() OVER (ORDER BY lead_id) rn
                FROM campaign_leads
               WHERE campaign_id=%s
            )
            UPDATE campaign_leads cl
               SET ready = ((t.rn % %s) = 1)
              FROM t
             WHERE cl.campaign_id=%s AND cl.lead_id=t.lead_id
            """,
            (cid, every_n, cid),
        )
    elif rule == "start_middle":
        qi(
            """
            WITH t AS (
              SELECT lead_id,
                     row_number() OVER (ORDER BY lead_id) rn,
                     COUNT(*) OVER () AS total
                FROM campaign_leads
               WHERE campaign_id=%s
            )
            UPDATE campaign_leads cl
               SET ready = (t.rn >= (t.total/2))
              FROM t
             WHERE cl.campaign_id=%s AND cl.lead_id=t.lead_id
            """,
            (cid, cid),
        )
    elif rule == "offset_start" and offset_start and offset_start > 0:
        qi(
            """
            WITH t AS (
              SELECT lead_id, row_number() OVER (ORDER BY lead_id) rn
                FROM campaign_leads
               WHERE campaign_id=%s
            )
            UPDATE campaign_leads cl
               SET ready = (t.rn >= %s)
              FROM t
             WHERE cl.campaign_id=%s AND cl.lead_id=t.lead_id
            """,
            (cid, offset_start, cid),
        )

    # push callable ones to now (except completed)
    qi(
        """
        UPDATE campaign_leads
           SET next_action_at = NOW(),
               status = CASE WHEN status='COMPLETED' THEN 'COMPLETED' ELSE 'NEW' END
         WHERE campaign_id=%s
           AND ready=true
           AND (next_action_at IS NULL OR next_action_at > NOW())
        """,
        (cid,),
    )
    return RedirectResponse(f"/campaigns/{cid}", status_code=303)

@app.post("/api/campaigns/{cid}/requeue")
def api_requeue_now(cid: int):
    qi(
        """
        UPDATE campaign_leads
           SET next_action_at = NOW(),
               status = CASE WHEN status='COMPLETED' THEN 'COMPLETED' ELSE 'NEW' END
         WHERE campaign_id=%s AND ready=true
        """,
        (cid,),
    )
    return RedirectResponse(f"/campaigns/{cid}", status_code=303)

# ---------- CSV exports ----------
@app.get("/export/leads.csv")
def export_leads(campaign_id: int = Query(...)):
    rows = qa(
        """
        SELECT l.id, l.company, l.email, l.phone_e164, l.city, l.state,
               cl.status, cl.ready, cl.attempts, cl.last_called_at, cl.next_action_at
          FROM campaign_leads cl
          JOIN leads l ON l.id = cl.lead_id
         WHERE cl.campaign_id=%s
         ORDER BY l.id
        """,
        (campaign_id,),
    )
    buf = io.StringIO()
    fieldnames = list(rows[0].keys()) if rows else [
        "id", "company", "email", "phone_e164", "city", "state",
        "status", "ready", "attempts", "last_called_at", "next_action_at",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k) for k in fieldnames})
    buf.seek(0)
    fn = f"leads_campaign_{campaign_id}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{fn}"'},
    )

@app.get("/export/calls.csv")
def export_calls(campaign_id: int = Query(...)):
    rows = qa(
        """
        SELECT ca.id, ca.lead_id, ca.campaign_id, ca.started_at, ca.ended_at, ca.duration_sec,
               ca.disposition, ca.recording_url, l.company, l.phone_e164
          FROM call_attempts ca
          JOIN leads l ON l.id = ca.lead_id
         WHERE ca.campaign_id=%s
         ORDER BY ca.started_at DESC
        """,
        (campaign_id,),
    )
    buf = io.StringIO()
    fieldnames = list(rows[0].keys()) if rows else [
        "id", "lead_id", "campaign_id", "started_at", "ended_at",
        "duration_sec", "disposition", "recording_url", "company", "phone_e164",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k) for k in fieldnames})
    buf.seek(0)
    fn = f"calls_campaign_{campaign_id}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{fn}"'},
    )
