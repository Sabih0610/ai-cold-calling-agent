#dashboard\app.py
import os, io, csv
from datetime import datetime
from typing import Optional, List

import psycopg2, psycopg2.extras
from fastapi import FastAPI, Request, Form, Query
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL", "postgresql://dialer:dialer@localhost:5432/dialer")

app = FastAPI(title="Dialer Dashboard")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Serve call recordings stored under project data/recordings
RECORDINGS_DIR = os.path.join(REPO_ROOT, "data", "recordings")
try:
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
except Exception:
    pass
app.mount("/recordings", StaticFiles(directory=RECORDINGS_DIR), name="recordings")

PAGE_SIZE_DEFAULT = 50

def db():
    return psycopg2.connect(DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)

# ---------- tiny DB helpers ----------
def q1(sql, params=None):
    with db() as conn, conn.cursor() as cur:
        cur.execute(sql, params or ())
        return cur.fetchone()

def qa(sql, params=None):
    with db() as conn, conn.cursor() as cur:
        cur.execute(sql, params or ())
        return cur.fetchall()

def qi(sql, params=None):
    with db() as conn, conn.cursor() as cur:
        cur.execute(sql, params or ())

def set_cfg(key, value):
    qi("""INSERT INTO config(key,value) VALUES(%s,%s)
          ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value""", (key, value))

def get_cfg(key, default=None):
    row = q1("SELECT value FROM config WHERE key=%s", (key,))
    return row["value"] if row and row["value"] is not None else default

# ---------- HOME ----------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    totals = q1("""
        SELECT
          (SELECT COUNT(*) FROM leads) AS leads_total,
          (SELECT COUNT(*) FROM leads_eligible_now) AS eligible_now,
          (SELECT COALESCE(SUM((status='COMPLETED')::int),0) FROM campaign_leads) AS completed_total
    """)
    campaigns = qa("""
        SELECT c.id, c.name,
               COUNT(cl.lead_id) AS leads,
               COALESCE(SUM((cl.ready)::int),0) AS ready,
               COALESCE(SUM((cl.status='COMPLETED')::int),0) AS completed
        FROM campaigns c
        LEFT JOIN campaign_leads cl ON cl.campaign_id = c.id
        GROUP BY c.id, c.name
        ORDER BY c.created_at DESC NULLS LAST, c.name
    """)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "totals": totals,
        "campaigns": campaigns,
        "dialer_enabled": get_cfg("DIALER_ENABLED", "false"),
        "active_cid": get_cfg("ACTIVE_CAMPAIGN_ID", None)
    })

# ---------- CAMPAIGNS ----------
@app.get("/campaigns", response_class=HTMLResponse)
def list_campaigns(request: Request):
    rows = qa("""
        SELECT c.id, c.name, c.created_at,
               COUNT(cl.lead_id) AS leads,
               COALESCE(SUM((cl.ready)::int),0) AS ready,
               COALESCE(SUM((cl.status='COMPLETED')::int),0) AS completed
        FROM campaigns c
        LEFT JOIN campaign_leads cl ON cl.campaign_id = c.id
        GROUP BY c.id, c.name, c.created_at
        ORDER BY c.created_at DESC NULLS LAST, c.name
    """)
    return templates.TemplateResponse("campaigns.html", {"request": request, "rows": rows})

@app.get("/campaigns/{cid}", response_class=HTMLResponse)
def campaign_detail(cid: int, request: Request):
    c = q1("SELECT * FROM campaigns WHERE id=%s", (cid,))
    if not c:
        return HTMLResponse("Not found", status_code=404)

    counts = q1("""
        SELECT
          COUNT(*) AS total,
          COALESCE(SUM((status='NEW')::int),0) AS new,
          COALESCE(SUM((status='IN_PROGRESS')::int),0) AS in_progress,
          COALESCE(SUM((status='CALLBACK_DUE')::int),0) AS callback_due,
          COALESCE(SUM((status='COMPLETED')::int),0) AS completed,
          COALESCE(SUM((ready)::int),0) AS ready
        FROM campaign_leads
        WHERE campaign_id=%s
    """, (cid,))

    return templates.TemplateResponse("campaign_detail.html", {
        "request": request,
        "c": c,
        "counts": counts,
        "dialer_enabled": get_cfg("DIALER_ENABLED", "false"),
        "active_cid": get_cfg("ACTIVE_CAMPAIGN_ID", None)
    })

# ========== LEADS TABLE (filters + paging) ==========
def _sql_bool(val: Optional[str]):
    if val is None or val == "all": return None
    if str(val).lower() in ("1","true","t","yes","y"): return True
    if str(val).lower() in ("0","false","f","no","n"): return False
    return None

@app.get("/campaigns/{cid}/leads", response_class=HTMLResponse)
def campaign_leads(
    cid: int,
    request: Request,
    status: str = Query("all"),          # NEW / IN_PROGRESS / CALLBACK_DUE / COMPLETED / all
    ready: str = Query("all"),           # true / false / all
    q: Optional[str] = Query(None),      # search (company/email/phone)
    page: int = Query(1, ge=1),
    size: int = Query(PAGE_SIZE_DEFAULT, ge=5, le=200),
    sort: str = Query("l.id")            # l.id / l.company / cl.last_called_at etc.
):
    where = ["cl.campaign_id=%s"]
    params: List = [cid]

    if status and status != "all":
        if status == "READY":            # convenience status for "callable now"
            where.append("cl.ready=true AND (cl.next_action_at IS NULL OR cl.next_action_at <= NOW())")
        else:
            where.append("cl.status=%s")
            params.append(status)

    rb = _sql_bool(ready)
    if rb is not None:
        where.append("cl.ready=%s")
        params.append(rb)

    if q:
        where.append("(l.company ILIKE %s OR l.email ILIKE %s OR l.phone_e164 ILIKE %s)")
        like = f"%{q}%"
        params.extend([like, like, like])

    where_sql = " AND ".join(where)
    # total count
    total = q1(f"SELECT COUNT(*) AS n FROM campaign_leads cl JOIN leads l ON l.id=cl.lead_id WHERE {where_sql}", params)["n"]

    # rows + last call disposition via LATERAL
    rows = qa(f"""
        SELECT l.id AS lead_id, l.company, l.email, l.phone_e164, l.city, l.state,
               cl.status, cl.ready, cl.attempts, cl.last_called_at, cl.next_action_at,
               last.disposition, last.duration_sec, last.recording_url, last.started_at AS last_started_at
        FROM campaign_leads cl
        JOIN leads l ON l.id = cl.lead_id
        LEFT JOIN LATERAL (
           SELECT ca.disposition, ca.duration_sec, ca.recording_url, ca.started_at
           FROM call_attempts ca
           WHERE ca.lead_id=l.id AND ca.campaign_id=cl.campaign_id
           ORDER BY ca.started_at DESC
           LIMIT 1
        ) last ON true
        WHERE {where_sql}
        ORDER BY {sort} NULLS LAST
        LIMIT %s OFFSET %s
    """, params + [size, (page-1)*size])

    return templates.TemplateResponse("campaign_leads.html", {
        "request": request,
        "cid": cid,
        "rows": rows,
        "page": page,
        "size": size,
        "total": total,
        "status": status,
        "ready": ready,
        "q": q,
        "sort": sort
    })

# Toggle a single lead's ready flag (quick CRUD)
@app.post("/api/leads/{cid}/{lead_id}/toggle_ready")
def toggle_lead_ready(cid: int, lead_id: int):
    qi("""
        UPDATE campaign_leads
        SET ready = NOT ready,
            next_action_at = CASE WHEN ready THEN next_action_at ELSE NOW() END
        WHERE campaign_id=%s AND lead_id=%s
    """, (cid, lead_id))
    return RedirectResponse(f"/campaigns/{cid}", status_code=303)

# ---------- CALLS (separate page with filters) ----------
@app.get("/calls", response_class=HTMLResponse)
def calls(
    request: Request,
    campaign_id: Optional[int] = Query(None),
    disposition: str = Query("all"),
    q: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    size: int = Query(PAGE_SIZE_DEFAULT, ge=5, le=200)
):
    where = ["1=1"]
    params: List = []

    if campaign_id:
        where.append("ca.campaign_id=%s")
        params.append(campaign_id)
    if disposition and disposition!="all":
        where.append("ca.disposition=%s")
        params.append(disposition)
    if q:
        where.append("(l.company ILIKE %s OR l.phone_e164 ILIKE %s)")
        like = f"%{q}%"
        params.extend([like, like])
    if date_from:
        where.append("ca.started_at >= %s")
        params.append(date_from)
    if date_to:
        where.append("ca.started_at < %s")
        params.append(date_to)

    where_sql = " AND ".join(where)
    total = q1(f"SELECT COUNT(*) AS n FROM call_attempts ca JOIN leads l ON l.id=ca.lead_id WHERE {where_sql}", params)["n"]

    rows = qa(f"""
      SELECT ca.*, l.company, l.phone_e164
      FROM call_attempts ca
      JOIN leads l ON l.id = ca.lead_id
      WHERE {where_sql}
      ORDER BY ca.started_at DESC
      LIMIT %s OFFSET %s
    """, params + [size, (page-1)*size])

    return templates.TemplateResponse("calls.html", {
        "request": request,
        "rows": rows,
        "campaign_id": campaign_id,
        "disposition": disposition,
        "q": q,
        "date_from": date_from,
        "date_to": date_to,
        "page": page,
        "size": size,
        "total": total
    })

# ---------- actions already present ----------
@app.post("/api/dialer/toggle")
def api_toggle_dialer(enabled: str = Form(...)):
    set_cfg("DIALER_ENABLED", "true" if enabled == "true" else "false")
    if enabled != "true":
        qi("""UPDATE campaign_leads SET status='CALLBACK_DUE'
              WHERE status='IN_PROGRESS'""")
    return RedirectResponse("/", status_code=303)

@app.post("/api/campaigns/{cid}/activate")
def api_activate_campaign(cid: int):
    qi("UPDATE campaign_leads SET ready=false WHERE campaign_id <> %s", (cid,))
    set_cfg("ACTIVE_CAMPAIGN_ID", str(cid))
    set_cfg("DIALER_ENABLED", "true")
    return RedirectResponse(f"/campaigns/{cid}", status_code=303)

@app.post("/api/campaigns/{cid}/pause")
def api_pause_campaign(cid: int):
    qi("UPDATE campaign_leads SET ready=false WHERE campaign_id=%s", (cid,))
    any_ready = q1("SELECT EXISTS(SELECT 1 FROM campaign_leads WHERE ready=true) AS e")
    if not any_ready["e"]:
        set_cfg("DIALER_ENABLED", "false")
    return RedirectResponse(f"/campaigns/{cid}", status_code=303)

@app.post("/api/campaigns/{cid}/select")
def api_select_rules(
    cid: int,
    rule: str = Form(...),
    every_n: Optional[int] = Form(None),
    offset_start: Optional[int] = Form(None)
):
    qi("UPDATE campaign_leads SET ready=false WHERE campaign_id=%s", (cid,))

    if rule == "all":
        qi("UPDATE campaign_leads SET ready=true WHERE campaign_id=%s", (cid,))
    elif rule == "odd":
        qi("UPDATE campaign_leads SET ready=((lead_id % 2)=1) WHERE campaign_id=%s", (cid,))
    elif rule == "even":
        qi("UPDATE campaign_leads SET ready=((lead_id % 2)=0) WHERE campaign_id=%s", (cid,))
    elif rule == "every_n" and every_n and every_n > 1:
        qi("""
        WITH t AS (
          SELECT lead_id, row_number() OVER (ORDER BY lead_id) rn
          FROM campaign_leads WHERE campaign_id=%s
        )
        UPDATE campaign_leads cl
        SET ready = ((t.rn % %s) = 1)
        FROM t
        WHERE cl.campaign_id=%s AND cl.lead_id=t.lead_id
        """, (cid, every_n, cid))
    elif rule == "start_middle":
        qi("""
        WITH t AS (
          SELECT lead_id,
                 row_number() OVER (ORDER BY lead_id) rn,
                 COUNT(*) OVER () AS total
          FROM campaign_leads WHERE campaign_id=%s
        )
        UPDATE campaign_leads cl
        SET ready = (t.rn >= (t.total/2))
        FROM t
        WHERE cl.campaign_id=%s AND cl.lead_id=t.lead_id
        """, (cid, cid))
    elif rule == "offset_start" and offset_start and offset_start > 0:
        qi("""
        WITH t AS (
          SELECT lead_id, row_number() OVER (ORDER BY lead_id) rn
          FROM campaign_leads WHERE campaign_id=%s
        )
        UPDATE campaign_leads cl
        SET ready = (t.rn >= %s)
        FROM t
        WHERE cl.campaign_id=%s AND cl.lead_id=t.lead_id
        """, (cid, offset_start, cid))

    qi("""
        UPDATE campaign_leads
        SET next_action_at = NOW(),
            status = CASE WHEN status='COMPLETED' THEN 'COMPLETED' ELSE 'NEW' END
        WHERE campaign_id=%s AND ready=true AND (next_action_at IS NULL OR next_action_at > NOW())
    """, (cid,))
    return RedirectResponse(f"/campaigns/{cid}", status_code=303)

@app.post("/api/campaigns/{cid}/requeue")
def api_requeue_now(cid: int):
    qi("""
      UPDATE campaign_leads
      SET next_action_at = NOW(),
          status = CASE WHEN status='COMPLETED' THEN 'COMPLETED' ELSE 'NEW' END
      WHERE campaign_id=%s AND ready=true
    """, (cid,))
    return RedirectResponse(f"/campaigns/{cid}", status_code=303)

# ---------- exports (unchanged) ----------
@app.get("/export/leads.csv")
def export_leads(campaign_id: int = Query(...)):
    rows = qa("""
      SELECT l.id, l.company, l.email, l.phone_e164, l.city, l.state,
             cl.status, cl.ready, cl.attempts, cl.last_called_at, cl.next_action_at
      FROM campaign_leads cl
      JOIN leads l ON l.id = cl.lead_id
      WHERE cl.campaign_id=%s
      ORDER BY l.id
    """, (campaign_id,))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()) if rows else [
        "id","company","email","phone_e164","city","state","status","ready","attempts","last_called_at","next_action_at"
    ])
    w.writeheader()
    for r in rows: w.writerow({k: r[k] for k in w.fieldnames})
    buf.seek(0)
    fn = f"leads_campaign_{campaign_id}.csv"
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{fn}"'})

@app.get("/export/calls.csv")
def export_calls(campaign_id: int = Query(...)):
    rows = qa("""
      SELECT ca.id, ca.lead_id, ca.campaign_id, ca.started_at, ca.ended_at, ca.duration_sec,
             ca.disposition, ca.recording_url, l.company, l.phone_e164
      FROM call_attempts ca
      JOIN leads l ON l.id = ca.lead_id
      WHERE ca.campaign_id=%s
      ORDER BY ca.started_at DESC
    """, (campaign_id,))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()) if rows else
        ["id","lead_id","campaign_id","started_at","ended_at","duration_sec","disposition","recording_url","company","phone_e164"])
    w.writeheader()
    for r in rows: w.writerow({k: r[k] for k in w.fieldnames})
    buf.seek(0)
    fn = f"calls_campaign_{campaign_id}.csv"
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{fn}"'})
