# dialer_worker.py
# -*- coding: utf-8 -*-
"""
Outbound dialer worker:
- Picks the next eligible lead for a campaign (race-safe)
- Dials via Asterisk AMI
- Logs attempts & schedules next actions
- Obeys campaigns.is_active and heartbeats every few seconds

Run modes:
  A) Spawned from API:
        python dialer_worker.py --campaign-id 4013 --selection all
  B) Manual / fallback (no args):
        python dialer_worker.py
     -> reads ACTIVE_CAMPAIGN_ID (or name) from config table.
"""

import os, sys, socket, time, json, threading, re, signal
from datetime import datetime
import psycopg2, psycopg2.extras
from dotenv import load_dotenv
import argparse
import requests

# ------------------ env ------------------
load_dotenv()
DB_URL   = os.getenv("DATABASE_URL", "postgresql://dialer:dialer@localhost:5432/dialer")
AMI_HOST = os.getenv("AMI_HOST", "127.0.0.1")
AMI_PORT = int(os.getenv("AMI_PORT", "5038"))
AMI_USER = os.getenv("AMI_USER", "dialer")
AMI_PASS = os.getenv("AMI_PASS", "SuperAMI123!")

PREWARM_URL = os.getenv("VOICE_PREWARM_URL", "http://127.0.0.1:8000/prewarm")
PREWARM_TIMEOUT = float(os.getenv("VOICE_PREWARM_TIMEOUT", "10"))
PREWARM_API_KEY = os.getenv("VOICE_PREWARM_KEY", os.getenv("API_KEY", ""))

# Context dir (AI reads JSON here)
CTX_DIR = os.getenv("CTX_DIR", "/var/tmp/dialer_ctx")
os.makedirs(CTX_DIR, mode=0o777, exist_ok=True)

# ------------------ signals ------------------
_stop_flag = False
def _graceful_stop(_sig, _frm):
    global _stop_flag
    _stop_flag = True
signal.signal(signal.SIGTERM, _graceful_stop)
signal.signal(signal.SIGINT,  _graceful_stop)

# ------------------ DB helpers ------------------
def db_connect():
    return psycopg2.connect(DB_URL, cursor_factory=psycopg2.extras.RealDictCursor)

def get_cfg_map(cur):
    cur.execute("SELECT key,value FROM config")
    return {r["key"]: r["value"] for r in cur.fetchall()}

def set_cfg(cur, key: str, value: str):
    cur.execute(
        """
        INSERT INTO config(key,value)
        VALUES(%s,%s)
        ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value
        """,
        (key, value),
    )

def clear_cfg(cur, key: str):
    cur.execute("DELETE FROM config WHERE key=%s", (key,))

def campaign_active(cur, cid: int) -> bool:
    cur.execute("SELECT is_active FROM campaigns WHERE id=%s", (cid,))
    r = cur.fetchone()
    return bool(r and r["is_active"])

def heartbeat(cur, cid: int):
    cur.execute("UPDATE campaigns SET last_heartbeat=now() WHERE id=%s", (cid,))

def resolve_active_campaign(cur, cfg):
    """Fallback when no --campaign-id is provided."""
    raw = cfg.get("ACTIVE_CAMPAIGN_ID")
    if raw:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass

    # By name/slug as a fallback
    alt = cfg.get("ACTIVE_CAMPAIGN") or cfg.get("ACTIVE_CAMPAIGN_NAME")
    if alt:
        cname = alt.strip()
        if cname:
            slug = re.sub(r"[^a-z0-9]+", "-", cname.lower()).strip("-")
            cur.execute("""
                SELECT id FROM campaigns
                WHERE lower(name)=lower(%s) OR slug=%s
                ORDER BY id DESC
                LIMIT 1
            """, (cname, slug))
            row = cur.fetchone()
            if row:
                return int(row["id"])
    return None

def prime_campaign_ready(cur, campaign_id, max_attempts):
    """Optional 'unpark' step if nothing is eligible yet."""
    if campaign_id is None:
        return 0
    cap = max_attempts if (max_attempts and max_attempts > 0) else None
    cur.execute("""
        UPDATE campaign_leads
           SET ready = true,
               status = CASE WHEN status NOT IN ('NEW','CALLBACK_DUE') THEN 'NEW' ELSE status END,
               attempts = CASE WHEN %s IS NOT NULL AND COALESCE(attempts,0) >= %s
                               THEN 0 ELSE COALESCE(attempts,0) END,
               next_action_at = CASE
                                  WHEN next_action_at IS NULL
                                    OR next_action_at <= NOW()
                                    OR (%s IS NOT NULL AND COALESCE(attempts,0) >= %s)
                                  THEN NOW()
                                  ELSE next_action_at
                                END
         WHERE campaign_id = %s
           AND status IN ('NEW','CALLBACK_DUE')
           AND (
                 ready = false
              OR next_action_at IS NULL
              OR next_action_at <= NOW()
              OR (%s IS NOT NULL AND COALESCE(attempts,0) >= %s)
           )
    """, (cap, cap, cap, cap, campaign_id, cap, cap))
    return cur.rowcount or 0

def _selection_where(selection: str, every: int | None) -> str:
    if selection == "odd":
        return " AND (l.lead_id % 2) = 1 "
    if selection == "even":
        return " AND (l.lead_id % 2) = 0 "
    if selection == "every" and every and every >= 2:
        return f" AND (l.lead_id % {int(every)}) = 0 "
    return ""  # 'all'

def claim_one(cur, cid: int, selection: str, every: int | None):
    """
    Pull exactly one eligible lead, race-safe.
    We read from your view 'leads_eligible_now' (campaign_id, lead_id, next_action_at, phone_e164, city, state, tz_name)
    and mark the link row IN_PROGRESS using SKIP LOCKED.
    """
    sel_sql = _selection_where(selection, every)
    cur.execute(f"""
      WITH one AS (
        SELECT l.*
          FROM leads_eligible_now l
         WHERE l.campaign_id = %s
               {sel_sql}
         ORDER BY l.next_action_at NULLS FIRST, l.lead_id
         LIMIT 1
         FOR UPDATE SKIP LOCKED
      )
      UPDATE campaign_leads cl
         SET status='IN_PROGRESS', last_called_at=now()
      FROM one
     WHERE cl.campaign_id = one.campaign_id AND cl.lead_id = one.lead_id
     RETURNING one.campaign_id, one.lead_id, one.phone_e164, one.city, one.state, one.tz_name;
    """, (cid,))
    row = cur.fetchone()
    if not row:
        return None
    # return as tuple to keep your originate_once() signature unchanged
    return (row["campaign_id"], row["lead_id"], row["phone_e164"], row["city"], row["state"], row["tz_name"])

# ------------------ AMI helpers (your originals) ------------------
def ami_send(sock, headers: dict):
    msg = "".join(f"{k}: {v}\r\n" for k, v in headers.items()) + "\r\n"
    sock.sendall(msg.encode())

def ami_login():
    s = socket.create_connection((AMI_HOST, AMI_PORT), timeout=5)
    s.recv(4096)  # banner
    ami_send(s, {"Action": "Login", "Username": AMI_USER, "Secret": AMI_PASS, "Events": "on"})
    s.recv(4096)
    return s

def ami_cmd(sock, action):
    ami_send(sock, action)
    buf = b""
    t0 = time.time()
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
        txt = buf.decode(errors="ignore")
        if "Complete" in txt or (time.time() - t0) > 2.0:
            return txt

def any_active_voice_channel(ami, patterns=("PJSIP/1001", "Local/7000", "Local/7100")):
    txt = ami_cmd(ami, {"Action": "CoreShowChannels"}) or ""
    for line in txt.splitlines():
        if line.startswith("Channel:"):
            for p in patterns:
                if p in line:
                    return True
    return False

def wait_until_no_active_call(ami, check_ms=400, max_wait_sec=7200):
    t0 = time.time()
    while True:
        if not any_active_voice_channel(ami):
            return
        if time.time() - t0 > max_wait_sec:
            return
        time.sleep(check_ms / 1000.0)

def hangup_active_channels(patterns=("PJSIP/", "Local/")):
    try:
        ami = ami_login()
    except Exception:
        return
    try:
        txt = ami_cmd(ami, {"Action": "CoreShowChannels"}) or ""
        for line in txt.splitlines():
            if not line.startswith("Channel:"):
                continue
            chan = line.split(":", 1)[1].strip()
            if not any(chan.startswith(p) for p in patterns):
                continue
            try:
                ami_cmd(ami, {"Action": "Hangup", "Channel": chan})
                print(f"[Dialer] Hung channel {chan}")
            except Exception:
                pass
            time.sleep(0.05)
    finally:
        try:
            ami.close()
        except Exception:
            pass

def originate_and_serial_wait(ami, channel, callerid, context, exten, timeout_ms=10000, variables=None):
    action_id = f"orig-{int(time.time()*1000)}"
    ami.settimeout((timeout_ms / 1000.0) + 2)
    action = {
        "Action": "Originate",
        "ActionID": action_id,
        "Channel": channel,
        "Context": context,
        "Exten": exten,
        "Priority": "1",
        "CallerID": callerid,
        "Timeout": str(timeout_ms),
        "Async": "true",
    }
    if variables:
        action["Variable"] = variables
    ami_send(ami, action)

    disp = "NOANSWER"
    ans_ts = None
    t0 = time.time()
    buf = b""
    try:
        while time.time() - t0 < (timeout_ms / 1000.0 + 2):
            chunk = ami.recv(4096)
            if not chunk:
                break
            buf += chunk
            txt = buf.decode(errors="ignore")
            if "OriginateResponse" in txt and "Response: Failure" in txt:
                disp = "BUSY" if ("Cause: 17" in txt or "Busy" in txt) else "FAILED"
                print(f"[AMI] Originate failure for {channel}:\n{txt.strip()}\n")
                break
            if ("Newstate: Up" in txt) or ("BridgeEnter" in txt):
                disp = "ANSWERED"
                ans_ts = time.time()
                break

        if disp == "ANSWERED":
            wait_until_no_active_call(ami)
            dur = int(max(0, time.time() - (ans_ts or time.time())))
            return disp, dur
        return disp, 0
    except socket.timeout:
        return disp, 0
    except Exception:
        return "FAILED", 0

# ------------------ JSON context & logging (your originals) ------------------
def atomic_write_json(path: str, payload: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, default=str)
    os.replace(tmp, path)

def write_context_json(payload: dict) -> str:
    if not payload:
        return ""
    fname = f"{payload['campaign_id']}_{payload['lead_id']}.json"
    final_path = os.path.join(CTX_DIR, fname)
    atomic_write_json(final_path, payload)
    atomic_write_json(os.path.join(CTX_DIR, "next.json"), {
        "context_path": final_path,
        "lead_id": payload.get("lead_id"),
        "campaign_id": payload.get("campaign_id"),
        "ts": time.time(),
    })
    return final_path

def fetch_full_raw(cur, lead_id: int, campaign_id: int):
    cur.execute("""
        SELECT
          row_to_json(l)  AS lead_full_raw,
          row_to_json(cl) AS link_raw,
          row_to_json(c)  AS campaign_raw
        FROM leads l
        JOIN campaign_leads cl ON cl.lead_id = l.id AND cl.campaign_id = %s
        JOIN campaigns c      ON c.id = cl.campaign_id
        WHERE l.id = %s
        LIMIT 1
    """, (campaign_id, lead_id))
    r = cur.fetchone()
    if r:
        return {
            "lead_id": lead_id,
            "campaign_id": campaign_id,
            "lead_full_raw": r["lead_full_raw"],
            "link_raw": r["link_raw"],
            "campaign_raw": r["campaign_raw"],
        }
    cur.execute("SELECT row_to_json(l) AS j FROM leads l WHERE l.id=%s LIMIT 1", (lead_id,))
    r2 = cur.fetchone()
    return {
        "lead_id": lead_id,
        "campaign_id": campaign_id,
        "lead_full_raw": (r2["j"] if r2 else None),
        "link_raw": None,
        "campaign_raw": None,
    }

def log_attempt(cur, lead_id, campaign_id, disp, duration=None, trunk=None, sip_call_id=None, rec_url=None, notes=None):
    cur.execute("""
      INSERT INTO call_attempts(lead_id,campaign_id,started_at,disposition,duration_sec,trunk,sip_call_id,recording_url,notes)
      VALUES (%s,%s,now(),%s,%s,%s,%s,%s,%s)
    """, (lead_id, campaign_id, disp, duration, trunk, sip_call_id, rec_url, notes))

def schedule_next(cur, lead_id, campaign_id, base_disp, mode, test_redial_sec):
    if mode == "TEST":
        cur.execute("""
          UPDATE campaign_leads
             SET attempts = COALESCE(attempts,0) + 1,
                 last_called_at = now(),
                 next_action_at = now() + (%s || ' seconds')::interval,
                 status = 'CALLBACK_DUE',
                 ready = true
           WHERE lead_id=%s AND campaign_id=%s
        """, (str(test_redial_sec), lead_id, campaign_id))
        return

    mapping = {"NOANSWER": "24 hours", "BUSY": "2 hours", "FAILED": "24 hours", "VOICEMAIL": "48 hours"}
    if base_disp in ("ANSWERED", "COMPLETED"):
        cur.execute("""
          UPDATE campaign_leads
             SET status='COMPLETED', ready=false, last_called_at=now()
           WHERE lead_id=%s AND campaign_id=%s
        """, (lead_id, campaign_id))
        return

    delta = mapping.get(base_disp, "24 hours")
    cur.execute(f"""
      UPDATE campaign_leads
         SET attempts = COALESCE(attempts,0) + 1,
             last_called_at = now(),
             next_action_at = now() + INTERVAL '{delta}',
             status = 'CALLBACK_DUE'
       WHERE lead_id=%s AND campaign_id=%s
    """, (lead_id, campaign_id))

# ------------------ core originate (your original) ------------------
def originate_once(cur, cfg, job):
    campaign_id, lead_id, phone, city, state, tz = job

    mode            = cfg.get("DIAL_MODE", "TEST").upper()
    teststr         = cfg.get("DIAL_STRING_TEST", "Local/7100@default")
    livestr         = cfg.get("DIAL_STRING_LIVE", "PJSIP/${PHONE_E164}@carrier_us_1")
    caller          = cfg.get("CALLER_ID_E164", "Cloumen Dialer")
    timeout_ms      = int(cfg.get("DIAL_TIMEOUT_MS", "25000"))
    test_redial_sec = int(cfg.get("TEST_REDIAL_SEC", "10"))

    channel = teststr if mode == "TEST" else livestr.replace("${PHONE_E164}", phone or "")

    ctx_payload  = fetch_full_raw(cur, lead_id, campaign_id)
    ctx_path     = write_context_json(ctx_payload)
    ctx_path_var = ctx_path.replace("\\", "/")

    if PREWARM_URL:
        prewarm_body = {
            "campaign_id": campaign_id,
            "lead_id": lead_id,
            "context_path": ctx_path_var,
            "lead_payload": ctx_payload,
        }
        headers = {}
        if PREWARM_API_KEY:
            headers["x-api-key"] = PREWARM_API_KEY
        try:
            resp = requests.post(
                PREWARM_URL,
                json=prewarm_body,
                headers=headers,
                timeout=PREWARM_TIMEOUT,
            )
            if os.getenv("PREPARED_DEBUG", "0") == "1":
                print(f"[Dialer] Prewarm response {resp.status_code}: {resp.text[:200]}")
            if resp.status_code >= 400:
                raise RuntimeError(f"Prewarm HTTP {resp.status_code}: {resp.text}")
        except Exception as exc:
            if os.getenv("VOICE_PREWARM_SILENT", "0") != "1":
                print(f"[Dialer] Prewarm call failed: {exc}")

    lead_raw = ctx_payload.get("lead_full_raw") or {}
    company = lead_raw.get("company") or lead_raw.get("business_name") or ""
    contact = lead_raw.get("contact") or " ".join(filter(None, [lead_raw.get("first_name"), lead_raw.get("last_name")]))
    label = company or contact or phone or f"Lead {lead_id}"

    rec_basename = f"cid{campaign_id}_lid{lead_id}_{int(time.time())}"
    vars_str = (
        f"LEAD_ID={lead_id}"
        f"|CAMPAIGN_ID={campaign_id}"
        f"|PHONE_E164={phone or ''}"
        f"|CITY={city or ''}"
        f"|STATE={state or ''}"
        f"|TZ_NAME={tz or ''}"
        f"|CONTEXT_JSON={ctx_path_var}"
        f"|REC_BASENAME={rec_basename}"
    )

    print(f"[{threading.current_thread().name}] PRECALL campaign={campaign_id} lead={lead_id} "
          f"target='{label}' phone={phone or 'N/A'} city={city or ''} state={state or ''} tz={tz or ''}")
    print(f"[{threading.current_thread().name}] Context -> {ctx_path}")

    ami = None
    disp = "NOANSWER"; dur = 0
    try:
        ami = ami_login()
        disp, dur = originate_and_serial_wait(
            ami, channel, callerid=caller, context="default", exten="7000",
            timeout_ms=timeout_ms, variables=vars_str
        )
    except Exception:
        disp = "FAILED"
    finally:
        try:
            if ami:
                ami.close()
        except Exception:
            pass

    print(f"[{threading.current_thread().name}] Lead {lead_id} -> {disp} ({dur}s)")
    rec_url = f"/recordings/{rec_basename}.wav"
    log_attempt(cur, lead_id, campaign_id, disp, duration=dur, rec_url=rec_url)
    schedule_next(cur, lead_id, campaign_id, disp, mode, test_redial_sec)

# ------------------ worker loop ------------------
def worker_loop(cid: int, selection: str, every: int | None, threads: int):
    """
    Keep threads small (1-3) until you scale. All threads share the same CID.
    """
    def _thread_main(idx: int):
        name = f"W{idx}"
        while not _stop_flag:
            try:
                with db_connect() as conn, conn.cursor() as cur:
                    # quick exit if campaign flipped off
                    if not campaign_active(cur, cid):
                        conn.commit()
                        print(f"[{name}] Campaign {cid} inactive; exiting.")
                        return

                    # heartbeat
                    heartbeat(cur, cid)
                    conn.commit()

                    # read config + optional unpark
                    cfg = get_cfg_map(cur)
                    # Optional: prime if queue looks empty after a claim miss
                    job = claim_one(cur, cid, selection, every)
                    if not job:
                        # try a one-time unpark:
                        max_attempts = None
                        try:
                            max_attempts = int(cfg.get("MAX_ATTEMPTS")) if cfg.get("MAX_ATTEMPTS") else None
                        except (TypeError, ValueError):
                            max_attempts = None
                        primed = prime_campaign_ready(cur, cid, max_attempts)
                        conn.commit()
                        if primed:
                            print(f"[{name}] Unparked {primed} lead(s) in campaign {cid}.")
                        else:
                            # small idle
                            time.sleep(2.0)
                        continue

                    print(f"[{name}] Calling lead {job[1]} ({job[2]}) in campaign {job[0]}...")
                    originate_once(cur, cfg, job)
                    conn.commit()

            except psycopg2.Error as e:
                # transient DB issue
                print(f"[{name}] DB error: {e}")
                time.sleep(1.0)
            except Exception as e:
                print(f"[{name}] Error: {e}")
                time.sleep(0.5)

    threads = max(1, int(threads))
    ts = []
    for i in range(threads):
        t = threading.Thread(target=_thread_main, args=(i+1,), daemon=True, name=f"W{i+1}")
        t.start()
        ts.append(t)

    print(f"Dialer worker started for campaign {cid} with {threads} thread(s); selection='{selection}' every={every}")
    try:
        while not _stop_flag and any(t.is_alive() for t in ts):
            time.sleep(1.0)
    finally:
        try:
            hangup_active_channels()
        except Exception:
            pass
        print("Dialer worker stopping...")

# ------------------ entrypoint ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-id", type=int, help="Campaign ID to run")
    ap.add_argument("--selection", choices=["all", "odd", "even", "every"], default="all")
    ap.add_argument("--every", type=int, default=None, help="N for 'every' selection")
    ap.add_argument("--parallel", type=int, default=1, help="number of worker threads")
    ap.add_argument("--stop", action="store_true", help="Stop the active worker for a campaign")
    args = ap.parse_args()

    cid = args.campaign_id
    selection = args.selection
    every = args.every
    parallel = args.parallel

    # If no --campaign-id is passed, fall back to config
    needs_resolution = cid is None
    if needs_resolution:
        with db_connect() as conn, conn.cursor() as cur:
            cfg = get_cfg_map(cur)
            cid = resolve_active_campaign(cur, cfg)
            if cid is None:
                if args.stop:
                    print("No active campaign found to stop.")
                    return
                print("No --campaign-id provided and no ACTIVE_CAMPAIGN[_ID] found in config. Exiting.")
                return
            # default selection from campaigns table if stored
            cur.execute("SELECT selection_rule, selection_n FROM campaigns WHERE id=%s", (cid,))
            r = cur.fetchone()
            if r and r["selection_rule"]:
                selection = r["selection_rule"]
                every = r["selection_n"]

    if args.stop:
        with db_connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT worker_pid FROM campaigns WHERE id=%s", (cid,))
            row = cur.fetchone()
            if not row:
                print(f"Campaign {cid} not found.")
                return
            pid = row.get("worker_pid")
            cur.execute(
                "UPDATE campaigns SET is_active=false, worker_pid=NULL WHERE id=%s",
                (cid,),
            )
            try:
                cur.execute("SELECT value FROM config WHERE key='ACTIVE_CAMPAIGN_ID'")
                cfg_row = cur.fetchone()
                if cfg_row and cfg_row.get("value") == str(cid):
                    clear_cfg(cur, "ACTIVE_CAMPAIGN_ID")
            except Exception:
                pass
            conn.commit()

        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
        try:
            hangup_active_channels()
        except Exception:
            pass
        print(f"Stop signal issued for campaign {cid}.")
        return

    # safety for every
    if selection == "every" and (every is None or every < 2):
        print("Selection 'every' requires --every >= 2; falling back to 'all'.")
        selection, every = "all", None

    try:
        with db_connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE campaigns
                   SET is_active=true,
                       worker_pid=%s,
                       selection_rule=%s,
                       selection_n=%s,
                       last_heartbeat=now()
                 WHERE id=%s
                """,
                (os.getpid(), selection, every, cid),
            )
            try:
                set_cfg(cur, "ACTIVE_CAMPAIGN_ID", str(cid))
            except Exception:
                pass
            conn.commit()
    except Exception as e:
        print(f"Failed to mark campaign {cid} active: {e}")

    worker_loop(cid, selection, every, parallel)

if __name__ == "__main__":
    main()
