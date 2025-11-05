# dialer_worker.py
import os, socket, time, json, threading, re
import psycopg2, psycopg2.extras
from dotenv import load_dotenv

load_dotenv()
DB_URL   = os.getenv("DATABASE_URL", "postgresql://dialer:dialer@localhost:5432/dialer")
AMI_HOST = os.getenv("AMI_HOST", "127.0.0.1")
AMI_PORT = int(os.getenv("AMI_PORT", "5038"))
AMI_USER = os.getenv("AMI_USER", "dialer")
AMI_PASS = os.getenv("AMI_PASS", "SuperAMI123!")

# ===== Context dir for full raw JSON (AI reads from here) =====
CTX_DIR = os.getenv("CTX_DIR", "/var/tmp/dialer_ctx")
os.makedirs(CTX_DIR, mode=0o777, exist_ok=True)

############################################################
# --- AMI helpers -----------------------------------------
############################################################

def ami_send(sock, headers: dict):
    msg = "".join(f"{k}: {v}\r\n" for k,v in headers.items()) + "\r\n"
    sock.sendall(msg.encode())

def ami_login():
    s = socket.create_connection((AMI_HOST, AMI_PORT), timeout=5)
    s.recv(4096)  # banner
    ami_send(s, {"Action":"Login","Username":AMI_USER,"Secret":AMI_PASS,"Events":"on"})
    s.recv(4096)
    return s

def ami_cmd(sock, action):
    """send a manager action and return the raw text until Complete"""
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
    """Return True if any live channel matches our test path."""
    txt = ami_cmd(ami, {"Action": "CoreShowChannels"})
    for line in txt.splitlines():
        if line.startswith("Channel:"):
            for p in patterns:
                if p in line:
                    return True
    return False

def wait_until_no_active_call(ami, check_ms=400, max_wait_sec=7200):
    """Block until there are no active channels that match our patterns."""
    t0 = time.time()
    while True:
        if not any_active_voice_channel(ami):
            return
        if time.time() - t0 > max_wait_sec:
            return
        time.sleep(check_ms / 1000.0)


def hangup_active_channels(patterns=("PJSIP/", "Local/")):
    """Attempt to hang up any lingering channels matching the given prefixes."""
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
    """
    Originate, classify quickly, but if answered, WAIT until the
    channel is gone before returning (strict single-call).
    """
    action_id = f"orig-{int(time.time()*1000)}"
    ami.settimeout((timeout_ms/1000.0) + 2)
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
    # carry variables to exten 7000 (optional but handy)
    if variables:
        # AMI accepts pipe-delimited k=v pairs in a single Variable header
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

        # STRICT SERIAL: if answered, block until call ends
        if disp == "ANSWERED":
            wait_until_no_active_call(ami)
            dur = int(max(0, time.time() - (ans_ts or time.time())))
            return disp, dur
        return disp, 0
    except socket.timeout:
        return disp, 0
    except Exception:
        return "FAILED", 0

############################################################
# --- Database helpers ------------------------------------
############################################################

def get_cfg_map(cur):
    cur.execute("SELECT key,value FROM config")
    return {k:v for k,v in cur.fetchall()}


def resolve_active_campaign(cur, cfg):
    raw = cfg.get("ACTIVE_CAMPAIGN_ID")
    if raw:
        try:
            return int(raw), True
        except (TypeError, ValueError):
            pass

    alt = cfg.get("ACTIVE_CAMPAIGN") or cfg.get("ACTIVE_CAMPAIGN_NAME")
    for candidate in filter(None, (raw, alt)):
        cname = candidate.strip()
        if not cname:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", cname.lower()).strip("-")
        cur.execute(
            """
            SELECT id FROM campaigns
            WHERE lower(name) = lower(%s) OR slug = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            (cname, slug)
        )
        row = cur.fetchone()
        if row:
            return int(row[0]), True
    return None, False

def prime_campaign_ready(cur, campaign_id, max_attempts):
    if campaign_id is None:
        return 0
    cap = max_attempts if max_attempts and max_attempts > 0 else None
    cur.execute(
        """
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
        """,
        (cap, cap, cap, cap, campaign_id, cap, cap)
    )
    return cur.rowcount or 0


def claim_one(cur, active_campaign_id=None):
    cur.execute("""
      WITH one AS (
        SELECT l.* FROM leads_eligible_now l
        WHERE (%s IS NULL OR l.campaign_id = %s)
        ORDER BY next_action_at NULLS FIRST
        FOR UPDATE SKIP LOCKED
        LIMIT 1
      )
      UPDATE campaign_leads cl
      SET status='IN_PROGRESS'
      FROM one
      WHERE cl.campaign_id = one.campaign_id AND cl.lead_id = one.lead_id
      RETURNING one.campaign_id, one.lead_id, one.phone_e164, one.city, one.state, one.tz_name;
    """, (active_campaign_id, active_campaign_id))
    return cur.fetchone()

def fetch_full_raw(cur, lead_id: int, campaign_id: int):
    """
    Full raw rows as JSON:
      - lead_full_raw:   row_to_json(leads.*)
      - link_raw:        row_to_json(campaign_leads.*)
      - campaign_raw:    row_to_json(campaigns.*)
    """
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
            "lead_full_raw": r[0],
            "link_raw": r[1],
            "campaign_raw": r[2],
        }
    # fallback: at least return the lead
    cur.execute("SELECT row_to_json(l) FROM leads l WHERE l.id=%s LIMIT 1", (lead_id,))
    r2 = cur.fetchone()
    return {
        "lead_id": lead_id,
        "campaign_id": campaign_id,
        "lead_full_raw": (r2[0] if r2 else None),
        "link_raw": None,
        "campaign_raw": None,
    }

def atomic_write_json(path: str, payload: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, default=str)
    for attempt in range(4):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except PermissionError:
                pass
            time.sleep(0.05 * (attempt + 1))
    # last attempt (will raise if still locked)
    os.replace(tmp, path)

def write_context_json(payload: dict) -> str:
    """
    Write the full raw payload to a per-lead file and also refresh a serial pointer.
    Returns the absolute path to the per-lead JSON.
    """
    if not payload:
        return ""
    fname = f"{payload['campaign_id']}_{payload['lead_id']}.json"
    final_path = os.path.join(CTX_DIR, fname)

    # 1) per-lead context file (atomic)
    atomic_write_json(final_path, payload)

    # 2) serial pointer for simple AIs: next.json (just a pointer + small meta)
    latest_ptr = os.path.join(CTX_DIR, "next.json")
    atomic_write_json(latest_ptr, {
        "context_path": final_path,
        "lead_id": payload.get("lead_id"),
        "campaign_id": payload.get("campaign_id"),
        "ts": time.time(),
    })

    return final_path

def log_attempt(cur, lead_id, campaign_id, disp, duration=None, trunk=None, sip_call_id=None, rec_url=None, notes=None):
    cur.execute("""
      INSERT INTO call_attempts(lead_id,campaign_id,started_at,disposition,duration_sec,trunk,sip_call_id,recording_url,notes)
      VALUES (%s,%s,now(),%s,%s,%s,%s,%s,%s)
    """, (lead_id,campaign_id,disp,duration,trunk,sip_call_id,rec_url,notes))

def schedule_next(cur, lead_id, campaign_id, base_disp, mode, test_redial_sec):
    """LIVE vs TEST requeue logic"""
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

    m = {"NOANSWER": "24 hours", "BUSY": "2 hours", "FAILED": "24 hours", "VOICEMAIL": "48 hours"}
    if base_disp in ("ANSWERED", "COMPLETED"):
        cur.execute("""
          UPDATE campaign_leads
          SET status='COMPLETED', ready=false, last_called_at=now()
          WHERE lead_id=%s AND campaign_id=%s
        """, (lead_id, campaign_id))
        return
    delta = m.get(base_disp, "24 hours")
    cur.execute(f"""
      UPDATE campaign_leads
      SET attempts = COALESCE(attempts,0) + 1,
          last_called_at = now(),
          next_action_at = now() + INTERVAL '{delta}',
          status = 'CALLBACK_DUE'
      WHERE lead_id=%s AND campaign_id=%s
    """, (lead_id, campaign_id))

############################################################
# --- Core calling logic ----------------------------------
############################################################

def originate_once(cur, cfg, job):
    campaign_id, lead_id, phone, city, state, tz = job

    mode            = cfg.get("DIAL_MODE","TEST").upper()
    teststr         = cfg.get("DIAL_STRING_TEST","Local/7100@default")
    livestr         = cfg.get("DIAL_STRING_LIVE","PJSIP/${PHONE_E164}@carrier_us_1")
    caller          = cfg.get("CALLER_ID_E164","Cloumen Dialer")
    timeout_ms      = int(cfg.get("DIAL_TIMEOUT_MS","25000"))
    test_redial_sec = int(cfg.get("TEST_REDIAL_SEC","10"))

    channel = teststr if mode=="TEST" else livestr.replace("${PHONE_E164}", phone or "")

    # ===== PRECALL: fetch full raw & write JSON BEFORE dialing =====
    ctx_payload = fetch_full_raw(cur, lead_id, campaign_id)
    ctx_path    = write_context_json(ctx_payload)
    ctx_path_var = ctx_path.replace("\\", "/")

    # optional variables to the call (kept for convenience)
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

    print(f"[{threading.current_thread().name}] PRECALL lead={lead_id} wrote {ctx_path} "
          f"(company={ctx_payload.get('lead_full_raw',{}).get('company','')})")

    ami = None
    disp = "NOANSWER"; dur = 0
    try:
        ami = ami_login()
        # Strict serial originate (unchanged) + variables
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

############################################################
# --- Worker thread ---------------------------------------
############################################################

def worker_thread(idx):
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False
    name = f"W{idx}"
    try:
        with conn, conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            primed_last_loop = False
            while True:
                cfg = get_cfg_map(cur)
                active_cid, strict = resolve_active_campaign(cur, cfg)
                max_attempts = None
                try:
                    max_attempts = int(cfg.get("MAX_ATTEMPTS")) if cfg.get("MAX_ATTEMPTS") else None
                except (TypeError, ValueError):
                    max_attempts = None
                if cfg.get("DIALER_ENABLED","true").lower() != "true":
                    time.sleep(2)
                    continue

                job = claim_one(cur, active_cid)
                if not job:
                    if active_cid is not None and not primed_last_loop:
                        primed = prime_campaign_ready(cur, active_cid, max_attempts)
                        if primed:
                            conn.commit()
                            primed_last_loop = True
                            print(f"[{name}] Unparked {primed} lead(s) in campaign {active_cid}.")
                            continue
                    primed_last_loop = False
                    target = active_cid if active_cid is not None else "ANY"
                    print(f"[{name}] No eligible leads for campaign {target}."
                          " Check ready/status/next_action_at parameters.")
                    if active_cid is not None:
                        cur.execute(
                            """
                            SELECT lead_id, ready, status, attempts, next_action_at
                            FROM campaign_leads
                            WHERE campaign_id = %s
                            ORDER BY lead_id
                            LIMIT 5
                            """,
                            (active_cid,)
                        )
                        sample = cur.fetchall()
                        for row in sample or []:
                            print(f"    lead {row['lead_id']}: ready={row['ready']} status={row['status']} attempts={row['attempts']} next_action_at={row['next_action_at']}")
                    time.sleep(1.0)
                    continue

                primed_last_loop = False

                print(f"[{name}] Calling lead {job[1]} ({job[2]}) in campaign {job[0]}...")
                originate_once(cur, cfg, job)
                conn.commit()
    finally:
        conn.close()

############################################################
# --- Bootstrap / main ------------------------------------
############################################################

def main():
    conn = psycopg2.connect(DB_URL)
    with conn, conn.cursor() as cur:
        cfg = get_cfg_map(cur)
        conc = int(cfg.get("CONCURRENCY","1"))
    conn.close()

    threads = []
    for i in range(conc):
        t = threading.Thread(target=worker_thread, args=(i+1,), daemon=True, name=f"W{i+1}")
        t.start()
        threads.append(t)

    print(f"Dialer service running with {len(threads)} worker(s). Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Stopping...")
        try:
            hangup_active_channels()
        except Exception:
            pass

if __name__ == "__main__":
    main()
