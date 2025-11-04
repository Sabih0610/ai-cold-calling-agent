# dialer_service.py
import os, socket, time, json, threading, re
import psycopg2, psycopg2.extras
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()
DB_URL   = os.getenv("DATABASE_URL", "postgresql://dialer:dialer@localhost:5432/dialer")
AMI_HOST = os.getenv("AMI_HOST", "127.0.0.1")
AMI_PORT = int(os.getenv("AMI_PORT", "5038"))
AMI_USER = os.getenv("AMI_USER", "dialer")
AMI_PASS = os.getenv("AMI_PASS", "SuperAMI123!")

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

def originate_and_serial_wait(ami, channel, callerid, context, exten, timeout_ms=10000):
    """
    Originate, classify quickly, but if answered, WAIT until the
    channel is gone before returning (strict single-call).
    """
    action_id = f"orig-{int(time.time()*1000)}"
    ami.settimeout((timeout_ms/1000.0) + 2)
    ami_send(ami, {
        "Action": "Originate",
        "ActionID": action_id,
        "Channel": channel,
        "Context": context,
        "Exten": exten,
        "Priority": "1",
        "CallerID": callerid,
        "Timeout": str(timeout_ms),
        "Async": "true",
    })

    disp = "NOANSWER"
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
                break
            if ("Newstate: Up" in txt) or ("BridgeEnter" in txt):
                disp = "ANSWERED"
                break

        # STRICT SERIAL: if answered, block until call ends
        if disp == "ANSWERED":
            wait_until_no_active_call(ami)
        return disp
    except socket.timeout:
        return disp
    except Exception:
        return "FAILED"


############################################################
# --- Database helpers ------------------------------------
############################################################

def get_cfg_map(cur):
    cur.execute("SELECT key,value FROM config")
    return {k:v for k,v in cur.fetchall()}

def claim_one(cur):
    cur.execute("""
      WITH one AS (
        SELECT l.* FROM leads_eligible_now l
        ORDER BY next_action_at NULLS FIRST
        FOR UPDATE SKIP LOCKED
        LIMIT 1
      )
      UPDATE campaign_leads cl
      SET status='IN_PROGRESS'
      FROM one
      WHERE cl.campaign_id = one.campaign_id AND cl.lead_id = one.lead_id
      RETURNING one.campaign_id, one.lead_id, one.phone_e164, one.city, one.state, one.tz_name;
    """)
    return cur.fetchone()

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
    mode    = cfg.get("DIAL_MODE","TEST").upper()
    teststr = cfg.get("DIAL_STRING_TEST","Local/7100@default")
    livestr = cfg.get("DIAL_STRING_LIVE","PJSIP/${PHONE_E164}@carrier_us_1")
    caller  = cfg.get("CALLER_ID_E164","Cloumen Dialer")
    timeout_ms = int(cfg.get("DIAL_TIMEOUT_MS","25000"))
    test_redial_sec = int(cfg.get("TEST_REDIAL_SEC","10"))

    channel = teststr if mode=="TEST" else livestr.replace("${PHONE_E164}", phone or "")
    vars_str = f"LEAD_ID={lead_id}|CAMPAIGN_ID={campaign_id}|PHONE_E164={phone or ''}|CITY={city or ''}|STATE={state or ''}"

    ami = None
    disp = "NOANSWER"
    try:
        ami = ami_login()

        # Use the new strict serial originate
        disp = originate_and_serial_wait(
            ami, channel, callerid=caller, context="default", exten="7000", timeout_ms=timeout_ms
        )

    except Exception:
        disp = "FAILED"
    finally:
        try:
            if ami:
                ami.close()
        except Exception:
            pass

    print(f"[{threading.current_thread().name}] Lead {lead_id} -> {disp}")
    log_attempt(cur, lead_id, campaign_id, disp)
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
            while True:
                cfg = get_cfg_map(cur)
                if cfg.get("DIALER_ENABLED","true").lower() != "true":
                    time.sleep(2)
                    continue

                job = claim_one(cur)
                if not job:
                    time.sleep(1.0)
                    continue

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

if __name__ == "__main__":
    main()
