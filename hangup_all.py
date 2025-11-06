# hangup_all.py

import os
import socket

AMI_HOST = os.getenv("AMI_HOST", "127.0.0.1")
AMI_PORT = int(os.getenv("AMI_PORT", "5038"))
AMI_USER = os.getenv("AMI_USER", "dialer")
AMI_PASS = os.getenv("AMI_PASS", "SuperAMI123!")


def send(sock, headers):
    msg = "".join(f"{k}: {v}\r\n" for k, v in headers.items()) + "\r\n"
    sock.sendall(msg.encode())


def main():
    s = socket.create_connection((AMI_HOST, AMI_PORT))
    s.recv(4096)
    send(s, {"Action": "Login", "Username": AMI_USER, "Secret": AMI_PASS, "Events": "off"})
    s.recv(4096)
    send(s, {"Action": "CoreShowChannels"})
    buf = s.recv(65535).decode(errors="ignore")
    channels = []
    for line in buf.splitlines():
        if line.startswith("Channel:"):
            chan = line.split(":", 1)[1].strip()
            if chan.startswith("PJSIP/") or chan.startswith("Local/"):
                channels.append(chan)
    for chan in channels:
        send(s, {"Action": "Hangup", "Channel": chan})
        print("Hung", chan)
    send(s, {"Action": "Logoff"})
    s.close()


if __name__ == "__main__":
    main()
