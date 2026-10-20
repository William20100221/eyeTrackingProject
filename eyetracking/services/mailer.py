"""
Emails the tester's results: each calibration and gaze check as soon as it is saved, zipped with its
folders under data/, so the zip unzips straight into the lab's data/ folder.

The account it sends from is in mail.ini beside main.py (not in git: it holds a password):

    [smtp]
    host = smtp.gmail.com
    port = 465
    user = <the sending address>
    password = <its app password>

SSL only (port 465). A saved file waits in data/outbox.txt until an email with it has gone: a send that
fails (no internet, no mail.ini) leaves it there for the next save or the next start.
"""
from __future__ import annotations

import configparser
import io
import smtplib
import threading
import zipfile
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

from eyetracking.services import storage

TO = "zh25764@my.westlake.school.nz"
# ponytail: the password travels with the app; send from an account made only for this, an upload server if it leaks
CONFIG = Path(__file__).resolve().parents[2] / "mail.ini"
TIMEOUT_S = 60
_sending = threading.Lock()        # one email at a time: two would carry the same files
_outbox_lock = threading.Lock()    # the outbox file: held only to read or write it, never while sending


def outbox() -> Path:
    return storage.BASE_DIR / "outbox.txt"


def _waiting() -> list[str]:
    return outbox().read_text(encoding="utf-8").splitlines() if outbox().exists() else []


def queue(paths) -> None:
    """These saved files (under data/) go with the next email."""
    with _outbox_lock:
        outbox().parent.mkdir(parents=True, exist_ok=True)
        with open(outbox(), "a", encoding="utf-8") as f:
            f.writelines(f"{Path(p).relative_to(storage.BASE_DIR).as_posix()}\n" for p in paths)


def smtp_send(msg: EmailMessage) -> None:
    """Send through the account in mail.ini."""
    cfg = configparser.ConfigParser()
    if not cfg.read(CONFIG, encoding="utf-8"):
        raise RuntimeError("no mail.ini")
    smtp = cfg["smtp"]
    msg["From"] = smtp["user"]
    with smtplib.SMTP_SSL(smtp["host"], smtp.getint("port", 465), timeout=TIMEOUT_S) as server:
        server.login(smtp["user"], smtp["password"])
        server.send_message(msg)


def send_outbox() -> str:
    """Email every file waiting in the outbox, zipped. What happened, in words ("" if nothing waited)."""
    with _sending:
        with _outbox_lock:
            names = list(dict.fromkeys(_waiting()))
        files = [n for n in names if (storage.BASE_DIR / n).is_file()]   # one deleted by hand meanwhile: dropped
        if files:
            people = ", ".join(dict.fromkeys(n.split("/")[0] for n in files))
            zipped = io.BytesIO()
            with zipfile.ZipFile(zipped, "w", zipfile.ZIP_DEFLATED) as z:
                for n in files:
                    z.write(storage.BASE_DIR / n, n)
            msg = EmailMessage()
            msg["To"] = TO
            msg["Subject"] = f"Eye tracking test: {people}, {len(files)} file{'s' if len(files) != 1 else ''}"
            msg.set_content("Unzip into the lab's data/ folder.\n\n" + "\n".join(files))
            msg.add_attachment(zipped.getvalue(), maintype="application", subtype="zip",
                               filename=f"{people.replace(', ', '_')}_{datetime.now():{storage.STAMP}}.zip")
            try:
                smtp_send(msg)
            except Exception as e:         # no internet, no mail.ini, a wrong password: the files wait
                return f"Results not sent yet ({e}). They go with the next one."
        with _outbox_lock:
            left = [n for n in _waiting() if n not in names]       # queued while this email was on its way
            if left:
                outbox().write_text("".join(f"{n}\n" for n in left), encoding="utf-8")
            else:
                outbox().unlink(missing_ok=True)
        return f"Results sent to {TO}." if files else ""
