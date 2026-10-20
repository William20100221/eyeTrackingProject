"""
Self-check for emailing the results (no network):  python -m eyetracking.testing.test_mailer

A queued file waits in the outbox until an email with it has gone; the zip keeps its folders under
data/; a failed send (no mail.ini, no internet) keeps it for the next one and puts the zip on the
Desktop to send by hand; a network that blocks port 465 is sent through 587 (STARTTLS); a file queued
while an email is on its way waits for the next one; a file deleted by hand meanwhile is dropped.
"""
import io
import tempfile
import zipfile
from pathlib import Path

import eyetracking.services.storage as storage
from eyetracking.services import mailer

tmp = Path(tempfile.mkdtemp())
storage.BASE_DIR = tmp / "data"                                 # never the real data/
mailer.CONFIG = tmp / "mail.ini"                                # none yet: the real smtp_send fails before any network
mailer.FALLBACK_DIR = tmp                                       # never the real Desktop
real_send = mailer.smtp_send


def saved(name: str) -> Path:
    path = tmp / "data" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name)
    return path


a, b = saved("Ann/points/points_2026-10-20_120000.json"), saved("Ann/line-model/line-model_2026-10-20_120000.json")
mailer.queue([a, b])
copy = tmp / "Eye tracking results - Ann.zip"
assert mailer.send_outbox() == f"Not emailed (no mail.ini beside main.py). Please email {copy} to {mailer.TO}."
assert mailer._waiting() == ["Ann/points/points_2026-10-20_120000.json", "Ann/line-model/line-model_2026-10-20_120000.json"]
with zipfile.ZipFile(copy) as z:                                # the copy to send by hand: the same zip
    assert z.namelist() == mailer._waiting()

sent = []
mailer.smtp_send = sent.append
assert mailer.send_outbox() == f"Results emailed to {mailer.TO}. Thank you!"
msg = sent[0]
assert msg["To"] == mailer.TO and msg["Subject"] == "Eye tracking test: Ann, 2 files"
attachment = next(msg.iter_attachments())
assert attachment.get_filename().startswith("Ann_2026") and attachment.get_filename().endswith(".zip")
with zipfile.ZipFile(io.BytesIO(attachment.get_content())) as z:
    assert z.namelist() == ["Ann/points/points_2026-10-20_120000.json", "Ann/line-model/line-model_2026-10-20_120000.json"]
    assert z.read("Ann/points/points_2026-10-20_120000.json") == b"Ann/points/points_2026-10-20_120000.json"
assert not mailer.outbox().exists() and mailer.send_outbox() == ""                # nothing waits: no email

c, d = saved("Bo/sweep/sweep_2026-10-20_130000.json"), saved("Bo/sweep/gaze-check_x.csv")
mailer.smtp_send = lambda m: (mailer.queue([d]), sent.append(m))  # d saved while c's email is on its way
mailer.queue([c])
assert mailer.send_outbox().startswith("Results emailed") and mailer._waiting() == ["Bo/sweep/gaze-check_x.csv"]
d.unlink()                                                      # deleted by hand before it went
assert mailer.send_outbox() == "" and not mailer.outbox().exists() and len(sent) == 2


# port 465 blocked on this network: the same email through 587 with STARTTLS
class Blocked:
    def __init__(self, host, port, **kw):
        raise ConnectionRefusedError("465 blocked")


class Server:
    def __init__(self, host, port, **kw):
        self.log = [("connect", host, port)]
        calls.append(self.log)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def starttls(self, context):
        self.log.append("starttls")

    def login(self, user, password):
        self.log.append(("login", user, password))

    def send_message(self, msg):
        self.log.append(("send", msg["From"], msg["To"]))


calls = []
mailer.CONFIG.write_text("[smtp]\nhost = smtp.example.com\nuser = sender@example.com\npassword = secret\n")
mailer.smtplib.SMTP_SSL, mailer.smtplib.SMTP = Blocked, Server
mailer.smtp_send = real_send
mailer.queue([c])
assert mailer.send_outbox().startswith("Results emailed"), calls
assert calls == [[("connect", "smtp.example.com", 587), "starttls", ("login", "sender@example.com", "secret"),
                  ("send", "sender@example.com", mailer.TO)]], calls
print("mailer ok")
