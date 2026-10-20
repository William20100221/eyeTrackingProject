"""
Self-check for emailing the results (no network):  python -m eyetracking.testing.test_mailer

A saved file waits in the outbox until an email with it has gone; the zip keeps its folders under
data/; a failed send (no mail.ini, no internet) keeps it for the next one; a file queued while an
email is on its way waits for the next one; a file deleted by hand meanwhile is dropped.
"""
import io
import tempfile
import zipfile
from pathlib import Path

import eyetracking.services.storage as storage
from eyetracking.services import mailer

tmp = Path(tempfile.mkdtemp())
storage.BASE_DIR = tmp / "data"                                 # never the real data/
mailer.CONFIG = tmp / "mail.ini"                                # none: the real smtp_send fails before any network


def saved(name: str) -> Path:
    path = tmp / "data" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name)
    return path


a, b = saved("Ann/points/points_2026-10-20_120000.json"), saved("Ann/line-model/line-model_2026-10-20_120000.json")
mailer.queue([a, b])
assert mailer.send_outbox() == "Results not sent yet (no mail.ini). They go with the next one."   # no account: kept
assert mailer._waiting() == ["Ann/points/points_2026-10-20_120000.json", "Ann/line-model/line-model_2026-10-20_120000.json"]

sent = []
mailer.smtp_send = sent.append
assert mailer.send_outbox() == f"Results sent to {mailer.TO}."
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
assert mailer.send_outbox().startswith("Results sent") and mailer._waiting() == ["Bo/sweep/gaze-check_x.csv"]
d.unlink()                                                      # deleted by hand before it went
assert mailer.send_outbox() == "" and not mailer.outbox().exists() and len(sent) == 2
print("mailer ok")
