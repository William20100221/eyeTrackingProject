"""
Self-check (no webcam):  python -m eyetracking.testing.test_lab_data

1. Calibration: Space starts, Space pauses (timer stops, nothing recorded), Space carries on, as often
   as you like; Esc leaves at any time
2. Data per method: data/<method>/<method>_<date>_<time>.json, each file carrying its method and
   grid; a file in words ("dot (2, 1) of the 3x3 grid"); disable (the method's disabled/), enable, remove
   samples / the file (the original kept in the method's trash/); the Testing page lists the chosen method only
3. Gaze checks in their method's folder (nearest dot of THAT method's grid), blink scores in blink-test/;
   the other branches' folders are never read or moved
4. Folders cut out come back empty; person mode puts the same folders under data/<name>/
5. Clear data: one person's files (or everyone's) go to data/trash/<date>_<time>/, their empty folders come back
6. Editing several at once: Move to another person (disabled stays disabled, nothing overwritten), Remove,
   then Restore from Removed (an edited file's original only after a yes)
"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QEvent, QSettings, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

import eyetracking.services.storage as storage
import eyetracking.ui.control_window as cw
from eyetracking.core.calibration import describe_target, point_grid
from eyetracking.core.features import N_FEATURES
from eyetracking.core.methods import METHOD_BY_ID
from eyetracking.core.models import CalibrationRecording, CalibrationSample
from eyetracking.ui.calibration_window import CalibrationWindow

qt = QApplication([])
tmp = Path(tempfile.mkdtemp())
cw.QSettings = lambda *_: QSettings(str(tmp / "settings.ini"), QSettings.IniFormat)   # never the user's settings
storage.BASE_DIR = tmp                                                                 # never the real data/
storage.use_person("")                                                                 # straight under it


def key(window, k):
    QApplication.sendEvent(window, QKeyEvent(QEvent.Type.KeyPress, k, Qt.KeyboardModifier.NoModifier))


# ---- 1. pause ------------------------------------------------------------
window = CalibrationWindow(METHOD_BY_ID["sweep"], get_features=lambda: (0.5,) * N_FEATURES)
key(window, Qt.Key.Key_Space)                                  # start
assert window._timer.isActive() and not window.paused and window._phase == "sweep"
for _ in range(3):                                             # pause / carry on, again and again
    key(window, Qt.Key.Key_Space)
    assert window.paused and not window._timer.isActive()
    key(window, Qt.Key.Key_Space)
    assert not window.paused and window._timer.isActive() and window._phase == "sweep"
key(window, Qt.Key.Key_Space)                                  # paused...
left = []
window.cancelled.connect(lambda: left.append(True))
key(window, Qt.Key.Key_Escape)                                 # ...and Esc still leaves
assert left and not window._timer.isActive()                   # nothing collected yet: no question asked

window = CalibrationWindow(METHOD_BY_ID["original-3x3"], get_features=lambda: (0.5,) * N_FEATURES)
key(window, Qt.Key.Key_Space)
for _ in range(6):                                             # 6 frames on the first dot: 3 kept, 3 held back for blinks
    window._session.add_sample((0.5,) * N_FEATURES)
asked, answer, left = [], [False], []
window.confirm_leave = lambda n: (asked.append(n), answer[0])[1]
window.cancelled.connect(lambda: left.append(True))
key(window, Qt.Key.Key_Escape)                                 # samples collected: are you sure?
assert asked == [3] and not left and window.paused and not window._timer.isActive()   # No: stays, paused
key(window, Qt.Key.Key_Space)
assert not window.paused                                       # Space carries on as usual
answer[0] = True
key(window, Qt.Key.Key_Escape)                                 # Yes: leaves, nothing saved
assert asked == [3, 3] and left and not window._timer.isActive()

# ---- 2. data per method ----------------------------------------------------
assert describe_target(*point_grid(3)[1], 3) == "dot (2, 1) of the 3x3 grid"      # 2nd dot = column 2, row 1
assert describe_target(0.43, 0.61) == "the point 43% across, 61% down"
sample = lambda x, y: CalibrationSample((0.1, 0.2) + (0.0,) * (N_FEATURES - 2), x, y)
x, y = point_grid(3)[1]
path = storage.save_calibration(CalibrationRecording((), (), (sample(x, y),) * 2), "original-3x3")
assert path.parent == tmp / "original-3x3" and path.name.startswith("original-3x3_20"), path
assert storage.method_of(path) == "original-3x3" and storage.file_date(path).year >= 2026
text = storage.calibration_text(path)
assert text.startswith("Original 3×3, ") and "The 3x3 dot grid: 2 samples" in text, text
assert "dot (2, 1) of the 3x3 grid: 2 samples" in text and "#2: eyes: iris 10% across" in text, text
other = storage.save_calibration(CalibrationRecording((), (), (sample(*point_grid(5)[1]),)), "original-5x5")
assert "dot (2, 1) of the 5x5 grid" in storage.calibration_text(other)
assert storage.list_calibrations("original-3x3") == [path] and storage.list_calibrations("original-5x5") == [other]
assert storage.list_calibrations("sweep") == [] and storage.calibration_usable(path)
broken = storage.method_dir("original-3x3") / "original-3x3_2026-01-01_000000.json"
broken.write_text("{}")
assert "Can't show this file" in storage.calibration_text(broken) and not storage.calibration_usable(broken)
off = storage.set_calibration_enabled(path, False)
assert off == storage.method_dir("original-3x3") / "disabled" / path.name and storage.method_of(off) == "original-3x3"
assert storage.list_calibrations("original-3x3") == [broken] and storage.list_disabled("original-3x3") == [off]
page = cw.TestingPage("original-3x3")
texts = lambda lst: [lst.item(i).text() for i in range(lst.count())]
files = lambda lst: [lst.item(i).data(0x0100) for i in range(lst.count())]
assert page.lists_title.text() == "Calibrations"                 # every type, each under its name
assert texts(page.active_list)[0] == "Original 5×5" and files(page.active_list) == [None, str(other)]
assert texts(page.disabled_list)[0] == "Original 3×3" and files(page.disabled_list) == [None, str(off)]
assert texts(page.inactive_list)[0] == "Original 3×3" and files(page.inactive_list) == [None, str(broken)]
retrained = []
page.data_changed.connect(lambda: retrained.append(True))
page.set_enabled([off], True)                                   # Enable: back in the method's folder, the model retrains
assert storage.list_calibrations("original-3x3") == [path, broken] and storage.list_disabled("original-3x3") == [] and retrained
assert texts(page.active_list)[::2] == ["Original 3×3", "Original 5×5"] and files(page.active_list) == [None, str(path), None, str(other)]
assert texts(page.disabled_list) == ["None"]
chosen = []
page.method_changed.connect(chosen.append)
page.set_method("original-5x5")                                 # the chosen method doesn't filter the lists
assert chosen == ["original-5x5"] and files(page.active_list) == [None, str(path), None, str(other)]
assert storage.remove_samples(path, {0}) == 1 and "1 sample," in storage.calibration_text(path)
assert (storage.method_dir("original-3x3") / "trash" / path.name).exists()
storage.remove_calibration(path)
assert storage.list_calibrations("original-3x3") == [broken] and not path.exists()
# Sweep with its dots reused from an Original 4x4: remembered in the file, shown in words, kept through an edit
assert storage.dots_source(METHOD_BY_ID["sweep"]) is None and storage.dots_source(METHOD_BY_ID["original-4x4"]) is None
four = storage.save_calibration(CalibrationRecording((), (), (sample(*point_grid(4)[0]),)), "original-4x4")
assert storage.dots_source(METHOD_BY_ID["sweep"]) == four and storage.dots_source(METHOD_BY_ID["random-sweep"]) is None
reused = storage.save_calibration(CalibrationRecording((sample(0.05, 0.5),) * 2, (sample(0.5, 0.5),), (sample(0.5, 0.5),) * 2),
                                  "sweep", dots_from=four.name)
assert f"(the dots reused from {four.name})" in storage.calibration_text(reused)
storage.remove_samples(reused, {0})
assert f"(the dots reused from {four.name})" in storage.calibration_text(reused)
reused.unlink()
four.unlink()
try:                                                            # an empty calibration is never written
    storage.save_calibration(CalibrationRecording((), (), ()), "sweep")
    raise AssertionError("an empty calibration was saved")
except ValueError:
    pass

# ---- 3. gaze checks in the method's folder, blink scores, the other branches' folders ----
old_branch = tmp / "calibrations" / "calibration_2026-01-02_000000.json"   # master's file in the shared data/
old_branch.parent.mkdir()
old_branch.write_text("{}")
check = storage.save_gaze_check([(0, 0.05, 0.05, 0.05, 0.05), (1, 0.35, 0.37, 0.35, 0.35)], "original-3x3")
assert check.parent == tmp / "original-3x3" and check.name.startswith("gaze-check_original-3x3_20"), check
assert storage.list_calibrations("original-3x3") == [broken]   # a gaze check beside them is not a calibration
words = storage.gaze_check_lines(check)
assert words[0].startswith("Gaze check of No person's Original 3×3 model,") and "2 samples on 4x4 rings, average error 1.0%" in words[0], words[0]
assert words[2] == "#0: looking at dot (1, 1) of the 4x4 grid; the model said 5% across, 5% down, 0.0% of the screen away", words
assert words[3] == "#1: looking at dot (2, 2) of the 4x4 grid; the model said 35% across, 37% down, 2.0% of the screen away", words
assert storage.list_gaze_checks("original-3x3") == [check] and storage.list_gaze_checks("sweep") == []
older = check.with_name("gaze-check_original-3x3_2026-01-01_000000.csv")      # from before the lit rings
older.write_text("index,x,y\n0,0.3,0.26\n")
assert storage.read_gaze_check(older) == [(0, 0.3, 0.26, 0.35, 0.35)]        # the nearest 4x4 ring as its target
results = {m: r for m, *r in storage.method_results()}                          # the comparison, per type
calibs, checks, latest, mean = results["original-3x3"]
assert checks == 2 and abs(latest - 0.01) < 1e-6 and mean > latest, results["original-3x3"]   # latest = the newest file
assert results["sweep"][1:] == [0, None, None]
older.unlink()
two = storage.save_gaze_check([(0, 0.5, 0.5, 0.5, 0.5)], "original-3x3", calibrations=2)
assert two.name.startswith("gaze-check_original-3x3_2-calibrations_20") and storage.file_date(two).year >= 2026, two.name
assert storage.check_calibrations(two) == 2 and storage.check_calibrations(check) is None
assert "model (trained on the newest 2 calibrations)," in storage.gaze_check_lines(two)[0]
two.unlink()
six = storage.save_gaze_check([(0, 0.5, 0.5, 0.05, 0.05), (1, 0.3, 0.3, 0.23, 0.23)], "sweep", tag="multi")
assert six.name.startswith("gaze-check_sweep_multi_20") and storage.check_calibrations(six) is None, six.name
assert storage.check_grid(storage.read_gaze_check(six)) == 6                   # 0.23 is a 6x6 ring, not a 4x4 one
words = storage.gaze_check_lines(six)
assert "model with every model at once," in words[0] and "2 samples on 6x6 rings" in words[0], words[0]
assert words[3].startswith("#1: looking at dot (2, 2) of the 6x6 grid;"), words[3]
video = storage.save_gaze_check([(0, 0.5, 0.5, 0.5, 0.5)], "sweep", calibrations=1, tag="video-2026-10-18_173000")
assert storage.check_calibrations(video) == 1 and "from the recording of 18 Oct 2026, 17:30," in storage.gaze_check_lines(video)[0]
six.unlink()
video.unlink()
rows = [("open", 0, 0.05, 0.06), ("open", 33, 0.07, 0.05), ("down", 66, 0.2, 0.21), ("closed", 99, 0.8, 0.85),
        ("closed", 132, 0.9, 0.88), ("wink_left", 165, 0.9, 0.1)]
blink = storage.save_blink_scores(rows)
assert blink.parent == tmp / "blink-test" and blink.name.startswith("blink-scores_20") and storage.load_blink_scores(blink) == rows and storage.list_blink_scores() == [blink]
words = storage.blink_score_lines(blink)
assert "Closed eyes score at least 0.8" in words[-1] and "never pass 0.21" in words[-1] and "wink" in words[-1], words[-1]
assert storage.trash_file(blink) == storage.BLINK_DIR / "trash" / blink.name and storage.list_blink_scores() == []

# ---- 4. folders cut out (another person's data kept elsewhere) come back empty ----
import shutil
assert storage.list_calibrations("sweep") == [] and storage.list_disabled("sweep") == []   # listing a missing folder: nothing
missing = {k for k in storage.KINDS if not (tmp / k).is_dir()}          # the folders no test above has made yet
made = storage.ensure_layout()
assert {f.name for f in made} == missing and "sweep" not in missing and storage.ensure_layout() == []
shutil.rmtree(tmp / "points")                                   # one method's folder cut away
assert storage.ensure_layout() == [tmp / "points"]
shutil.rmtree(tmp / "original-4x4")
path = storage.save_calibration(CalibrationRecording((), (), (sample(0.5, 0.5),)), "original-4x4")   # saving never needs the folder first
assert path.parent == tmp / "original-4x4"
assert old_branch.exists() and storage.list_people() == []     # the other branches' folders: untouched, not people

assert storage.person_name("  ../Ann <Lee>. ") == "Ann Lee" and storage.person_name(r"/\:*?") == ""
assert storage.person_name("Sweep") == "" and storage.person_name("calibrations") == ""   # names taken by data folders
assert storage.use_person("Ann Lee") == tmp / "Ann Lee" and storage.list_calibrations("original-4x4") == []
storage.ensure_layout()
ann = storage.save_calibration(CalibrationRecording((), (), (sample(0.5, 0.5),)), "original-4x4")
assert ann.parent == tmp / "Ann Lee" / "original-4x4", ann
assert storage.save_gaze_check([(0, 0.5, 0.5, 0.5, 0.5)], "sweep").parent == tmp / "Ann Lee" / "sweep"
assert storage.method_results(tmp / "Ann Lee")[3][2] == 1 and storage.method_results(tmp)[3][2] == 0   # per person
assert storage.save_blink_scores(rows).parent == tmp / "Ann Lee" / "blink-test"
assert storage.set_calibration_enabled(ann, False).parent == tmp / "Ann Lee" / "original-4x4" / "disabled"
storage.use_person("Bo")
assert storage.list_people() == ["Ann Lee"] and storage.list_disabled("original-4x4") == []    # Bo has no folder until made
storage.ensure_layout()
assert storage.list_people() == ["Ann Lee", "Bo"]
assert storage.use_person("") == tmp and storage.list_calibrations("original-4x4") == [path]    # back to data/ itself

# ---- 5. clear data: one person's, or everyone's, into one dated trash folder ----------
one = CalibrationRecording((), (), (sample(0.5, 0.5),))
storage.save_calibration(one, "original-3x3", tmp / "Ann" / "original-3x3" / "original-3x3_2026-10-18_150000.json")
mine = storage.save_calibration(one, "original-3x3")
assert "Ann" in storage.list_people()
where = storage.clear_data("Ann")
assert where.parent == tmp / "trash" and (where / "Ann" / "original-3x3" / "original-3x3_2026-10-18_150000.json").exists()
assert "Ann" not in storage.list_people() and mine.exists()                     # Ann gone, the other files untouched
where = storage.clear_data(None)
assert not mine.exists() and (where / "data" / "original-3x3" / mine.name).exists()
assert (tmp / "original-3x3").is_dir() and storage.list_calibrations("original-3x3") == []   # the empty folders are back

# ---- 6. several files at once: move to a person, remove, restore ---------------------------
from PySide6.QtWidgets import QMessageBox
cw.QMessageBox.question = lambda *a, **k: QMessageBox.Yes                     # every "are you sure" says yes
a = storage.save_calibration(one, "original-3x3", tmp / "original-3x3" / "original-3x3_2026-10-18_160000.json")
b = storage.save_calibration(one, "original-3x3", tmp / "original-3x3" / "original-3x3_2026-10-18_161000.json")
b = storage.set_calibration_enabled(b, False)
page.refresh()
said = cw.move_all([a, b], "James")                                          # Settings never had a James: a new person
assert said == "Moved 2 files to James." and storage.list_people() == ["James"], said
assert storage.list_calibrations("original-3x3", tmp / "James") == [tmp / "James" / "original-3x3" / a.name]
assert storage.list_disabled("original-3x3", tmp / "James") == [tmp / "James" / "original-3x3" / "disabled" / b.name]
storage.save_calibration(one, "original-3x3", a)                              # the same name again here...
assert "Already there, left in place: " + a.name in cw.move_all([a], "James") and a.exists()   # ...is never overwritten
james = storage.list_calibrations("original-3x3", tmp / "James") + storage.list_disabled("original-3x3", tmp / "James")
page.remove_files(james)                                                      # both at once, into their trash folder
page.refresh()
removed = [Path(page.removed_list.item(i).data(0x0100)) for i in range(page.removed_list.count())
           if page.removed_list.item(i).data(0x0100)]
assert sorted(p.name for p in removed) == sorted([a.name, b.name]) and all(p.parent.name == "trash" for p in removed)
page.restore_files(removed)
assert sorted(p.name for p in storage.list_calibrations("original-3x3", tmp / "James")) == sorted([a.name, b.name])
edited = storage.save_calibration(CalibrationRecording((), (), (sample(0.5, 0.5), sample(0.2, 0.2))), "original-3x3")
storage.remove_samples(edited, {0})                                           # an edit: the original waits in trash/
assert len(storage.load_calibration(edited).dots) == 1 and storage.list_removed("original-3x3") == [edited.parent / "trash" / edited.name]
page.restore_files(storage.list_removed("original-3x3"))                      # the original back, the edit undone
assert len(storage.load_calibration(edited).dots) == 2 and storage.list_removed("original-3x3") == []

# ---- 7. Sweep / Points / Line model: one recording, all three types --------------------------
from eyetracking.core.methods import partners
assert [m.id for m in partners(METHOD_BY_ID["points"])] == ["sweep", "line-model"]
assert [m.id for m in partners(METHOD_BY_ID["random-sweep"])] == ["random-points", "random-line-model"]
assert partners(METHOD_BY_ID["original-4x4"]) == []
pts = storage.save_calibration(one, "points", tmp / "James" / "points" / "points_2026-10-18_171729.json")
line = storage.copy_as(pts, "line-model")
assert line == tmp / "James" / "line-model" / "line-model_2026-10-18_171729.json"
assert storage.load_calibration(line) == storage.load_calibration(pts) and storage.list_calibrations("line-model", tmp / "James") == [line]
assert "(the same recording as points_2026-10-18_171729.json)" in storage.calibration_lines(line)[0][0]
page.copy_files([pts], "line-model")                                          # already there: left alone, said so
assert "Already there: points_2026-10-18_171729.json" in page.status.text()
print("lab data ok")
