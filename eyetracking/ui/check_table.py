"""
Data > Gaze check table: every model's gaze-check errors as a table, like a spreadsheet, for one
tester or for everyone, and saved as a .csv (it opens in Excel).

  One tester   a row per model (the 24, then the old types), as the Gaze check page's comparison:
               what it trains on, how many checks, the latest, mean and best error, and how far
               its mean is from the Original 5x5 baseline (the reference every model is judged by).
  Everyone     a row per model, a column per tester (their mean error), then the mean over the
               testers, the spread between them and how many testers it is the best model for.
               Rows sorted by that mean, the best first. Each tester's column is coloured from
               their own best (green) to their own worst (red): testers differ far more from each
               other than models do, so one scale for the whole table would colour by tester; a
               scale per column shows which models win FOR EACH PERSON, the question the lab asks.
Click a heading to sort by it (numbers sort as numbers; an empty cell last).

    table_rows(person) / group_rows()   the numbers, no Qt (tested on their own)
    csv_text(header, rows)              a table as csv text
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QHeaderView, QMenu, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from eyetracking.core.methods import DEFAULT_MODEL, model_name
import eyetracking.services.storage as storage

EVERYONE = None                       # the person picker's "Everyone"
BASELINE = DEFAULT_MODEL              # Original 5x5: what "vs baseline" is measured from

ONE_HEAD = ("Model", "Calibrations", "Gaze checks", "Latest error", "Mean error", "Best error", "vs Original 5×5")
GOOD, BAD = QColor(56, 142, 60), QColor(211, 47, 47)   # the heat scale's ends: lowest error .. highest


def _mean(errors: list[float]) -> float | None:
    return sum(errors) / len(errors) if errors else None


def table_rows(person: str) -> list[tuple]:
    """One tester: (model id, calibrations, checks, latest, mean, best, mean - baseline mean), the best mean first,
    the models without a check last (in the app's order)."""
    results = storage.model_results(storage.person_root(person))
    base = next((_mean(e) for m, _, e in results if m == BASELINE), None)
    rows = []
    for m, used, errors in results:
        mean = _mean(errors)
        rows.append((m, used, len(errors), errors[0] if errors else None, mean, min(errors, default=None),
                     None if mean is None or base is None or m == BASELINE else mean - base))
    return sorted(rows, key=lambda r: (r[4] is None, r[4] if r[4] is not None else 0))


def group_rows() -> tuple[list[str], list[tuple]]:
    """Everyone: (testers, rows). A row: (model id, a mean error per tester (None: no check), the mean over
    the testers who checked it, the spread (worst tester - best), how many testers it is the best model
    for). Only models someone has checked; the lowest mean first."""
    people = [(name or "No person", storage.person_root(name)) for name, _ in storage.person_roots()]
    per = {}                                          # model id -> {tester: mean error}
    for name, root in people:
        for m, _, errors in storage.model_results(root):
            if errors:
                per.setdefault(m, {})[name] = _mean(errors)
    testers = [name for name, _ in people if any(name in d for d in per.values())]
    best_of = {t: min((m for m in per if t in per[m]), key=lambda m: per[m][t]) for t in testers}
    rows = []
    for m, d in per.items():
        vals = [d.get(t) for t in testers]
        have = [x for x in vals if x is not None]
        rows.append((m, vals, _mean(have), max(have) - min(have) if len(have) > 1 else None,
                     sum(best_of[t] == m for t in testers)))
    return testers, sorted(rows, key=lambda r: r[2])


def csv_text(header, rows) -> str:
    """A table as csv: errors as fractions of the screen (0.123), so a spreadsheet can compute with them."""
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(header)
    w.writerows(["" if v is None else round(v, 4) if isinstance(v, float) else v for v in row] for row in rows)
    return out.getvalue()


def one_csv(person: str) -> tuple[list, list]:
    rows = [(model_name(m), used, n, last, mean, best, vs) for m, used, n, last, mean, best, vs in table_rows(person)]
    return list(ONE_HEAD), rows


def group_csv() -> tuple[list, list]:
    testers, rows = group_rows()
    return (["Model", *testers, "Mean over testers", "Spread", "Best model for"],
            [(model_name(m), *vals, mean, spread, wins) for m, vals, mean, spread, wins in rows])


def everyone_long_csv() -> tuple[list, list]:
    """Every tester's own table one under the other, a column naming the tester: the shape a pivot table
    or a statistics package wants (one row per tester and model)."""
    rows = []
    for name, _ in storage.person_roots():
        _, mine = one_csv(name)
        rows += [(name or "No person", *r) for r in mine if r[2]]        # models with a gaze check
    return ["Tester", *ONE_HEAD], rows


class _Num(QTableWidgetItem):
    """A cell that sorts by its number, shown as a percentage (or as `text`). An empty cell (None) comes
    after every number whichever way the column is sorted: nothing measured is never the best or the worst."""

    def __init__(self, value: float | None, signed: bool = False, text: str | None = None):
        super().__init__(text if text is not None else "" if value is None else f"{value:+.1%}" if signed
                         else f"{value:.1%}")
        self.value = value
        self.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

    def __lt__(self, other):
        a, b = self.value, getattr(other, "value", None)
        if (a is None) != (b is None):                       # Qt sorts down by asking b < a: flip for "empty last"
            table = self.tableWidget()
            down = table is not None and table.horizontalHeader().sortIndicatorOrder() == Qt.DescendingOrder
            return (a is None) == down
        return a is not None and a < b


class _Int(_Num):
    def __init__(self, value: int | None, text: str | None = None):
        super().__init__(value, text=str(value) if text is None else text)


def heat(value: float, lo: float, hi: float, alpha: int = 90) -> QColor:
    """lo (green) .. hi (red), faint enough for the text to read in light and dark themes."""
    t = 0.5 if hi <= lo else (value - lo) / (hi - lo)
    c = QColor.fromRgbF(*(a + (b - a) * t for a, b in zip(GOOD.getRgbF()[:3], BAD.getRgbF()[:3])))
    c.setAlpha(alpha)
    return c


class CheckTablePage(QWidget):
    said = Signal(str)                # a line for the app's status line (a download saved, or why not)

    def __init__(self):
        from eyetracking.ui.control_window import I_CHEVRON, I_DOWNLOAD, I_TABLE, button, label
        super().__init__()
        self.person: str | None = EVERYONE
        v = QVBoxLayout(self)
        v.setContentsMargins(24, 20, 24, 24)
        v.setSpacing(12)
        v.addWidget(label(f"{I_TABLE}   Gaze check table", "sheetTitle"))
        lead = label("Every model's gaze-check errors, as a table: for one tester, or everyone side by side. Click "
                     "a heading to sort. Lower is better (the mean distance from where the model said to the lit "
                     "ring, as a share of the screen).", "lead")
        lead.setWordWrap(True)
        v.addWidget(lead)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(label("Tester", "rowtitle"))
        self.person_btn = button("", "assist")
        menu = QMenu(self.person_btn)
        menu.aboutToShow.connect(lambda: (menu.clear(), menu.addAction("Everyone", lambda: self.show_person(EVERYONE)),
                                          [menu.addAction(n or "No person (data/)", lambda n=n: self.show_person(n))
                                           for n, _ in storage.person_roots()]))
        self.person_btn.setMenu(menu)
        row.addWidget(self.person_btn)
        row.addStretch(1)
        self.save_btn = button(f"{I_DOWNLOAD}  Download", "outlined")
        save_menu = QMenu(self.save_btn)
        save_menu.addAction("This table (.csv)…", lambda: self.save(*self.current_csv(), self.default_name()))
        save_menu.addAction("Every tester, one row per tester and model (.csv)…",
                            lambda: self.save(*everyone_long_csv(), "gaze-checks_every-tester.csv"))
        save_menu.addAction("Everyone side by side (.csv)…", lambda: self.save(*group_csv(), "gaze-checks_everyone.csv"))
        self.save_btn.setMenu(save_menu)
        row.addWidget(self.save_btn)
        v.addLayout(row)
        self.table = QTableWidget(objectName="table")
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)    # Ctrl / Shift-click, Ctrl+C copies nothing extra
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        v.addWidget(self.table, 1)
        self.note = label("", "sheetText")
        self.note.setWordWrap(True)
        v.addWidget(self.note)
        self._chevron = I_CHEVRON

    def show_person(self, person: str | None):
        """Another tester, or Everyone: the columns change, so a sort picked before (by column number) would
        land on another column: back to the table's own order, the lowest mean first."""
        self.person = person
        self.table.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        self.refresh()

    def default_name(self) -> str:
        who = "everyone" if self.person is EVERYONE else (self.person or "no-person").replace(" ", "-")
        return f"gaze-checks_{who}.csv"

    def current_csv(self) -> tuple[list, list]:
        return group_csv() if self.person is EVERYONE else one_csv(self.person)

    def save(self, header, rows, name: str, path: str | None = None) -> Path | None:
        """Ask where (a Save dialog in Downloads), then write the csv; say what happened in the status line.
        Returns the file, or None if cancelled or it couldn't be written (a file open in Excel is locked)."""
        if path is None:
            path, _ = QFileDialog.getSaveFileName(self, "Download the table", str(Path.home() / "Downloads" / name),
                                                  "CSV (*.csv)")
        if not path:
            return None
        try:
            Path(path).write_text(csv_text(header, rows), encoding="utf-8-sig")   # the BOM: Excel reads ×, é right
        except OSError as e:
            self.said.emit(f"Couldn't save {Path(path).name} ({e.strerror or e}). Is it open in Excel? Close it and "
                           "try again, or pick another name.")
            return None
        self.said.emit(f"Saved {len(rows)} rows as {path}.")
        return Path(path)

    def refresh(self):
        self.person_btn.setText(f"{'Everyone' if self.person is EVERYONE else self.person or 'No person (data/)'}"
                                f"  {self._chevron}")
        self.table.setSortingEnabled(False)          # filling a sorted table moves rows under the loop
        self.table.clear()
        if self.person is EVERYONE:
            self._fill_group()
        else:
            self._fill_one()
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)   # the model's name takes the room left

    def _fill_one(self):
        rows = table_rows(self.person)
        self.table.setColumnCount(len(ONE_HEAD))
        self.table.setHorizontalHeaderLabels(ONE_HEAD)
        self.table.setRowCount(len(rows))
        means = [r[4] for r in rows if r[4] is not None]
        for i, (m, used, n, last, mean, best, vs) in enumerate(rows):
            grids = int(used.split()[0]) if used[:1].isdigit() else None          # "2 + 1 of 2": sorts by the 2
            cells = [QTableWidgetItem(model_name(m)), _Int(grids, used), _Int(n), _Num(last), _Num(mean),
                     _Num(best), _Num(vs, signed=True)]
            if mean is not None and len(means) > 1:
                cells[4].setBackground(heat(mean, min(means), max(means)))
            if vs is not None:
                cells[6].setForeground(GOOD if vs < 0 else BAD)
            for c, cell in enumerate(cells):
                self.table.setItem(i, c, cell)
        self.note.setText("vs Original 5×5: the mean error minus Original 5×5's, in points of the screen; green = "
                          "better than the baseline. Calibrations: the grid's, + the special's. Rows: the lowest mean "
                          "first; models with no gaze check yet at the end.")

    def _fill_group(self):
        testers, rows = group_rows()
        head = ["Model", *testers, "Mean over testers", "Spread", "Best model for"]
        self.table.setColumnCount(len(head))
        self.table.setHorizontalHeaderLabels(head)
        self.table.setRowCount(len(rows))
        cols = [[r[1][j] for r in rows if r[1][j] is not None] for j in range(len(testers))]
        for i, (m, vals, mean, spread, wins) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(model_name(m)))
            for j, x in enumerate(vals):
                cell = _Num(x)
                if x is not None and len(cols[j]) > 1:
                    cell.setBackground(heat(x, min(cols[j]), max(cols[j])))
                    if x == min(cols[j]):
                        f = cell.font()
                        f.setBold(True)                      # this tester's best model
                        cell.setFont(f)
                self.table.setItem(i, 1 + j, cell)
            win = _Int(wins)
            for c, cell in enumerate((_Num(mean), _Num(spread), win)):
                self.table.setItem(i, 1 + len(testers) + c, cell)
        self.note.setText("Each tester's column: their mean error per model, coloured from their own best (green, "
                          "bold) to their own worst (red), because testers differ more from each other than models "
                          "do. Spread: worst tester minus best. Best model for: how many testers it wins for. Rows: "
                          "the lowest mean over the testers first." if rows else "No gaze check yet, by anyone.")


if __name__ == "__main__":
    # the numbers on a made-up data folder: two testers, three models
    import os
    import tempfile
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    tmp = Path(tempfile.mkdtemp())
    storage.BASE_DIR = tmp
    storage.use_person("")
    check = lambda person, m, err: storage.save_gaze_check([(0, 0.5 + err, 0.5, 0.5, 0.5)], m,
                                                           storage.person_root(person) / m)
    for person, m, err in [("Ann", "original-5x5", .20), ("Ann", "original-3x3", .30), ("Ann", "sweep-original-4x4", .10),
                           ("Bo", "original-5x5", .40), ("Bo", "original-3x3", .25), ("Bo", "sweep", .5)]:
        check(person, m, err)
    rows = table_rows("Ann")
    assert [r[0] for r in rows[:3]] == ["sweep-original-4x4", "original-5x5", "original-3x3"], rows[:3]
    assert abs(rows[0][6] + 0.10) < 1e-6 and rows[1][6] is None                     # 10 points better than the baseline
    testers, rows = group_rows()
    assert testers == ["Ann", "Bo"] and rows[0][0] == "sweep-original-4x4" and rows[-1][0] == "sweep", rows
    three = next(r for r in rows if r[0] == "original-3x3")
    assert abs(three[2] - 0.275) < 1e-6 and abs(three[3] - 0.05) < 1e-6 and three[4] == 1   # Bo's best
    header, long = everyone_long_csv()
    assert header[0] == "Tester" and {r[0] for r in long} == {"Ann", "Bo"} and len(long) == 6
    text = csv_text(*group_csv())
    assert text.splitlines()[0] == "Model,Ann,Bo,Mean over testers,Spread,Best model for", text
    assert "Sweep (old),,0.5,0.5,,0" in text, text

    # the page: numbers sort as numbers, empty cells last both ways, a new tester starts unsorted, downloads say so
    from PySide6.QtWidgets import QApplication
    from eyetracking.core.models import CalibrationRecording, CalibrationSample
    storage.save_calibration(CalibrationRecording((), (), (CalibrationSample((0.5,) * 12, 0.5, 0.5),)), "original-4x4",
                             storage.person_root("Ann") / "original-4x4" / "original-4x4_2026-10-20_100000.json")
    app = QApplication([])                                              # Ann's Original 4x4: a row with no check yet
    page = CheckTablePage()
    said = []
    page.said.connect(said.append)
    page.show_person("Ann")
    column = lambda c: [page.table.item(r, c).text() for r in range(page.table.rowCount())]
    page.table.sortByColumn(4, Qt.DescendingOrder)                                  # mean error, worst first
    assert column(4)[:3] == ["30.0%", "20.0%", "10.0%"] and set(column(4)[3:]) == {""}, column(4)
    page.table.sortByColumn(4, Qt.AscendingOrder)
    assert column(4)[:3] == ["10.0%", "20.0%", "30.0%"] and set(column(4)[3:]) == {""}, column(4)
    a, b = _Int(10, "10"), _Int(2, "2 + 1")
    assert b < a and not a < b                                                       # 2 before 10, not as text
    page.show_person(EVERYONE)
    assert page.table.horizontalHeader().sortIndicatorSection() == -1 and column(0)[0] == "Sweep + Original 4×4"
    out = page.save(*page.current_csv(), "x.csv", str(tmp / "everyone.csv"))
    assert out.read_text(encoding="utf-8-sig").startswith("Model,Ann,Bo") and said[-1].startswith("Saved 4 rows")
    assert page.save(*page.current_csv(), "x.csv", str(tmp / "no-such-folder" / "x.csv")) is None
    assert said[-1].startswith("Couldn't save x.csv"), said
    print("ok")
