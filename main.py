"""
Entry point:  python main.py   (installed: StudyAssistant.exe)

Starts the whole app (eyetracking/app.py): floating ball + main window.
Add --mouse to try it without a camera, --self-test to check an install.

An installed app has no console, so an error would vanish: anything that
escapes is written to %LOCALAPPDATA%\\StudyAssistant\\crash.log (from source:
data/../crash.log, i.e. the project folder) as well as printed.
"""
import sys
import traceback
from datetime import datetime


def _log_crash(text: str) -> None:
    from path import DATA_DIR
    try:
        log = DATA_DIR.parent / "crash.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"--- {datetime.now():%Y-%m-%d %H:%M:%S}\n{text}\n")
    except OSError:
        pass
    if sys.stderr is not None:                   # a windowed .exe has no stderr
        print(text, file=sys.stderr)


def _excepthook(kind, value, tb) -> None:
    """Errors inside Qt callbacks (a button, a timer) land here instead of vanishing."""
    _log_crash("".join(traceback.format_exception(kind, value, tb)))


if __name__ == "__main__":
    sys.excepthook = _excepthook
    try:
        from eyetracking.app import main
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        _log_crash(traceback.format_exc())
        sys.exit(1)
