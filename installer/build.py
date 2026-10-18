"""
Build the Windows installer, from the project root:

    uv run python installer/build.py            the app + the installer
    uv run python installer/build.py --app-only  just dist/StudyAssistant/ (no Inno Setup needed)
    add --clean to rebuild PyInstaller's cache from scratch

Steps: the icon -> PyInstaller (installer/StudyAssistant.spec) -> the frozen app's
--self-test (stops here if anything is missing) -> Inno Setup
(installer/StudyAssistant.iss) -> installer/Output/StudyAssistant-Setup-<version>.exe.
The version comes from pyproject.toml. Needs Inno Setup 6 (iscc) for the last step:
    winget install --id JRSoftware.InnoSetup --scope user
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "StudyAssistant"
EXE = DIST / "StudyAssistant.exe"


def version() -> str:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def run(*cmd, **kw) -> None:
    print(">", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, **kw)


def folder_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def find_iscc() -> str | None:
    """Inno Setup's compiler: on PATH, or where its installer puts it (per user or for all users)."""
    candidates = [shutil.which("iscc")] + [str(Path(base) / "Inno Setup 6" / "ISCC.exe") for base in (
        os.environ.get("LOCALAPPDATA", "") + r"\Programs", os.environ.get("ProgramFiles(x86)", ""),
        os.environ.get("ProgramFiles", ""))]
    return next((c for c in candidates if c and Path(c).is_file()), None)


def self_test() -> None:
    """Run the frozen app's own check; it writes its report next to the data folder."""
    report = Path(os.environ["LOCALAPPDATA"]) / "StudyAssistant" / "self_test.txt"
    report.unlink(missing_ok=True)
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    start = time.monotonic()
    code = subprocess.run([str(EXE), "--self-test"], env=env, cwd=ROOT).returncode
    print(f"self-test took {time.monotonic() - start:.1f} s, exit code {code}")
    print(report.read_text(encoding="utf-8") if report.exists() else "(no report written)")
    crash = report.with_name("crash.log")
    if crash.exists():
        print("crash.log:\n" + crash.read_text(encoding="utf-8")[-3000:])
    if code != 0:
        sys.exit("The frozen app failed its self-test: see the report above and docs/installer_dev_log.md.")


def main() -> None:
    v = version()
    run(sys.executable, "installer/make_icon.py")
    clean = ["--clean"] if "--clean" in sys.argv else []     # --clean: rebuild from scratch (slower)
    run(sys.executable, "-m", "PyInstaller", "--noconfirm", *clean,
        "--distpath", "dist", "--workpath", "build", "installer/StudyAssistant.spec")
    print(f"app folder: {DIST} ({folder_mb(DIST):.0f} MB)")
    self_test()
    if "--app-only" in sys.argv:
        return
    iscc = find_iscc()
    if iscc is None:
        sys.exit("Inno Setup 6 not found: winget install --id JRSoftware.InnoSetup --scope user")
    run(iscc, f"/DAppVersion={v}", "installer/StudyAssistant.iss")
    setup = ROOT / "installer" / "Output" / f"StudyAssistant-Setup-{v}.exe"
    print(f"installer: {setup} ({setup.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
