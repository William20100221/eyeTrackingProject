"""
Build Study Assistant for the system this runs on, from the project root:

    uv run python installer/build.py            the app + its download
    uv run python installer/build.py --app-only  just the app in dist/ (nothing packaged)
    add --clean to rebuild PyInstaller's cache from scratch

Steps: the icon -> PyInstaller (installer/StudyAssistant.spec) -> the frozen app's
--self-test (stops here if anything is missing) -> the download, in installer/Output/:
    Windows  StudyAssistant-Setup-<version>.exe          Inno Setup (installer/StudyAssistant.iss)
    macOS    StudyAssistant-<version>-macos-arm64.dmg     the .app on a disk image (hdiutil)
    Linux    StudyAssistant-<version>-linux-x64.tar.gz    the app folder, start ./StudyAssistant/StudyAssistant
The version comes from pyproject.toml. PyInstaller only builds for the system it runs on, so
the three downloads come from GitHub Actions (.github/workflows/release.yml); this script is
what each of its machines runs. Windows needs Inno Setup 6 (iscc) for the last step:
    winget install --id JRSoftware.InnoSetup --scope user
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "installer" / "Output"
DIST = ROOT / "dist" / "StudyAssistant"
APP = ROOT / "dist" / "StudyAssistant.app"                  # macOS only
EXE = (DIST / "StudyAssistant.exe" if sys.platform == "win32" else
       APP / "Contents" / "MacOS" / "StudyAssistant" if sys.platform == "darwin" else DIST / "StudyAssistant")
sys.path.insert(0, str(ROOT))
from path import _user_data                                   # noqa: E402  where the installed app writes its report


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
    report = _user_data() / "StudyAssistant" / "self_test.txt"
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


def package(v: str) -> Path:
    """The download for this system, in installer/Output/."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        iscc = find_iscc()
        if iscc is None:
            sys.exit("Inno Setup 6 not found: winget install --id JRSoftware.InnoSetup --scope user")
        run(iscc, f"/DAppVersion={v}", "installer/StudyAssistant.iss")
        return OUTPUT / f"StudyAssistant-Setup-{v}.exe"
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
    if sys.platform == "darwin":
        out = OUTPUT / f"StudyAssistant-{v}-macos-{arch}.dmg"
        out.unlink(missing_ok=True)
        # a .dmg keeps the .app's symlinks and permissions (a plain zip of Qt's frameworks breaks them)
        run("hdiutil", "create", "-volname", "Study Assistant", "-srcfolder", str(APP), "-ov", "-format", "UDZO", str(out))
        return out
    out = OUTPUT / f"StudyAssistant-{v}-linux-{arch}.tar.gz"
    with tarfile.open(out, "w:gz") as tar:                     # keeps the executable bit and symlinks
        tar.add(DIST, arcname="StudyAssistant")
    return out


def main() -> None:
    v = version()
    os.environ["APP_VERSION"] = v                              # the .app's version (StudyAssistant.spec)
    run(sys.executable, "installer/make_icon.py")
    clean = ["--clean"] if "--clean" in sys.argv else []     # --clean: rebuild from scratch (slower)
    run(sys.executable, "-m", "PyInstaller", "--noconfirm", *clean,
        "--distpath", "dist", "--workpath", "build", "installer/StudyAssistant.spec")
    print(f"app folder: {DIST} ({folder_mb(DIST):.0f} MB)")
    self_test()
    if "--app-only" in sys.argv:
        return
    out = package(v)
    print(f"download: {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
