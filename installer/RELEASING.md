# Releasing Study Assistant on GitHub

How to turn the code on `master` into a downloadable `StudyAssistant-Setup-<version>.exe`
on the repository's Releases page. Takes about 15 minutes, most of it the build.

## One-time setup

```bash
uv sync                                                       # the app's packages + PyInstaller (dev)
winget install --id JRSoftware.InnoSetup --scope user         # Inno Setup 6, no admin needed
```

## Each release

**1. Choose the version** and write it in `pyproject.toml` (`version = "0.2.0"`).
Use [semantic versioning](https://semver.org): `0.x.y` while the app is still changing,
the middle number for new features, the last for fixes only. The installer and the tag
both take it from here.

**2. Build and test the installer:**

```bash
uv run python installer/build.py
```

This makes the icon, freezes the app with PyInstaller into `dist/StudyAssistant/`, runs the
frozen app's `--self-test` (the build stops if anything is missing), then compiles
`installer/Output/StudyAssistant-Setup-<version>.exe` with Inno Setup (about 7 minutes in all).
Add `--clean` if a build behaves strangely: it rebuilds PyInstaller's cache from scratch.

**3. Try the installer yourself:** run it, start the app from the Start menu, calibrate,
do a gaze check and a Go Reading. Then install the previous version's setup and this one
over it to check that the upgrade keeps your calibrations.

**4. Commit and tag:**

```bash
git add pyproject.toml uv.lock
git commit -m "Release 0.2.0"
git tag -a v0.2.0 -m "Study Assistant 0.2.0"
git push origin master
git push origin v0.2.0
```

**5. Write down the installer's fingerprint** (users can check their download wasn't changed):

```bash
certutil -hashfile installer/Output/StudyAssistant-Setup-0.2.0.exe SHA256
```

**6. Create the release.** Either way gives the same result.

*On github.com:* the repository → **Releases** → **Draft a new release** → *Choose a tag*:
`v0.2.0` → title `Study Assistant 0.2.0` → paste the notes (template below) → drag
`installer/Output/StudyAssistant-Setup-0.2.0.exe` into *Attach binaries* → **Publish release**.
(Tick *Set as a pre-release* while the app is still in testing.)

*With the GitHub CLI:*

```bash
gh release create v0.2.0 installer/Output/StudyAssistant-Setup-0.2.0.exe --title "Study Assistant 0.2.0" --notes-file release-notes.md
```

(If `gh` fails with a certificate error, the network is intercepting HTTPS; use the website.)

**7. Check:** open the release page in a private window, download the installer, and
compare its SHA-256 with step 5.

## Release notes template

```markdown
## Study Assistant 0.2.0

**Download:** StudyAssistant-Setup-0.2.0.exe below (Windows 10/11, 64-bit, about 150 MB).

### What's new
- …

### Install
1. Download the setup file and run it. No administrator password is needed.
2. Windows may say "Windows protected your PC" (the installer isn't code-signed yet):
   click **More info**, then **Run anyway**.
3. Start Study Assistant from the Start menu. Allow camera access if Windows asks
   (Settings > Privacy & security > Camera > Let desktop apps access your camera).

Upgrading: run the new setup; your calibrations, history and keys are kept.

SHA-256: `<the hash from step 5>`
```

## What not to do

- Don't commit `dist/`, `build/` or `installer/Output/` (they're in `.gitignore`); the
  installer belongs on the Releases page, not in git (GitHub refuses files over 100 MB in
  a repository; release files can be up to 2 GB).
- Don't change the `AppId` in `installer/StudyAssistant.iss`: it is how a new version
  finds and replaces the old one.
- Don't re-use a tag for a different build. If a release is broken, publish the next
  patch version.
