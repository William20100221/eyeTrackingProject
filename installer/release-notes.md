An eye-tracking reading assistant: it follows where you look with your webcam, works out which parts
of a text you read and which you skipped, and an AI summarises what you missed and quizzes you on it.

**This is a pre-release.** The installers aren't code-signed yet, so your system will warn you once (see below).

## What's new in <version>

- **A new look.** A sidebar like Claude's: New reading, Calibrate, Tests, Data, your recent readings, and Settings
  at the bottom. Drawn in the App Store's style: neutral greys, bold titles, coloured icon tiles, and your theme
  colour as the one accent (made darker where needed so white text on it stays readable), in light and dark.
- **Three start pages** to choose from in Settings > Appearance > Start page: **Clean** (the default: the input
  box with Go Reading and three test shortcuts), **Hero card** (a big Go Reading card beside your calibration
  and last gaze-check error) and **Dashboard** (your tests as rows with Start buttons, and your readings with
  how much of each you read).
- **Tests and Data** have tabs instead of a second sidebar; the calibrations are one list with Active, Off,
  Not usable and Removed tabs, each with its count.
- New reading brings the start page back; this run's messages stay under **Today** in the sidebar, and the
  sidebar's search finds a past reading by its title.

## Download

| System | File |
|---|---|
| Windows 10 / 11 (64-bit) | `StudyAssistant-Setup-<version>.exe` |
| macOS 12+ on Apple Silicon (M1 and later) | `StudyAssistant-<version>-macos-arm64.dmg` |
| Linux (64-bit, X11 or Wayland) | `StudyAssistant-<version>-linux-x64.tar.gz` |

Intel Macs aren't supported: MediaPipe, the face tracker, has no build for them.

## Install

**Windows:** run the setup file (no administrator password needed). If Windows says "Windows protected
your PC", click **More info**, then **Run anyway**. Start Study Assistant from the Start menu, and allow
the camera if Windows asks (Settings > Privacy & security > Camera > Let desktop apps access your camera).

**macOS:** open the `.dmg` and drag **StudyAssistant** into **Applications**. The first time, right-click
it in Applications and choose **Open**, then **Open** again (macOS blocks unsigned apps on a plain
double-click). Allow the camera when asked.

**Linux:**
```bash
tar -xzf StudyAssistant-<version>-linux-x64.tar.gz
./StudyAssistant/StudyAssistant
```

Upgrading keeps your calibrations, history and AI keys.

## What works where

- **Everywhere:** calibration, the gaze check, the blink test, the sample text with its AI summary and
  quiz, the AI providers (Claude, ChatGPT, Gemini, Ollama, or your own server such as LM Studio).
- **Windows only, for now:** **Go Reading** in other apps (Chrome, Word, PDFs). It reads their text
  through Windows' accessibility interface and Windows' OCR, which macOS and Linux don't have.
- On macOS and Linux, cameras are listed as "Camera 1", "Camera 2" (the systems don't give their names),
  and AI keys are stored as plain text in the app's settings (on Windows they are encrypted for your login).

## Check your download

`SHA256SUMS.txt` holds each file's SHA-256. Compare it with yours:
`certutil -hashfile <file> SHA256` (Windows), `shasum -a 256 <file>` (macOS), `sha256sum <file>` (Linux).
