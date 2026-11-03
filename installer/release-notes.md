An eye-tracking reading assistant: it follows where you look with your webcam, works out which parts
of a text you read and which you skipped, and an AI summarises what you missed and quizzes you on it.

**This is a pre-release.** The installers aren't code-signed yet, so your system will warn you once (see below).

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
