## Eye-Tracking Reading Assistant

Read on your computer and find out what you missed. Using only your webcam, the app tracks which paragraphs you actually read. An AI then summarises the parts you skipped and quizzes you on them.

## Install Requirements

This project is built with Mediapipe, OpenCV, scikit-learn and PySide6 (Qt) which all requires to be installed

## Install Library Commands

### Required

```bash
pip install mediapipe
pip install opencv-contrib-python
pip install PySide6
pip install PySide6 materialyoucolor
pip install scikit-learn
pip install pygrabber
pip install anthropic
```

### Optional

Only needed if you use the Gemini or OpenAI reading assistant:

```bash
pip install google-genai
pip install openai
```

> **Tip:** If `pip` isn't recognized, use `python -m pip install <package>` instead.

## Features

- **Reading tracker:** paragraphs turn green as you read them
- **Catch-up summary:** a short summary of everything you skipped
- **Quiz:** questions about what you missed, with a score and feedback on each answer
- **Hands-free buttons:** look at a button for a moment to click it
- **Floating assistant:** a small ball on the edge of your screen opens the chat panel, wherever you are

## Installation

For Windows 10 and 11 (64-bit). You don't need Python or an administrator password.

1. Download **StudyAssistant-Setup-&lt;version&gt;.exe** from the [Releases page](https://github.com/William20100221/eyeTrackingProject/releases) and run it.
2. If Windows says **"Windows protected your PC"**, click **More info**, then **Run anyway**. This appears because the installer isn't code-signed yet.
3. Start **Study Assistant** from the Start menu. If the camera doesn't turn on, check **Settings > Privacy & security > Camera > Let desktop apps access your camera**.

To upgrade, run the newer setup: your calibrations, chat history and AI keys are kept. To remove the app, use **Settings > Apps**. Your data stays in `%LOCALAPPDATA%\StudyAssistant`, so delete that folder too if you want everything gone.

If the app won't start, run `StudyAssistant.exe --self-test` from its folder (`%LOCALAPPDATA%\Programs\StudyAssistant`) and look at `%LOCALAPPDATA%\StudyAssistant\self_test.txt` and `crash.log`.

To build the installer yourself, see [installer/RELEASING.md](installer/RELEASING.md).

## How to use it

1. **Calibrate:** follow a few dots with your eyes so the app learns how your eyes move.
2. **Read:** open a text and read it as you normally would.
3. **Catch up:** when you're done, get a summary of what you skipped and answer a short quiz.

## Privacy

- Your calibration data stays on your computer.
- No camera images are saved or sent anywhere.
- Only the text you read is sent to the AI.
