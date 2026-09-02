# Report Structure — Quick Reference

## Short version

1. **Introduction** — what problem, for who, what's in/out of scope
2. **Research** — options considered vs. what you picked, with pros/cons of each
3. **Requirements** — measurable success criteria (e.g. error in pixels)
4. **Design** — architecture diagram + why each design choice, and why not the alternative
5. **Development log** — what you built, problems hit, how you fixed them
6. **Testing** — unit tests, accuracy tests vs. your requirements, edge cases (blinking, glasses, lighting)
7. **Evaluation** — did you meet requirements? limitations? what would you change with hindsight?
8. **References** — docs/papers you used

Rule of thumb for "double-sided": every decision = what you chose + what you rejected + why.

---

## Furthermore (detail on each section)

**1. Introduction**
- Problem: e.g. track gaze using only a webcam, no special hardware
- Why it matters: accessibility, UX research, attention tracking, etc.
- Scope limits: single face, fixed screen distance, not robust to big head turns

**2. Research**
- Landmarks: MediaPipe (built-in iris + blendshapes, real-time, but heavy/black-box) vs. Dlib (lighter, more transparent, no iris tracking)
- Gaze method: calibration regression (little data needed, explainable, you can justify it) vs. deep learning appearance models (possibly more accurate, but needs huge datasets/GPU and you can't fully explain it — risky for an assessment)
- Hardware eye-trackers (Tobii etc.) as the "gold standard" you're approximating, and why you're not using one (cost/access)

**3. Requirements**
- Make them numeric/testable: e.g. "gaze error under X px after <30s calibration"
- These numbers get reused directly in Testing/Evaluation later

**4. Design**
- Diagram: Camera → Landmarker → Feature extraction → Calibration data → Train regression → Live prediction → UI
- Justify feature choice: e.g. iris position normalized relative to eye corners (isolates eye movement from head movement) vs. raw landmark coords (would need far more calibration data to separate head motion from eye motion)
- Show your data models (`FrameResult`/`HeadPose`) as evidence of clean design

**5. Development log**
- Chronological, evidenced with snippets/screenshots
- For each step: what you tried, what broke, what you changed and why
  (e.g. "normalized by face width first → drifted on head turn → switched to eye-corner-relative normalization")

**6. Testing**
- Unit: sane numbers when eye closed / looking far left, etc.
- System: measured error vs. your requirement, ideally across a few people
- Edge cases: blink during prediction, glasses, bad lighting, post-calibration head movement — graceful failure or silent garbage?

**7. Evaluation**
- Met requirements? Show the actual numbers.
- Honest limitations: head-movement sensitivity, per-person recalibration, lighting sensitivity
- With hindsight — regression vs. deep learning? More/fewer calibration points? (this is the highest-value section for "double-sided" marks)
- Future improvements: e.g. temporal smoothing, more calibration points, deep learning if given more data/time

**8. References**
- MediaPipe docs, sklearn docs, any papers/articles read

**Practical tip:** keep a running dated notes file *while* you build, not after — reconstructing "why" afterward is where the double-sided quality usually falls apart under time pressure.
