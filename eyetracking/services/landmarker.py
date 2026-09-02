#Mediapipe Wrapper
from pathlib import Path;
p = Path('eyetracking/services/landmarker.py').resolve()
print(p)
[print(i, x) for i, x in enumerate(p.parents)]