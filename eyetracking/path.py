from pathlib import Path
# Get the directory of the current script
script_dir = Path(__file__).parent
# Construct the relative path
relative_path = script_dir / 'models/face_landmarker.task'
print(relative_path, relative_path.exists())
# Open the file
# with relative_path.open('r') as file:
#    content = file.read()
#    print(content)
Path(__file__).resolve()