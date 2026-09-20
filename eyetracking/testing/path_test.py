import os, sys
from pathlib import Path
print("cwd     :", os.getcwd())
print("__file__:", Path(__file__).resolve())
print("__file__:", Path(__file__))
print(len(Path(__file__).resolve().parents))
print(len(Path(__file__).parents))
print(Path(__file__).parents)
print(Path(__file__).parent)