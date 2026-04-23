import sys
from pathlib import Path


PROJ_PATH = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJ_PATH))


from setup import global_setup

def _setup():
    global_setup()