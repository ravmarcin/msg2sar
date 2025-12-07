import sys
import os
from os.path import join, abspath, dirname, isdir


SETT_DIR = dirname(abspath(__file__))
PROJ_DIR = dirname(SETT_DIR)
DATA_DIR = join(PROJ_DIR, 'data')
KEYS_DIR = join(PROJ_DIR, '.secrets')
NOTE_DIR = join(PROJ_DIR, 'notebooks')
SCRI_DIR = join(PROJ_DIR, 'scripts')
UTIL_DIR = join(PROJ_DIR, 'utils')
EXTU_DIR = join(UTIL_DIR, 'external')
INTU_DIR = join(UTIL_DIR, 'internal')

def setup() -> None:
    """
    Add project directories to system path
    """
    sys.path.insert(0, SETT_DIR)
    sys.path.insert(0, PROJ_DIR)
    sys.path.insert(0, DATA_DIR)
    sys.path.insert(0, KEYS_DIR)
    sys.path.insert(0, NOTE_DIR)
    sys.path.insert(0, SCRI_DIR)
    sys.path.insert(0, UTIL_DIR)
    sys.path.insert(0, EXTU_DIR)
    sys.path.insert(0, INTU_DIR)

    for f in os.listdir(EXTU_DIR):
        if isdir(f):
            sys.path.insert(0, join(EXTU_DIR, f))


