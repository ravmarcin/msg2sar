import sys
from os.path import abspath, dirname


PROJ_PATH = dirname(abspath(abspath("")))
sys.path.insert(0, PROJ_PATH)

from settings.paths import global_setup


def local_setup():
    global_setup()