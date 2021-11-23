import apyres
import numpy as np
import pytest
import os

def test_load_file():

    path = "src/tests/data"
    filenames = [os.path.join(path, f) for f in os.listdir(path) \
        # ... only if the returned file in dir is indeed a file
        # ... not a directory
        if os.path.isfile(os.path.join(path, f)) \
        # ... and if the file is *.dat or *.DAT (case insensitive)
        and os.path.splitext(f)[1].lower() == ".dat"]

    for f in filenames:

        apyres.load(f)