import pyapres
import numpy as np
import pytest
import os

from pyapres.base import ApRESBurst

def test_load_file():

    path = "src/tests/data"
    filenames = [os.path.join(path, f) for f in os.listdir(path) \
        # ... only if the returned file in dir is indeed a file
        # ... not a directory
        if os.path.isfile(os.path.join(path, f)) \
        # ... and if the file is *.dat or *.DAT (case insensitive)
        and os.path.splitext(f)[1].lower() == ".dat"]

    for f in filenames:

        bursts = pyapres.read(f)

def test_load_dir():

    path ="src/tests/data"
    
    bursts = pyapres.read(path)

def test_load_burst():

    path = "src/tests/data"

    bursts = pyapres.read(path)

    assert len(bursts) > 0

    print("Loaded {:d} bursts".format(len(bursts)))

    for burst in bursts:

        assert isinstance(burst, ApRESBurst)

        burst.load_data()

        print(burst.path)
        print(repr(burst.fmcw_parameters))

        assert burst.data.shape[0] == burst.get_n_chirps_in_burst()

def test_range_profile():

    burst = pyapres.read("src\\tests\\data\\rhonegletscher-hf.dat", skip_burst=False)
    rp = pyapres.RangeProfile.calculate_from_chirp([], burst.chirp_voltage, burst.fmcw_parameters)

    import matplotlib.pyplot as plt
    plt.plot(np.abs(rp.transpose()))
    plt.show()
    