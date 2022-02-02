import pyapres
import numpy as np
import pytest

def test_data_time_voltage_assign():

    # Define valid input arguments -----------------------------------
    valid_time = np.linspace(0,0.9,10)
    valid_data_single = np.cos(np.linspace(0,0.9,10) * 2*np.pi)
    valid_data_multiple = np.array([
            np.cos(np.linspace(0,0.9,10) * 2*np.pi),
            np.cos(np.linspace(0,0.9,10) * 2*np.pi + np.pi/3),
            np.cos(np.linspace(0,0.9,10) * 2*np.pi + 2*np.pi/3)
        ]
    )

    
    # Check invalid types error --------------------------------------
    #   second argument is None for tests as this code shouldn't be
    #   executed and indicates that we are only testing first argument
    with pytest.raises(ValueError):
        data = pyapres.FMCWData([1,2,3], valid_data_single)

    with pytest.raises(ValueError):
        data = pyapres.FMCWData((0.1,0.2,0.3), valid_data_single)

    with pytest.raises(ValueError):
        data = pyapres.FMCWData("hello world", valid_data_single)
    
    with pytest.raises(ValueError):
        data = pyapres.FMCWData(np.array(['one','two','three']), valid_data_single)

    # Check size mismatch errors -------------------------------------
    #   we should see that only 1xN or Nx1 arrays are accepted for
    #   time.  If nx1 is provided then the result should be transposed
    #   to be a 1xN array.  
    #
    #   Provide valid voltage arguments here to test final time
    #   results.

    with pytest.raises(ValueError):
        data = pyapres.FMCWData(np.zeros((2,3)), valid_data_single)
    
    with pytest.raises(ValueError):
        data = pyapres.FMCWData(np.zeros((2,3,3)), valid_data_single)

    with pytest.raises(ValueError):
        data = pyapres.FMCWData(valid_time, np.zeros((2,2)))

    with pytest.raises(ValueError):
        data = pyapres.FMCWData(valid_time, np.zeros((2,2,3)))
    
    # Check valid approaches actually work
    data = pyapres.FMCWData(valid_time, valid_data_single)
    assert (data.voltage == valid_data_single).all()
    assert (data.time == valid_time).all()

    data = pyapres.FMCWData(valid_time, valid_data_multiple)
    assert (data.voltage == valid_data_multiple).all()
    assert (data.time == valid_time).all()  

    data = pyapres.FMCWData(valid_time.transpose(), valid_data_single)
    assert (data.voltage == valid_data_single).all()
    assert (data.time == valid_time).all()

    data = pyapres.FMCWData(valid_time.transpose(), valid_data_single.transpose()) 
    assert (data.voltage == valid_data_single).all()
    assert (data.time == valid_time).all()

    data = pyapres.FMCWData(valid_time, valid_data_single.transpose())
    assert (data.voltage == valid_data_single).all()
    assert (data.time == valid_time).all()
    
    data = pyapres.FMCWData(valid_time.transpose(), valid_data_multiple)
    assert (data.voltage == valid_data_multiple).all()
    assert (data.time == valid_time).all()

    with pytest.raises(ValueError):
        data = pyapres.FMCWData(valid_time.transpose(), valid_data_multiple.transpose()) 
    
    with pytest.raises(ValueError):
        data = pyapres.FMCWData(valid_time, valid_data_multiple.transpose())
