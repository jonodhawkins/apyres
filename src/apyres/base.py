import datetime
import errno
import importlib.resources
import json
import logging
import os
import re

import numpy as np

class FMCWParameters:

    def __init__(self, fc=3e8, B=2e8, T=1, fs=40000):
        
        if _is_positive_real_number(B):
            self.B = B
        else:
            raise ValueError("Bandwidth (B) should be positive, real number")

        if _is_positive_real_number(fc):
            self.fc = fc
        else:
            raise ValueError(
                "Centre frequency (fc) should be positive, real number")

        # Can validate here whether B and fc form valid combination
        # i.e. is fc - B/2 < 0
        if fc - B/2 < 0: 
            raise ValueError(
                ("Lower limit of chirp frequency is less than 0 Hz."
                "Check bandwidth and centre frequency."))
            
        if _is_positive_real_number(T):
            self.T = T
        else:
            raise ValueError(
                "Chirp period (T) should be positive, real number")

        if _is_positive_real_number(fs):
            self.fs = fs
        else:   
            raise ValueError(
                "Sampling frequency (fs) should be positive real number")

    def getCentreFrequency(self):
        return self.fc

    def getBandwidth(self):
        return self.B

    def getChirpPeriod(self):
        return self.T

    def getSamplingFrequency(self):
        return self.fs
    
    def getChirpFrequencyLimits(self):
        return (self.fc - self.B/2, self.fc + self.B/2)

def _is_positive_real_number(number, eq_zero=True):
    """validate whether number is real and positive include/excluding zero

    :param number: Number to validate
    :type number: Any
    :param eq_zero: Return True for input equal to zero, defaults to True
    :type eq_zero: bool, optional
    :return: if number is real and positive
    :rtype: bool
    """
    flag = isinstance(number, (float, int))
    if eq_zero:
        flag = flag and number >= 0
    else:
        flag = flag and number > 0
    return flag

class FMCWData:

    def __init__(self, time, voltage):
        
        # FMCWParameters.__init__(self, **kwargs)

        # Default to "numerical" data
        self.source = "NUMERICAL"

        # Check time and voltage are numeric
        if not _is_numeric_numpy_array(time):
            raise ValueError(
                "time should be numpy.ndarray with numeric dtype")

        if not _is_numeric_numpy_array(voltage):
            raise ValueError(
                "voltage should be numpy.ndarray with numeric dtype")
            
        N = -1 # number of time samples
        time_valid = False
        # One dimensional is valid
        if len(time.shape) == 1:
            N = len(time)
            time_valid = True
        # Two dimensional is also valid
        elif len(time.shape) == 2:
            # Check that array is singular dimension
            if 1 in time.shape:
                if time.shape.index(1) == 1:
                    time = time.transpose()
                N = time.shape[1]
                time_valid = True
        
        if not time_valid:
            raise ValueError("time should be a 1xN array")

        M = 1 # number of voltage records
        voltage_valid = False
        # Check if voltage is either unidimensional or two dimensional
        if len(voltage.shape) == 1:
            if len(voltage) == N:
                voltage_valid = True
        elif len(voltage.shape) == 2:
            if 1 in voltage.shape:
                # Check whether we need to transpose to get 1xN matrix
                if voltage.shape.index(1) == 0:
                    voltage = voltage.transpose()
                # Check whether number of samples agree
                if voltage.shape[1] == N:
                    M = voltage.shape[0]
                    voltage_valid = True

            elif voltage.shape[1] == N:
                M = voltage.shape[0]
                voltage_valid = True

        if not voltage_valid:
            raise ValueError(
                ("voltage{:s} should be 2-dimensional and have at least second dimension the"
                 " same as time{:s}").format(
                     str(voltage.shape), str(time.shape)
                )
            )

        self.voltage = voltage
        self.time = time

    def getSubBurst(self, index):
        return self.voltage[index]


def _is_numeric_numpy_array(arr):
    """Check whether array is instance of numpy.ndarray and dtype is
    subclass of numeric.

    :param arr: input array to check type of
    :type arr: any

    :return: true if numeric numpy array, false otherwise
    """
    return (
        isinstance(arr, np.ndarray) \
        and np.issubdtype(arr.dtype, np.number)
    )

class ApRESFMCWData(FMCWData):

    pass

    # def __init__(self, path, f):
    
    #     with open(path, 'rb') as fh:
            
    #         burst_start = self._read_burst_header(fh)

    #     # After reading *.dat file header and data, we need to call the
    #     # super().__init__ method to assign time, voltage and 
    #     # FMCWParamter values  

class ApRESParameters(FMCWParameters):

    HEADER_START_STR = "*** Burst Header ***"
    HEADER_END_STR = "*** End Header ***"

    def __init__(self, params, **kwargs):
        """Create instance of a new ApRESParameters object

        :param params: dictionary of parameter name => value pairs
        :type params: dict
        :raises ValueError: raised if params is not a dict object
        """
        if not isinstance(params, dict):
            raise ValueError("params should be a dictionary");
        
        # Assign 'private' variable to parameters
        self._parameters = params
        # Call super constructor
        super().__init__(**kwargs)

    def __getattr__(self, item):
        if item in self._parameters:
            return self._parameters[item]
        else:
            raise AttributeError("No ApRESParameter named {:s}".format(item))


    @staticmethod 
    def read_header(fh):

        line = fh.readline()

        header_line = 0

        while len(line) > 0 and line.rstrip() != ApRESParameters.HEADER_START_STR.encode('ascii'):
            line = fh.readline()

        if len(line) == 0:
            return None, None, None
        
        # Now we are at the start of the header
        header_index = fh.tell() - len(line)

        logging.info("Loading JSON apres_data_properties.json")
        # If we get to this stage, load the apres_dat_properties file
        # to validate header string.
        apres_dat_properties = json.loads(
            importlib.resources.read_text(
            "apyres.resources", "apres_dat_properties.json")
        )

        # Read next line
        line = fh.readline()

        # Create empty header dictionary
        header_dict = dict()

        while line.rstrip() != ApRESParameters.HEADER_END_STR.encode('ascii'):
            
            # Process line
            header_line += 1

            # Decode linestring
            try:
                
                line_ASCII = line.decode('ascii').rstrip()

                # Parse ini regex
                s_result = re.match(r"(.+)\=(.*)", line_ASCII)
                
                if s_result != None:
                    # Argument name
                    argName = s_result.group(1)
                    argVal = s_result.group(2)

                    if not argName in apres_dat_properties:
                        logging.warning(
                            "Invalid property {:s} on line {:d}.".format(
                                argName, header_line
                            )
                        )
                    else:

                        # Get arg type from dictionary
                        argType = apres_dat_properties[argName]["type"]
                        if argType == "int":
                            argVal = int(argVal)
                        elif argType =="float":
                            argVal = float(argVal)

                        # Parse special value if available 
                        argVal = ApRESParameters.parse_special_parameter(
                            argName, argVal
                        )

                        # Need to take care with attribute names that
                        # have spaces in
                        argNameSanitized = re.sub(r"\s+", "_", argName)

                        # Check disagreement
                        if argNameSanitized != argName:
                            logging.info(
                                "Sanitizing header parameter {:s} to {:s}."
                                .format(argName, argNameSanitized)
                            )

                        header_dict[argNameSanitized] = argVal

            except UnicodeDecodeError as e:
                logging.warning("Ignoring line {:d} of header. "\
                    "Not ASCII format.".format(header_line))

            line = fh.readline()

        # Create special parameter for number of antenna combinations
        nTx = 0  # default to mono mode
        nRx = 0
        if "TxAnt" in header_dict:
            for v in header_dict["TxAnt"]:
                if v == 1:
                    nTx += 1
        if "RxAnt" in header_dict:
            for v in header_dict["RxAnt"]:
                if v == 1:
                    nRx += 1

        # Check values have been assigned - if not default to 1
        if nTx == 0:
            nTx = 1

        if nRx == 0:
            nRx = 1

        # Compute number of antennas
        header_dict["NAntennas"] = nTx * nRx

        # Create ApRESParameters object from header_dict
        return ApRESParameters(header_dict), header_index, header_line

    @staticmethod
    def parse_special_parameter(parameter, value):

        # Parse Time Stamp
        if parameter == "Time Stamp":
            return datetime.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        
        elif parameter == "TxAnt":
            return [int(v) for v in value.split(",")]

        elif parameter == "RxAnt":
            return [int(v) for v in value.split(",")]#

        else:
            return value
        

def load(path=None, *args):
    """Load radar profile from *.dat file or *.dat files in folder.

    :param path: file path to file or directory to load.
    :type path: str or list of str
    :param \*args: if path is None    
    """

    if path == None:
        path = args

    # Check if path argument is valid
    if isinstance(path, (list, tuple)):
        # Iterate over list or tuple
        for c_path in path:
            load(c_path)
    elif isinstance(path, dict):
        # Iterate over dict
        for key in path:
            load(path[key])
    elif not isinstance(path, str):
        raise ValueError("'path' should be of type str")
    
    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(
            errno.ENOENT, 
            os.strerror(errno.ENOENT), 
            path
        )

    # Check whether it is directory or file
    if os.path.isdir(path):
        return _load_dir(path)
    else:
        return _load_file(path)

def _load_dir(path):
    """Return a dictionary of FMCWData objects indexed by filename.

    :param path: [description]
    :type path: [type]
    """
    
    # Create empty dictionary
    fileDict = dict()

    # Load all *.dat or *.DAT files in the path directory
    return [_load_file(os.path.join(path,f)) for f in os.listdir(path) \
        # ... only if the returned file in dir is indeed a file
        # ... not a directory
        if os.path.isfile(os.path.join(path, f)) \
        # ... and if the file is *.dat or *.DAT (case insensitive)
        and os.path.splitext(f)[1].lower() == ".dat"]
    
def _load_file(path):
    """Loads a *.dat file as an .. py:class::apyres.ApRESFMCWData class

    :param path: File path pointing to *.dat class to load
    :type path: str
    :return: ApRESFMCWData object loaded from file path
    :rtype: .. py:class::apyres.ApRESFMCWData
    """
    
    # Have to be careful here because files can contain multiple bursts
    # so we can abstract the read_header and read_body methods to the
    # ApRESFMCWData class but need to handle the file pointer and 
    # traversing the file here
    with open(path, 'rb') as fh:

        # Store header index
        header_index = []
        header_lines = []
        header_objects = []
       
        # Store chirp index
        burst_index = []
        burst_length = []
        burst_n_chirps = []
        
        # Find the next index of a burst header
        hObj, hIndex, hLines = ApRESParameters.read_header(fh)

        while hObj != None:

            header_objects.append(hObj)
            header_index.append(hIndex)
            header_lines.append(hLines)

            # Use the data we have from hObj, we store the file index
            # as we should be at the start of the chirp data
            burst_index.append(fh.tell())

            # Calculate number of bursts
            n_chirps =    hObj.NAntennas \
                        * hObj.NSubBursts \
                        * hObj.nAttenuators

            # Calculate total number of bytes
            n_bytes = 2 * hObj.N_ADC_SAMPLES \
                        * n_chirps

            # Append number of chirps
            burst_n_chirps.append(n_chirps)

            # Append number of bytes
            burst_length.append(n_bytes)

            # Seek ahead number of data
            fh.seek(n_bytes, 1)

            # Find next burst header in file
            hObj, hIndex, hLines = ApRESParameters.read_header(fh)

        print("Loaded {:d} headers and {:d} bursts from file {:s}".format(len(header_objects), sum(burst_n_chirps), path))

        # Iterate over each header_object and create an ApRESFMCWData object
        apres_data_objects = []

        for hObj in header_objects:
            
            dObj = ApRESFMCWData(ApRESParameters)