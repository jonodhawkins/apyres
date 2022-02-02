import datetime
import errno
import importlib.resources
import json
import logging
import os
import pathlib
import re

import numpy as np

# Set logging level
# logging.basicConfig(level=logging.DEBUG)

PYAPRES_DAT_PROPERTIES = json.loads(
    importlib.resources.read_text(
    "pyapres.resources", "apres_dat_properties.json")
)

class FMCWParameters:

    FREQ_SYS_CLK    = 1e9 # 1 GHz upscaled DDS clock.
    DEFAULT_REG_00  = "00000008"
    DEFAULT_REG_01  = "000C0820"
    DEFAULT_REG_02  = "0D1F41C8"
    DEFAULT_REG_0B  = "6666666633333333"
    DEFAULT_REG_0C  = "000053E3000053E3"
    DEFAULT_REG_0D  = "186A186A"
    DEFAULT_REG_0E  = "08B5000000000000"

    DDS_VALUE_REGEX = "[^0-9A-Fa-f]?([0-9a-fA-F]+)[^0-9a-fA-F]?"

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

    def get_centre_frequency(self):
        return self.fc

    def get_bandwidth(self):
        return self.B

    def get_chirp_period(self):
        return self.T

    def get_sampling_frequency(self):
        return self.fs
    
    def get_chirp_frequency_limits(self):
        return (self.fc - self.B/2, self.fc + self.B/2)

    def get_number_of_samples(self):
        return (self.T * self.fs)

    def __repr__(self):
        return (
            "Centre Frequency   : {:>7.3f} MHz\n" + 
            "Bandwidth          : {:>7.3f} MHz\n" +
            "Chirp Period       : {:>7.3f} s\n" + 
            "Sampling Frequency : {:>7.3f} kHz"
        ).format(
            self.fc / 1e6,
            self.B / 1e6,
            self.T,
            self.fs / 1e3
        )

    @staticmethod
    def from_dat_parameters(params=None):
        
        # Default parameter values
        fmcw_params = FMCWParameters()

        # Set default register values
        reg00 = __class__.DEFAULT_REG_00
        reg01 = __class__.DEFAULT_REG_01
        reg02 = __class__.DEFAULT_REG_02
        reg0B = __class__.DEFAULT_REG_0B
        reg0C = __class__.DEFAULT_REG_0C
        reg0D = __class__.DEFAULT_REG_0D
        reg0E = __class__.DEFAULT_REG_0E

        # Frequency limit
        if params != None and "Reg0B" in params:
            dds_match = re.match(__class__.DDS_VALUE_REGEX, params["Reg0B"])
            if dds_match != None:
                reg0B = dds_match.group(1)
        

        if len(reg0B) == 16:
            f_upper = np.round(int(reg0B[0:8], 16) / (2**32) * __class__.FREQ_SYS_CLK)
            f_lower = np.round(int(reg0B[8:16], 16) / (2**32) * __class__.FREQ_SYS_CLK)
        else:
            raise ValueError("Invalid DDS Reg0B length ({:d}{:s}).".format(len(reg0B), reg0B))

        # Frequency step
        if params != None and "Reg0C" in params:
            dds_match = re.match(__class__.DDS_VALUE_REGEX, params["Reg0C"])
            if dds_match != None:
                reg0C = dds_match.group(1)

        if len(reg0C) == 16:
            df_negative = int(reg0C[0:8], 16) / (2**32) * __class__.FREQ_SYS_CLK
            df_positive = int(reg0C[8:16], 16) / (2**32) * __class__.FREQ_SYS_CLK
        else:
            raise ValueError("Invalid DDS Reg0C length ({:d}:{:s}).".format(len(reg0C), reg0C))

        # Time step
        if params != None and "Reg0D" in params:
            dds_match = re.match(__class__.DDS_VALUE_REGEX, params["Reg0D"])
            if dds_match != None:
                reg0D = dds_match.group(1)
        
        if len(reg0D) == 8:
            dt_negative = int(reg0D[0:4], 16) / __class__.FREQ_SYS_CLK * 4
            dt_positive = int(reg0D[4:8], 16) / __class__.FREQ_SYS_CLK * 4
        else:
            raise ValueError("Invalid DDS Reg0D length ({:d}:{:s}).".format(len(reg0D), reg0D))

        # Calculate center frequency
        fmcw_params.fc = np.round((f_upper + f_lower) / 2)
    
        # Calculate bandwidth
        fmcw_params.B = np.round(f_upper - f_lower)

        # Calculate period
        fmcw_params.T = fmcw_params.B / df_positive * dt_positive

        if params != None:
            if "SamplingFreqMode" in params \
                and params["SamplingFreqMode"] == 1:
                # Assign higher sampling frequency
                fmcw_params.fs = 80e3 # 40 kHz
            else:
                fmcw_params.fs = 40e3 # 80 kHz

        return fmcw_params

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

    def __init__(self, chirp_voltage=None, fmcw_parameters=None, **kwargs):

        # If None, assign otherwise use setter function
        if chirp_voltage == None:
            self.chirp_voltage = None
        else:
            self.set_chirp_voltage(chirp_voltage, **kwargs)
        
        if fmcw_parameters == None:
            # Create default FMCW parameters instance
            self.fmcw_parameters = FMCWParameters()
        elif isinstance(fmcw_parameters, FMCWParameters):
            # Store FMCW parameters as provided
            self.fmcw_parameters = fmcw_parameters
        else:
            # error
            raise ValueError("fmcw_parameters should be of type FMCWParameters.")
            
        # Assign source
        if "source" in kwargs:
            self.source = kwargs["source"]
        else:
            self.source = "NUMERICAL"

    def chirp_data(self, **kwargs):
        return self.chirp_voltage

    def chirp_time(self):
        # Chirp period derived from self.fmcw_parameters.T
        # Chirp sampling period from 1/self.fmcw_parameters.fs
        #
        # Hence create array from 0 to fmcw_parameters.T-1/fs in steps of
        # the sampling period 1/fs
        return np.arange(0, self.fmcw_parameters.T, 1/self.fmcw_parameters.fs)

    def set_chirp_voltage(self, chirp_voltage, **kwargs):

        if not _is_numeric_numpy_array(chirp_voltage):
            raise ValueError(
                "chirp_voltage should be numpy.ndarray with numeric dtype")

        if np.shape(chirp_voltage)[1] != \
            self.fmcw_parameters.get_number_of_samples():
            # Array is incorrect size
            raise ValueError(
                "chirp_voltage should be Nx{:d} sized array.".format(
                    self.fmcw_parameters.get_number_of_samples()
                )
            )

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

class ApRESBurst(FMCWData):

    HEADER_START_STR = "*** Burst Header ***"
    HEADER_END_STR = "*** End Header ***"
    ADC_VOLTAGE = 2.5

    def __init__(self, path, params=None, header_index=0, burst_index=-1, **kwargs):
        """Create instance of a new ApRESParameters object

        :param path: path to file containing burst
        :type path: str or pathlib.Path
        :param params: dictionary of parameter name => value pairs
        :type params: dict
        :param header_index: byte offset for start of header in file
        :type header_index: int
        :param burst_index: byte offset for start of burst data in file.
                            default of -1 indicates that header must be reparsed
        :type burst_index: int
        :raises ValueError: raised if params is not a dict object
        """

        # Call super constructor
        super().__init__(source="DAT", **kwargs)

        if isinstance(path, str):
            # Convert to pathlib.Path object
            path = pathlib.Path(path)
        elif not isinstance(path, pathlib.Path):
            # Otherwise, is not already a pathlib.Path object - error
            raise ValueError(
                "path should be a str path or pathlib.Path object")

        if not params != None and isinstance(params, dict):
            raise ValueError("params should be a dictionary");

        # ------------------ END input arg validation -----------------

        # Path to file containing data
        self.path = path

        # Assign 'private' variable to parameters
        self._parameters = params

        # Set empty header and burst indexes
        self.header_index = header_index
        self.header_lines = -1
        self.burst_index = burst_index
        self.chirp_voltage = None

        self.load_fmcw_parameters()

    def __getattr__(self, item):

        # Check that we have actually loaded the parameters
        if self.header_lines < 1:
            raise Exception("Header not loaded from file")

        if item in self._parameters:
            return self._parameters[item]
        else:
            raise AttributeError("No parameter named {:s}".format(item))

    def load(self, fh=None):
        self.load_header(fh)
        self.load_data(fh)

    def load_header(self, fh=None):

        if fh == None:
            with open(str(self.path), 'rb') as fh:
                self.load_header(fh)
        else:
            
            # Seek header start
            fh.seek(self.header_index)

            # Read line and reset header count 
            line = fh.readline()
            header_line = 0

            while len(line) > 0 and line.rstrip() != ApRESBurst.HEADER_START_STR.encode('ascii'):
                line = fh.readline()

            if len(line) == 0:
                logging.error("No header found after {:d} lines in file {:s}.".format(
                    header_line, str(self.path)
                ))
                raise NoHeaderFoundError("No header found in file.")
            
            # Now we are at the start of the header
            header_index = fh.tell() - len(line)

            logging.info("Loading JSON apres_data_properties.json")
            # If we get to this stage, load the apres_dat_properties file
            # to validate header string.
            

            # Read next line
            line = fh.readline()

            # Create empty header dictionary
            header_dict = dict()

            while line.rstrip() != ApRESBurst.HEADER_END_STR.encode('ascii'):
                
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

                        if not argName in PYAPRES_DAT_PROPERTIES:
                            logging.warning(
                                "Invalid property {:s} on line {:d}.".format(
                                    argName, header_line
                                )
                            )
                        else:

                            # Get arg type from dictionary
                            argType = PYAPRES_DAT_PROPERTIES[argName]["type"]
                            if argType == "int":
                                argVal = int(argVal)
                            elif argType =="float":
                                argVal = float(argVal)

                            # Parse special value if available 
                            argVal = ApRESBurst.parse_special_parameter(
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

            if not "NAntennas" in header_dict:
                # Compute number of antennas
                header_dict["NAntennas"] = nTx * nRx

            # Create ApRESParameters object from header_dict
            self.burst_index = fh.tell()
            self.header_lines = header_line
            self._parameters = header_dict
            
            self.load_fmcw_parameters()

    def load_data(self, fh=None):

        logging.info("Seeking burst at [{:d}] in {:s}".format(
            self.burst_index, str(self.path)
        ))

        raw_burst = np.fromfile(
            self.path, 
            dtype=np.uint16, 
            count=int(self.get_n_bytes_in_burst()/2),
            offset=int(self.burst_index)
        ) / 2**16 * __class__.ADC_VOLTAGE

        # Reshape array into each chirp
        self.chirp_voltage = np.reshape(
            raw_burst,
            [self.get_n_chirps_in_burst(), self.N_ADC_SAMPLES]
        )

        logging.info("Loaded burst")

    def load_fmcw_parameters(self):

        self.fmcw_parameters = FMCWParameters.from_dat_parameters(self._parameters)

    def number_of_tx(self):
        return ApRESBurst._get_number_of_antennas(self.TxAnt)

    def number_of_rx(self):
        return ApRESBurst._get_number_of_antennas(self.RxAnt)

    @staticmethod
    def _get_number_of_antennas(arr):
        """Returns the number of elements in an array equal to 1

        :param arr: array to check number of elements in
        :type arr: iterable
        :return: number of elements in arr equal to 1
        :rtype: int
        """
        return sum([int(v) == 1 for v in arr])

    def get_n_chirps_in_burst(self):

        n_chirps = self.number_of_tx() \
                 * self.number_of_rx() \
                 * self.NSubBursts \
                 * self.nAttenuators

        return n_chirps

    def get_n_bytes_in_burst(self):

        # Calculate total number of bytes
        n_bytes = 2 * self.N_ADC_SAMPLES \
                    * self.get_n_chirps_in_burst()

        return n_bytes

    @staticmethod 
    def read_header(path, fh=None, fp=-1):
        # Convert path from str or as an already existing pathlib.Path

        if fh == None:
            # Open new file handle
            with open(str(path), 'rb') as fh:
                return ApRESBurst.read_header(path, fh=fh, fp=fp)

        # Otherwise we have a file handler provided so continue with that
        else:

            # If file pointer is >= 0 then seek to that file point
            if fp >= 0:
                fh.seek(fp)
                
            # Convert path to pathlib.Path
            path = pathlib.Path(path)
            return ApRESBurst._read_header(path, fh)
            
    @staticmethod
    def _read_header(path, fh):

        # Create null ApRESBurst object
        burst_obj = ApRESBurst(
            path,
            params=None,
            header_index=fh.tell(),
            burst_index=-1
        )

        try:
            burst_obj.load_header()
            return burst_obj
        except NoHeaderFoundError:
            return None

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

class NoHeaderFoundError(Exception):
    pass
        
def read(path=None, skip_burst=True, *args):
    """Read bursts headers from *.dat file or all *.dat files in folder

    :param path: file path to file or directory to load.
    :type path: str or list of str
    :param \*args: if path is None    
    """

    if path == None:
        path = args

    # Check if path argument is valid
    if isinstance(path, (list, tuple)):
        # Iterate over list or tuple
        bursts = []
        for c_path in path:
            bursts.append(read(c_path, skip_burst, *args))
        return bursts
        
    elif isinstance(path, dict):
        # Iterate over dict
        bursts = []
        for key in path:
            read(path[key], skip_burst, *args)
        return bursts
        
    elif not isinstance(path, (str, pathlib.Path)):
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
        return _read_dir(path, skip_burst)
    else:
        return _read_file(path, skip_burst)

def _read_dir(path, skip_burst):
    """

    :param path: [description]
    :type path: [type]
    """
    
    # Create empty dictionary
    fileDict = dict()

    # Load all *.dat or *.DAT files in the path directory
    bursts = [_read_file(os.path.join(path,f), skip_burst) \
        for f in os.listdir(path) \
        # ... only if the returned file in dir is indeed a file
        # ... not a directory
        if os.path.isfile(os.path.join(path, f)) \
        # ... and if the file is *.dat or *.DAT (case insensitive)
        and os.path.splitext(f)[1].lower() == ".dat"]

    return [burst for sub_bursts in bursts for burst in sub_bursts]
    
def _read_file(path, skip_burst):
    """Reads a *.dat file to find bursts within file

    :param path: File path pointing to *.dat class to load
    :type path: str
    :return: instance or tuple of py:class::pyapres.ApRESBurst
    :rtype: .. py:class::pyapres.ApRESBurst
    """
    
    # Have to be careful here because files can contain multiple bursts
    # so we can abstract the read_header and read_body methods to the
    # ApRESFMCWData class but need to handle the file pointer and 
    # traversing the file here
    with open(str(path), 'rb') as fh:

        bursts = []
        
        # Find the next index of a burst header
        hObj = ApRESBurst.read_header(path, fh)

        while hObj != None:

            bursts.append(hObj)

            next_index = fh.tell() + hObj.get_n_bytes_in_burst();

            if not skip_burst:
                hObj.load_data(fh)
                
            fh.seek(next_index)
            

            logging.info("Read header #{:d} from {:s} with {:d} lines.".format(
                len(bursts), str(path), hObj.header_lines 
            ))

            # Read next header if available
            hObj = ApRESBurst.read_header(path, fh)

    if len(bursts) == 1:
        bursts = bursts[0]

    return bursts

class RangeProfile:

    def __init__(self, 
        chirp_time=None,
        chirp_voltage=None,
        fmcw_parameters=None,
        data_obj=None, 
        mean=None
    ):
        pass

    @staticmethod
    def calculate_from_chirp(chirp_time, chirp_voltage, fmcw_parameters, pad_factor=2):

        if len(chirp_voltage.shape) == 1:
            chirp_voltage = chirp_voltage.reshape((1,chirp_voltage.shape[0]))

        Nchirps  = chirp_voltage.shape[0]
        Nsamples = chirp_voltage.shape[1]

        # Substract mean
        pad_voltage = chirp_voltage - np.mean(chirp_voltage, axis=1, keepdims=1)

        # Perform blackman window
        window = np.blackman(Nsamples)
        pad_voltage = window * pad_voltage

        # Perform Pad
        pad_voltage = np.concatenate(
            (pad_voltage,
             np.zeros((Nchirps, (pad_factor-1)*Nsamples))
            ),
            axis=1
        )   

        # Perform Shift
        pad_voltage = np.roll(pad_voltage, shift=int(-np.floor(Nsamples/2)), axis=1)

        # Calculate Range Profile
        range_vector = np.fft.fft(pad_voltage, axis=1)
        
        # Spectral Normalisation
        range_vector = range_vector * np.sqrt(2*pad_factor) \
                     / (pad_factor * Nsamples) \
                     / np.sqrt(np.mean(np.power(window,2)))

        # Multiplication by range vector
        n = np.arange(0,Nsamples)
        B = fmcw_parameters.B
        wc = 2*np.pi*fmcw_parameters.fc
        K = 2*np.pi*fmcw_parameters.B / fmcw_parameters.T
        ref_vector = np.exp(-1j*(n * wc / (B*pad_factor) - np.power(n, 2) * K / (B*B*pad_factor*pad_factor)))

        return range_vector[:, 0:Nsamples] * ref_vector
