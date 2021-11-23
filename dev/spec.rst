============================
ApyRES Library Specification
============================

----------------
KWN & CS Scripts
----------------
Processing
    fmcw_phase2range.m
    fmcw_range.m
    fmcw_xcorr.m
    fmcw_burst_mean.m
    fmcw_burst_split_by_att.m
    fmcw_burst_subset.m
    fmcw_meanchirp.m
    fmcw_melt.m
    fmcw_nbursts.m

Plotting
    fmcw_kwnprofile.m
    fmcw_plot.m
    fmcw_plot_vif.m
    subplottight.m

Loading Data
    fmcw_load.m
    fmcw_derive_parameters.m
    fmcw_file_format.m
    fmcw_ParametersRMB1b.m
    fmcw_ParametersRMB2.m
    LoadBurstRMB3.m
    LoadBurstRMB4.m
    LoadBurstRMB5.m
    LongBurstRMB3.m
    LongBurstRMB4.m
    LongBurstRMB5.m

Misc
    fmcw_data_dir.txt
    lastlist.txt

-----------------
Key Functionality
-----------------
From above, key functionality can be split into

1. Loading data
2. Plotting data
3. Processing Data

Additionally, apyres could encompass 

- Synthetic data generation
- SAR processing
- Polarimetric functionality

Hence have key functionality above in base package

--------------------
Functional Structure
--------------------
::

      (CST/pRESim)
    Simulated Output ---+
                        |
                [Convert to *.DAT]
                        |
    Measured Output ----+---> Range Profile ----> [Processor] --> [Product]

Then we need a class to represent :code:`*.DAT`` files, which can be converted to a range profile and then processed.

::

    class apyres.FMCWData
    - time (float 1xN)
    - voltage (float MxN)
    - source (str)
    - FMCW parameters

Which can be extended as either a class for measured data from the ApRES

:: 

    class apyres.ApRESFMCWData
    - [various RMB parameters]

Multiple bursts can be stored in a single file (seems to be a behaviour
specific to unattended mode?)

::

    class apyres.pRESimFMCWData
    - [additional pRESim parameters]



