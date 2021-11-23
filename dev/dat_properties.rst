==========================================
Properties of .dat ApRES Parameter Listing
==========================================

Format: JSON

Each entry in the parameter list is as follows:

::

    "parameter_name" : {
        "format" : "str",
        "example" : "an example",
        "description" : "parameter_name is an example parameter with a default value of 'an example'",
        "valid_source" : ["measured", "simulated"],
        "format" : "%Y-%m-%d"
    }

Description of Parameter Block Options
--------------------------------------

+----------------+------------------------+-------------------------------------------------------------------+
| Property Value | Type                   | Description                                                       |
+================+========================+===================================================================+
| format         | str                    | "str", "float", "int", "datetime"                                 |
+----------------+------------------------+-------------------------------------------------------------------+
| example        | defined by "format"    | Example value for variable                                        |
+----------------+------------------------+-------------------------------------------------------------------+
| description    | str                    | Text description of Parameter                                     |
+----------------+------------------------+-------------------------------------------------------------------+
| valid_source   | list                   | Can contain "measured" or "simulated" to indicate property origin |
+----------------+------------------------+-------------------------------------------------------------------+
| format         | str                    | i.e. "%Y-%m-%d" for datetime, or text description of format       |
+----------------+------------------------+-------------------------------------------------------------------+



