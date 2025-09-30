# Rapid_Inner_Core_Growth
How fast can Earth's inner core grow in from supercooled core liquids once nucleated?
Following ideas from https://doi.org/10.1038/s43017-024-00639-6


Fred's code for rapid inner core growth.

test-profiles.pkl gives radial properties of the core from a prior thermal history as an example.

The thermodynamics comes from Walker et al., 2025 (https://doi.org/10.1098/rspa.2024.0505) which uses the model of Komabayashi 2014 (doi:10.1002/2014JB010980).

An example is given as both a python script and notebook ("IC_Growth_Example") and outputs to pickle files (example outputs also given here).



Note: you will likely encounter a warning message similar to "/Users/fwilson/Documents/Data/Fast_Freezing/6-reboot/thermodynamic_model.py:82: RuntimeWarning: invalid value encountered in sqrt". The code should still run and this should only occur during initialisation. This is usually not a problem but will be remedied in the future.
