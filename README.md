#Photometric Calibration of HiT\&MIS
the source code here can convert HiT\&MIS iamges from detector dependent units (ADU/s) to detector independent units (Photons/s or Rayleigh) given images taken by HiT\&MIS of a calibrated light source (Here, we use Gamma Scientific RS-12D Series Calibration Light Source)

step1_fits2ds.py
step2_l1a_processing.py
step3_secondary_straightening.py
    - to do the straightening, line profiles for each window will be needed.
        - use test_line_profile.py to determine the bounds for the line and the background
        - add those bounds to bounds.csv
        -run generate_line_profile.py to create line profiles for all the windows for which the bounds are provided.

