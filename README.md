#  Create Photometric Calibration Maps for HiT&MIS data

The code here creates calibration_maps. The calibration map will be used to convert from detector dependent units (ADU/s) to detector indenpendent units (Photons/s or Rayleigh.)

Requirements:
1. calibration dataset which should include: Background images, Calibration lamp images, Dark images, and Flatfield images. 
2. Model.json - Model config file that describes the instrument design. Required for L1A_converter. This will be an input to MisDesigner. (i.e. hms_origin_ship.json)
3. Line_profiles.nc -  shape of the line that needs to be straightned. Required for L1B_converter for each window. This informs the secondary straightening process.  (i.e. line_profile_{5577}.py)
        - if you need to produce line profiles -> follow steps in hmsao_1lb_converter repo.
4. Calib_params.csv - Parameters about the source and foreoptic to calc throughput. Required for create_photometric_calibmap. (i.e. calib_params_hmsao_slit_100um.csv)
        - if it needs to be created, use create_photometric_calib_map() params as a guide.
5. Calib_curve.nc - calibration curve (Radiance vs wavelength) of the Calibration lamp used. Required for create_phoptometric_caliblamp. (i.e. D300 HL2372 2025-09-11.nc )


run each file in the order of step number to create calibration maps. These are inputs to L1C_converter.

Description:

Step1-fits2ds.py : Converts the raw calibration images to relevent datasets
step2-ds_l1a_converter.py : converts raw.nc to L1A dataset (.nc)
step3-L1b-converter.py : converts L1A files to L1B files
step4-create_photometric_calibmap.py : creates the photometric_calibration maps required to perform a calibration on the full dataset.

