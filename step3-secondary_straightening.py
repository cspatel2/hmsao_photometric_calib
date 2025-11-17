# perform secondary straightening on a l1a dataset based on a provided line profile
import numpy as np
import xarray as xr
from skimage import transform
from pathlib import Path