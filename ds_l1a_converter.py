#%% 
from PIL import Image
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from misdesigner import MisInstrumentModel, MisCurveRemover
from functions import find_outlier_pixels, convert_gamma_to_zenithangle
# %%
# ################# USER INPUTS ############################################
datadir = 'data/l0'
destdir = 'data/l1a'
modelpath = 'hmsa_origin_ship.json'
###########################################################################
#%%
################## FUNCTIONS ############################################

def convert_dims_xy_gammabeta_core(data, imgsize):
    data = Image.fromarray(data)
    data = data.rotate(-.311,resample=Image.Resampling.BILINEAR, fillcolor=np.nan)
    data = data.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    image = Image.new('F', imgsize, color=np.nan)
    image.paste(data, (110, 410))
    data = np.asarray(image).copy()
    return data

def convert_dims_xy_gammabeta(ds: xr.Dataset, predictor):
    imgsize = (len(predictor.beta_grid), len(predictor.gamma_grid))
    data = xr.apply_ufunc(
        convert_dims_xy_gammabeta_core,
        ds['countrate'],
        input_core_dims=[['y', 'x']],
        output_core_dims=[['gamma', 'beta']],
        kwargs={'imgsize': imgsize},
        vectorize=True,
    )
    data = data.assign_coords(beta = predictor.beta_grid, gamma = predictor.gamma_grid)
    return data
    # data = 
    # datads = xr.DataArray(
    #     data,
    #     dims=['gamma', 'beta'],
    #     coords={
    #         'gamma': predictor.gamma_grid,
    #         'beta': predictor.beta_grid
    #     },
    #     attrs={'unit': 'ADU/s'}
    # ) 
    # return datads
################################################################################3
#%%
# Create model and confirm that the Instrument file provided works
model = MisInstrumentModel.load(modelpath)
predictor = MisCurveRemover(model)  # line straightening
windows = predictor.windows
#%%
datadir = Path(datadir)
destdir = Path(destdir)
destdir.mkdir(parents=True, exist_ok=True)
# %%
files = sorted(datadir.glob('*.nc'))
if  len([f.stem for f in files if 'dark' in f.stem]) > 0:
    fn = list(datadir.glob('*dark*.nc'))
    for f in fn: files.remove(f)

#%%
#%%

fn = files[-2]
ds = xr.open_dataset(fn)
# ds.countrate.plot()
ds = ds.drop_vars(['counts', 'bias', 'bias_err'])

#hot pixel correction
data = xr.apply_ufunc(
    find_outlier_pixels,
    ds['countrate'],
    input_core_dims=[['y', 'x']],  
    output_core_dims=[['y', 'x']],  
    kwargs={'tolerance': 5},
    dask='parallelized',
    vectorize=True,
)
ds['countrate']= data
del data

#conver to gamma beta coordinates
ds = convert_dims_xy_gammabeta(ds, predictor) #convert to gamma beta
ds = ds.to_dataset(name='countrate')

#%%
out_countrate = {k: [] for k in windows}
out_noise = {k: [] for k in windows}
# %%
tds = ds.apply(predictor.straighten_image, win_name='5577')

# %%

dim = list(ds.countrate.dims)
dim.remove('beta')
dim.remove('gamma')
for w in windows:
    if len(dim)  == 1:
        dim = dim[0]
        for c in ds[dim].data:
                k = ds.sel({dim: c}).map(predictor.straighten_image, win_name=w)

# %%
