#%% 
from curses import window
from datetime import datetime
from operator import pos
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
        ds,
        input_core_dims=[['y', 'x']],
        output_core_dims=[['gamma', 'beta']],
        kwargs={'imgsize': imgsize},
        vectorize=True,
    )
    data = data.assign_coords( 
        gamma = predictor.gamma_grid,
        beta = predictor.beta_grid)
    return data

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
nds = xr.open_dataset(fn)

nds = nds.copy()

# ds.countrate.plot()
nds = nds.drop_vars(['counts', 'bias', 'bias_err'])

#hot pixel correction
nds = xr.apply_ufunc(
    find_outlier_pixels,
    nds,
    input_core_dims=[['y', 'x']],  
    output_core_dims=[['y', 'x']],  
    kwargs={'tolerance': 5},
    dask='parallelized',
    vectorize=True,
)
#conver to gamma beta coordinates
nds = convert_dims_xy_gammabeta(nds, predictor) #convert to gamma beta
#%%


#for each window, straighten the image(each data variable in nds) and convert gamma to zenith angle
for w in windows:
    dadict = {}
    for data_var in list(nds.data_vars):
        kda = nds[data_var] #data array
        dim = list(kda.dims)
        dim = np.unique([d for d in list(kda.dims) if d not in ['beta', 'gamma']])
        outds = None
        if len(dim) == 1:
            imglist = []
            for d in kda[dim[0]].data:
                kds = kda.sel({dim[0]: d}).to_dataset(name=data_var)
                k = kds.map(predictor.straighten_image, win_name=w, coord='Slit')
                k = convert_gamma_to_zenithangle(k[data_var])
                imglist.append(k)
                del k
            # outds = xr.concat(imglist, dim= xr.DataArray(np.array(kda[dim[0]].data, dtype= float), dims= dim[0]))
            outds = xr.concat(imglist, dim= kda[dim[0]])
            del imglist
        elif  len(dim) == 0:
            k = kda.map(predictor.straighten_image, win_name=w)
            k = convert_gamma_to_zenithangle(k)
            outds = k
        else: raise ValueError(f'Data has 2 or more extra dimensions other than gamma and beta: {dim}.')
        outds = outds.assign_attrs(kda.attrs)
        dadict[data_var] = outds
    saveds = xr.Dataset(dadict)
    saveds.attrs.update(
        dict(Description=" HMSA-O Straighted Spectra",
            ROI=f'{int(w)/10:.1f} nm',
            DataProcessingLevel='1A',
            FileCreationDate=datetime.now().strftime("%m/%d/%Y, %H:%M:%S EDT"),))
    encoding = {var: {'zlib': True}
                                for var in (*saveds.data_vars.keys(), *saveds.coords.keys())}
    savefn = destdir.joinpath(fn.stem.replace('l0',f'l1_{w}.nc'))
    print('Saving %s...\t' % (savefn), end='')
    saveds.to_netcdf(savefn, encoding=encoding)
    print(f'Done.')
    del dadict
    break
# %%

test = xr.open_dataset(savefn)
# %%
test
# %%
