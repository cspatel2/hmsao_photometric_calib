# %%
from curses import window
from datetime import datetime
from operator import pos
from PIL import Image
from git import List
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
# %%
################## FUNCTIONS ############################################

def convert_dims_xy_gammabeta_core(data, imgsize: tuple[int, int]) -> np.ndarray:
    """ converts an image with dimensions y,x to image with dimensions gamma, beta.
    Args:
        data (np.ndarray): input image with dimensions y,x
        imgsize (tuple[int, int]): size of the output image with dimensions gamma, beta

    Returns:
        np.ndarray: image with dimensions gamma, beta
    """    
    # rotate and flip the image to convert from x-y to gamma-beta
    data = Image.fromarray(data)  #type: ignore
    data = data.rotate(-.311, resample=Image.Resampling.BILINEAR,fillcolor=np.nan)
    data = data.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    image = Image.new('F', imgsize, color=np.nan)
    image.paste(data, (110, 410)) #type: ignore
    data = np.asarray(image).copy()
    return data


def convert_dims_xy_gammabeta(ds: xr.Dataset, predictor: MisCurveRemover) -> xr.Dataset:
    """ convert data variables with dim (...,y,x) to data varirables with dim (...,gamma, beta)

    Args:
        ds (xr.Dataset): input dataset with data variables having dimensions (..., y, x)
        predictor (MisCurveRemover): initialized MisCurveRemover object used for conversion

    Returns:
        xr.Dataset: dataset with data variables having dimensions (..., gamma, beta)
    """    
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
        gamma=predictor.gamma_grid,
        beta=predictor.beta_grid)
    return data

def convert_l0_to_l1a(fn: Path, destdir: Path, predictor: MisCurveRemover, windows: List[str]):
    """convert L0(.nc) dataset to L1A(.nc) dataset.
       L1A includes:
       0. fitting image to model (MisDesigner)
       1. hot pixel correction
       2. segregating images into corresponding windows
       3. Primary line straightening for each window
       4. tranform image to dims (zenith angle, wavelength) for each window

    Args:
        fn (Path): L0 file path (.nc)
        destdir (Path): destination directory for L1A files
        predictor (MisCurveRemover): initialized MisCurveRemover object used for conversion
        windows (List[str]): list of window names to process

    Raises:
        ValueError: If data has 2 or more extra dimensions other than gamma and beta.
    
    Returns:
        None

    """    
    ds = xr.open_dataset(fn)
    nds = ds.copy()
    nds = nds.drop_vars(['counts', 'bias', 'bias_err'])
    # hot pixel correction
    nds = xr.apply_ufunc(
        find_outlier_pixels,
        nds,
        input_core_dims=[['y', 'x']],
        output_core_dims=[['y', 'x']],
        kwargs={'tolerance': 5},
        dask='parallelized',
        vectorize=True,
    )
    # conver to gamma beta coordinates
    nds = convert_dims_xy_gammabeta(nds, predictor)  # convert to gamma beta

    # for each window, straighten the image(each data variable in nds) and convert gamma to zenith angle
    for w in windows:
        dadict = {}
        for data_var in list(nds.data_vars):
            kda = nds[data_var]  # data array
            dim = list(kda.dims)
            dim = np.unique([d for d in list(kda.dims)
                            if d not in ['beta', 'gamma']])
            outds = None
            if len(dim) == 1:
                imglist = []
                for d in kda[dim[0]].data:
                    kds = kda.sel({dim[0]: d}).to_dataset(name=data_var)
                    k = kds.map(predictor.straighten_image,win_name=w, coord='Slit')
                    k = convert_gamma_to_zenithangle(k[data_var])
                    imglist.append(k)
                    del k
                # outds = xr.concat(imglist, dim= xr.DataArray(np.array(kda[dim[0]].data, dtype= float), dims= dim[0]))
                outds = xr.concat(imglist, dim=kda[dim[0]])
                del imglist
            elif len(dim) == 0:
                kds = kda.to_dataset(name=data_var)
                k = kds.map(predictor.straighten_image, win_name=w,coord='Slit')
                k = convert_gamma_to_zenithangle(k[data_var])
                outds = k
            else:
                raise ValueError(
                    f'Data has 2 or more extra dimensions other than gamma and beta: {dim}.')
            if outds is not None:
                outds.attrs.update(kda.attrs) #type: ignore
            dadict[data_var] = outds
        saveds = xr.Dataset(dadict)
        # attributes from input file except creation date
        attrs = {k: val for k, val in ds.attrs.items() if ('date' not in k)}
        # addtional attributes for l1a file
        new_attrs = dict(Description=" HMSA-O Straighted Spectra",
                        ROI=f'{int(w)/10:.1f} nm',
                        DataProcessingLevel='1A',
                        FileCreationDate=datetime.now().strftime("%m/%d/%Y, %H:%M:%S EDT"),)
        #
        saveds_attrs = attrs | new_attrs

        saveds = saveds.assign_attrs(saveds_attrs)
        encoding = {var: {'zlib': True}
                    for var in (*saveds.data_vars.keys(), *saveds.coords.keys())}
        savefn = destdir.joinpath(fn.stem.replace('l0', f'l1_{w}.nc'))
        print('Saving %s...' % (savefn), end='')
        saveds.to_netcdf(savefn, encoding=encoding)
        print(f'Done.')
        del dadict
# %%
if __name__ == '__main__':
    # Create model and confirm that the Instrument file provided works
    model = MisInstrumentModel.load(modelpath)
    predictor = MisCurveRemover(model)  # line straightening
    windows = predictor.windows

    #input and output directories
    datadir = Path(datadir)
    destdir = Path(destdir)
    destdir.mkdir(parents=True, exist_ok=True)
#%%
    # list files to process, remove dark
    files = sorted(datadir.glob('*.nc'))
    if len([f.stem for f in files if 'dark' in f.stem]) > 0:
        fn = list(datadir.glob('*dark*.nc'))
        for f in fn:
            files.remove(f)
    # 
    for f in files:
        print(f'Processing {f.name}...')
        convert_l0_to_l1a(f, destdir, predictor, windows)
    print('All done!')
