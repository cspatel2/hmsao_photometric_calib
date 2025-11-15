#%%
import numpy as np
import xarray as xr
from misdesigner import MisInstrumentModel, MisCurveRemover
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
# %%
filepath =  Path('/home/sunip/Codes/charmi/hms-ao/l1a_converter/data/l1a/202503/hmsao_l1a_20250320_5577[0].nc')
profilepath = Path('line_profile_5577.nc')
modelpath = '../hmsa_origin_ship.json'
# %%
# %%
win = filepath.stem.split('_')[-1].strip('[0]')
# %%
# Create model and confirm that the Instrument file provided works
model = MisInstrumentModel.load(modelpath)
predictor = MisCurveRemover(model)  # line straightening
windows = predictor.windows
# %%
lprof = xr.open_dataset(profilepath)
ds = xr.open_dataset(filepath)
ds = ds.sel(za = lprof.za)

# %%
# %%
lprof.line_profile.plot(y = 'za')
plt.axvline(0, color = 'r', ls = '--')
# %%
lprof.line_profile.data = gaussian_filter1d(lprof.line_profile.data, sigma=2, axis=0)
# %%
lprof.line_profile.plot(y = 'za')
plt.axvline(0, color = 'r', ls = '--')

# %%
# zapix = np.arange(ds.za.size)
# wlpix = lprof.line_profile * ds.wavelength.size

# #%%
# # mzapix, mwlpix = np.meshgrid(zapix, wlpix, indexing='ij')
# mlam, mza = np.meshgrid(wlpix, zapix, indexing='ij')
# # %%
# imap  = np.zeros((2,*(mlam.shape)))
# # %%
# imap[0,:,:] = mza
# imap[1,:,:] = mlam
# # %%

# plt.imshow(mza)
# plt.colorbar()
# # %%
# plt.imshow(mlam)
# plt.colorbar()

# %%
# mlam, mza = np.meshgrid(ds.wavelength.data,lprof.line_profile.data, indexing='ij')
mza,mlam = np.meshgrid(lprof.line_profile.data, ds.wavelength.data, indexing='ij')
mza -= ds.za.min().data
mza /= (ds.za.max().data - ds.za.min().data)
# %%
imap = np.zeros((2,*(mlam.shape)))
imap[1,:,:] = mza* len(ds.za.data)
imap[0,:,:] = mlam * len(ds.wavelength.data)
# %%
plt.imshow(imap[0])
plt.colorbar()
# %%
plt.imshow(imap[1])
plt.colorbar()


# %%
