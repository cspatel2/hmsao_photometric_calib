#%%
import xarray as xr
import matplotlib.pyplot as plt
#%%
win = '6300'
afn = f'photometric_calib_map_{win}.nc'
ads = xr.open_dataset(afn)
bfn = f'photometric_calib_map_{win}_highlow.nc'
bds = xr.open_dataset(bfn)

#%%
fig, ax = plt.subplots(1,3, figsize=(20,5), sharey=True)
ads.kr.plot(ax=ax[0], cmap='viridis', vmin=0, vmax=2e2, cbar_kwargs={'label': 'K (R/countrate)'})
ax[0].set_title('Height 1')
bds.kr.plot(ax=ax[1], cmap='viridis', vmin=0, vmax=2e2, cbar_kwargs={'label': 'K (R/countrate)'})
ax[1].set_title('Height 2')

(ads.kr - bds.kr).plot(ax=ax[2], cmap='bwr', vmin=-1e2, vmax=1e2, cbar_kwargs={'label': 'Difference'})