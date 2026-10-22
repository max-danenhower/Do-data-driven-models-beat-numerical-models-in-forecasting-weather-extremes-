#from scipy.io import netcdf
import numpy as np
import sys
import xarray as xr
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
import linearmodels as lm
import zarr
import functions

def main():
    era5_temp = xr.open_dataset('era5_2m_temp_2020.nc')

    #Cosine weights
    cos_weights=np.cos(np.deg2rad(era5_temp.latitude))
    cos_weights=np.tile(cos_weights/np.mean(cos_weights), len(era5_temp.time)*len(era5_temp.longitude))

    hres_temp = xr.open_dataset('hres_2m_temp_2020.nc')
    hres_truth_temp = hres_temp.isel(prediction_timedelta=0)
    graph_temp = xr.open_dataset('graphcast_2m_temp_2020.nc')
    pangu_temp = xr.open_dataset('pangu_2m_temp_2020.nc')


    dat_temp_1= functions.prep_all(
        era5_temp['2m_temperature'],
        hres_truth_temp['2m_temperature'],
        hres_temp['2m_temperature'],
        pangu_temp['2m_temperature'],
        graph_temp['2m_temperature'],
        days_ahead=1
    )

    
    test = functions.RMSE_all(dat_temp_1, operational_comp=True, quant=0.01, greater=False)
    print(test)

if __name__ == "__main__":
    main()