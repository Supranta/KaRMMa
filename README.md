# KaRMMa
KaRMMa - Kappa Reconstruction for Mass Mapping

KaRMMa is a library for curved-sky mass map reconstruction using a lognormal prior. For more information, see our [paper](https://arxiv.org/abs/2105.14699).

## Producing Bayesian mass maps with DES-Y3 weak lensing data

You can use this repository to run KaRMMa on DES-Y3 weak lensing data. The DES-Y3 data used to create KaRMMa mass maps are included in this repository [here](https://github.com/Supranta/karmma/tree/master/data/des_y3). 

To run KaRMMa you need to run it as, 

```
CONFIGFILE=./config/desy3/desy3.yaml
python3 run_karmma.py $CONFIGFILE
```

Note that each KaRMMa run takes O(1 day) to run. So make sure you have the computational resources to run the code. 
We also include the conda environment used for running KaRMMa as a yaml file [here](https://github.com/Supranta/karmma/blob/master/environment.yml).

#### Understanding the config file

The config file contains three parts: 1) `analysis` 2) `io` 3) `mcmc` 

The description of each of the fields is given below. 

```
analysis:
    nbins: 4                                              # Number of tomographic bins
    nside: 256                                            # Healpix nside resolution
    sigma_e: 0.261                                        # Magnitude of shape noise
    shift: 0.00416708,0.0085054,0.01561475,0.0236787      # Lognormal shift parameters in each bin
    vargauss: 0.18815055,0.14421919,0.1114475,0.06784224  # Lognormal variance in each bin
    cl_file: data/des_y3/cl.npy                           # Theory C(l) used in the KaRMMa run
    pixwin: data/healpix/hybrid_pixwin_256.npy            # HEALPIX window function
```

```
io:     
    datafile: data/des_y3/desy3_shear_data.h5             # Path to the datafile
    io_dir: output/des_y3/                                # Output directory
    maskfile: data/des_y3/mask_desy3.fits                 # Mask file
```

```
mcmc:
    n_burn_in: 150                                        # Number of Burn-in steps           
    n_samples: 100                                        # Number of MCMC samples
```
