import h5py as h5

def save_datafile_G1(cosmo,cls,outpath):
    hf = h5.File(outpath+'training_data.h5', 'w')
    hf.create_dataset('cosmo',  data   = cosmo)
    hf.create_dataset('cl_NG',  data   = cls)
    hf.close()

def save_datafile_G2(cosmo,alpha,beta,cls,outpath):
    hf = h5.File(outpath+'training_data.h5', 'w')
    hf.create_dataset('cosmo',  data   = cosmo)
    hf.create_dataset('alpha',  data   = alpha)
    hf.create_dataset('beta',   data   = beta)
    hf.create_dataset('cl_G',   data   = cls)
    hf.close()

def save_datafile_G3(cosmo,a,b,c,cls,outpath):
    hf = h5.File(outpath+'training_data.h5', 'w')
    hf.create_dataset('cosmo',  data   = cosmo)
    hf.create_dataset('a',      data   = a)
    hf.create_dataset('b',      data   = b)
    hf.create_dataset('c',      data   = c)
    hf.create_dataset('cl_G',   data   = cls)
    hf.close()
