# compute framewise displacement
import numpy as np
import scipy.io as scio
import nibabel as nib
from nilearn import image, plotting

def load_mni_nii(in_file, name):
    """
    input:
        in_file: nii file in mni space
    output:
        data: 4-d if in_file is a fmri time-series, and the shape is (n_TR, 91, 109, 91)
    """
    img = nib.load(in_file)
    data = img.get_fdata() # data is a numpy array and can be saved as mat directly
    scio.savemat(name+'.mat', {'data':data})
    # nilearn can plot nii in mni space, but not in cifti, or it can but I don't know
    plotting.plot_glass_brain(in_file, colorbar=True, title='fmri')
    return data

def save_mni_nii(in_file):
    """
    To operate data in in_file, such as doing z-score.
    input:
        in_file: nii file in mni space
    output:
        out_file: nii file in mni space
    """
    img = nib.load(in_file)
    data = img.get_fdata()
    data -= 2
    sum_nii = nib.Nifti1Image(data, img.affine, img.header)
    sum_nii.to_filename('changed_'+in_file)

def load_cifti_nii(in_file, name):
    """
    basically same with load_mni_nii, only the loaded data has different shape
    input:
        in_file: nii file in cifti format
    output:
        data: (n_TR, 91282) or reverse, (91282, n_TR), can't remember
    """
    img = nib.load(in_file)
    data = img.get_fdata() # data is a numpy array and can be saved as mat directly
    scio.savemat(name+'.mat', {'data':data})
    return data

def save_cifti_nii(in_file):
    """
    To operate data in in_file, such as doing z-score.
    input:
        in_file: nii file in mni space
    output:
        out_file: nii file in mni space
    """
    img = nib.load(in_file)
    data = img.get_fdata()
    data -= 2
    sum_nii = nib.Cifti2Image(data, img.header)
    sum_nii.to_filename('changed_'+in_file)