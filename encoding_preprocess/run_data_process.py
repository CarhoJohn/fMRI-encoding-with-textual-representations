from load_nii import load_cifti_nii
from hrf import convolve
from hrf import load_ref_TRs

# data could be downladed from https://openneuro.org/datasets/ds004078/versions/1.0.4
for i in range(1,  61):
    load_cifti_nii('sub-01_task-RDR_run-'+str(i)+'_bold.dtseries.nii', 'story_'+str(i))
    # save as story_10.mat

time_root = './'
hrf = glm.first_level.spm_hrf(0.71, 71)
ref_length = load_ref_TRs()

in_root = './'
out_root = './'
for i in range(1, 13): # 12 subjects
    in_path = in_root + '/'
    out_path = out_root + '/result/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    convolve(in_path, time_root, out_path, hrf, ref_length)
