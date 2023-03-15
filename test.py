import numpy as np
import nibabel as nib


a = nib.load('Training_brats\\BraTS20_Training_001\\BraTS20_Training_001_flair.nii.gz').get_fdata()
a = np.array(a)
print(a.shape)