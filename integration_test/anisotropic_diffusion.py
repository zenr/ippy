import matplotlib.pyplot as plt
import scipy as sp
import scipy.misc
import scipy.ndimage

from ..filters.anisotropic_diffusion import anisotropic_diffusion

#%matplotlib inline
im = sp.misc.imread('Rat_Hippocampal_Neuron_he.png')
plt.imshow(im)
plt.gray()
im_ad = anisotropic_diffusion(im)
plt.figure(figsize=(14,6))
plt.subplot(131)
plt.imshow(im)
plt.subplot(132)
plt.imshow(im_ad)
plt.subplot(133)
plt.imshow(sp.ndimage.gaussian_filter(im, 2.0))
