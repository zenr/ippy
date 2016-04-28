import numpy as np
import scipy.misc

def shanbhag(imgdata) :
    """Returns a binary segmentation threshold using the Shanbhag algorithm
    given imgdata as a grayscale image of type numpy.ndarray
    """
    # get normalized histogram
    hist, bins = np.histogram(imgdata,range(0,257),density=True)

    # calculate cumulative ditribution function (and inverse)
    P1 = np.cumsum(hist) # cdf
    P2 = 1 - P1          # inverse cdf

    # find first and last non-zero bins    
    f = np.nonzero(P1)
    first_bin = f[0][0]
    last_bin = f[0][-1]
    
    # initialize minimum entropy to +infinity
    min_ent = float("inf")
    
    for i in range(first_bin, last_bin) :
        # calculate background entropy
        ent_back = 0
        term = 0.5 / P1[i]
        for j in range(1, i) :
            ent_back -= hist[j] * np.log(1 - term*P1[j-1])
        ent_back *= term
        
        # calculate foreground entropy
        ent_fore = 0
        term = 0.5 / P2[i]
        for j in range(i+1, 256) :
            ent_fore -= hist[j] * np.log(1 - term*P2[j-1])
        ent_fore *= term
        
        # set threshold to value where difference in entropy is minimal
        tot_ent = abs(ent_back - ent_fore)
        if (tot_ent < min_ent) :
            min_ent = tot_ent
            threshold = i

    return threshold

# test case        
from scipy.misc.pilutil import Image

a = Image.open('Rat_Hippocampal_Neuron.png').convert('L')
adata = scipy.misc.fromimage(a)
outimg = scipy.misc.toimage(adata > shanbhag(adata))
outimg.show()
