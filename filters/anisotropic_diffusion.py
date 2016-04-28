import numpy as np

def _exp(image_gradient, scale):
    return np.exp(-(np.absolute(image_gradient)/scale)**2)

def _inv(image_gradient, scale):
    return 1 / (1 + (np.absolute(image_gradient)/scale)**2)

def anisotropic_diffusion(image, num_iters=10, scale=10, step_size=0.2, conduction_function=_inv):
    # 'step_size' is Perona and Malik's lambda parameter; scale is their 'K' parameter.
    # The 'conduction_function' is the function 'g' in the original formulation;
    # if this function simply returns a constant, the result is Gaussian blurring.
    if step_size > 0.25:
        raise ValueError('step_size parameter must be <= 0.25 for numerical stability.')
    image = image.astype(np.float64)
    # simplistic boundary conditions -- no diffusion at the boundary
    central = image[1:-1, 1:-1]
    n = image[:-2, 1:-1]
    s = image[2:, 1:-1]
    e = image[1:-1, :-2]
    w = image[1:-1, 2:]
    directions = [s,e,w]
    for i in xrange(num_iters):
        di = n - central
        accumulator = conduction_function(di, scale)*di
        for direction in directions:
            di = direction - central
            accumulator += conduction_function(di, scale)*di
        accumulator *= step_size
        central += accumulator
    return image
