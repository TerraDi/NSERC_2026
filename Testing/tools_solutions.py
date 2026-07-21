import numpy as np

def square(arr):
    return arr**2

def get_pi():
    return np.pi

def picky(num):
    if not isinstance(num, (np.float64, float, np.int32, int)):
        raise TypeError("Must input a number")

def simulate_foregrounds(freqs, prm_array, model_num):

    dimensionless_freqs = freqs / 217.

    if model_num == 1:
        Afg = prm_array[0]
        alpha = prm_array[1]
        fgs = dimensionless_freqs**alpha
    elif model_num == 2:
        Afg = prm_array[0]
        alpha = prm_array[1]
        beta = prm_array[2]
        exponent = alpha + beta * dimensionless_freqs
        fgs = dimensionless_freqs**exponent
    else:
        raise ValueError("Unknown model number")

    return Afg * fgs

def add_noise(spectrum, frac_noise):
    sigma_n = frac_noise * spectrum / 100.
    num_freqs = spectrum.shape[0]
    n = np.random.normal(scale=sigma_n, size=num_freqs)
    return spectrum + n

def write2disk(freqs, spectrum, fname):
    np.savez(fname, freqs=freqs, spectrum=spectrum)

def normalize(x, fx):
    dx = x[1] - x[0]
    integral = dx * np.sum(fx)
    return fx / integral

def normalize2D(x, y, fxy):
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    integral = dx * dy * np.sum(fxy)
    return fxy / integral
