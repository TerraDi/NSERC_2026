import numpy as np
import matplotlib.pyplot as plt
import socket
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader, Dataset
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import os

trans_scin_range = (0.01,0.2)
AGN_scin_range = (0.00005,0.2)  ## replicates the observed variability fractions in CHILES VERDES
varbins = np.array([0,0.02,0.1]) ## fractional brightness fluctuation standard deviation. 0.1 Corresponds to a 10% RMS flux density fluctuation
varprobs = np.array([136/185,33/185,16/185]) ## correspond probabilities. Must sum to 1

### LOAD IN THE VARIABILITY: ###

scinlib = np.load('/Users/dtian/Documents/Programming/Di_Research/Di_codefromFir/scintillation_library.npz')['varlib']
#scinlib = np.load('/home/dtian/scratch/Di_Research/Di_codefromFir/scintillation_library.npz')['varlib']
varlib = np.load('/Users/dtian/Documents/Programming/Di_Research/Di_codefromFir/variation_library.npz')['varlib']
#varlib = np.load('/home/dtian/scratch/Di_Research/Di_codefromFir/variation_library.npz')['varlib']

transientsdata = (np.load('/Users/dtian/Documents/Programming/Di_Research/Di_codefromFir/TDE_afterglows/ISM_profile/JettedTDE_ISM1.npz')['data'])[:,:,20]

def make_gaussian_heatmap(binary_target, sigma=2.0):
    """
    binary_target: (H, W) array with a few 1s at feature locations.
    returns:       (H, W) float heatmap, peaks = 1.0 at each feature.
    """
    H, W = binary_target.shape
    heatmap = np.zeros((H, W), dtype=np.float16) #changed from 32 originally

    # locations of the 1s
    ys, xs = np.nonzero(binary_target)

    # precompute kernel radius (3 sigma covers ~99% of the blob)
    radius = int(np.ceil(3 * sigma))

    for cy, cx in zip(ys, xs):
        # bounds of the patch to write into (clipped to image edges)
        y0, y1 = max(0, cy - radius), min(H, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(W, cx + radius + 1)

        yy, xx = np.mgrid[y0:y1, x0:x1]
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))

        # max-combine so overlapping blobs stay <= 1
        heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], g)

    return heatmap

def gaussian2d(x,y,meanx,meany):

    FWHM = 5

    '''
    returns a 2d gaussian with the right shape
    '''

    return np.exp(-0.5*2.3**2*((x-meanx)**2+(y-meany)**2)/FWHM**2) # factor of 2.3 converts the FWHM to a standard deviation

def genbackground(xlen,ylen,tlen,Nbgnd,noiseamp=0.1):

    x = np.arange(xlen) ## x,y,t arrays
    y = np.arange(ylen)
    t = np.arange(tlen) 

    xx,yy,tt = np.meshgrid(x,y,t) ## create a grid of x,y positions to evaluate at

    #np.random.seed(41)
    rng = np.random.default_rng(123)
    
    noisefield = noiseamp*rng.standard_normal(size = (xlen,ylen,tlen)) # generate the noise

    backgroundfield = np.zeros(noisefield.shape) ## array for the transient layer to be created in
    
    x0 = [] ## these record the x, y, and time mean values for each transient signal
    y0 = [] ## these record the x, y, and time mean values for each transient signal
    positions = np.zeros((Nbgnd,2))

    for i in range(Nbgnd): ## for each background source

        x0.append(np.random.randint(xlen)) 
        y0.append(np.random.randint(ylen)) ## generate a uniform random map position

        bgfluxdensity = 10**(2*np.random.rand()-1) ## brightness, right now log-uniform between 0.1 and 10

        layer = bgfluxdensity*gaussian2d(xx,yy,x0[-1],y0[-1]) # add the map layer value

        ## apply intrinsic variability:

        idx = np.random.randint(2000)
        phase = np.random.randint(608)

        randfloat = np.random.rand()
        
        if randfloat <= varprobs[0]:
            amp = varbins[0]
        elif randfloat <= varprobs[0]+varprobs[1]:
            amp = varbins[1]
        else:
            amp = varbins[2]
        
        layer*= 1+amp*(varlib[idx][phase:])[np.newaxis,np.newaxis,:tlen]

        ## apply extrinsic variability:

        idx = np.random.randint(2000)
        phase = np.random.randint(608)
        amp = 10**((np.log10(AGN_scin_range[1]/AGN_scin_range[0]))*np.random.rand()+np.log10(AGN_scin_range[0]))
        #print(amp)

        layer*= 1+amp*(varlib[idx][phase:])[np.newaxis,np.newaxis,:tlen]

        backgroundfield += layer



    positions[:,0] = np.array(x0)
    positions[:,1] = np.array(y0)

    return backgroundfield,noisefield,positions
        
def gentransients(xlen,ylen,tlen,Ntrans):
    '''
    Return  transient_field:    100*100*608 transient field
            positions:          Ntrans * 2 transient positions
            gaussian_transient_map 100*100 2d-gaussian position of the transient
    '''

    global transientsdata
    #np.random.seed(41)
    x = np.arange(xlen) ## x and y arrays
    y = np.arange(ylen)
    t = np.arange(tlen) 

    xx,yy,tt = np.meshgrid(x,y,t) ## create a grid of x,y positions to evaluate at

    transient_field = np.zeros((xlen,ylen,tlen)) ## array for the transient layer to be created in
    
    x0 = [] ## these record the x, y, and time mean values for each transient signal
    y0 = [] ## these record the x, y, and time mean values for each transient signal
    positions = np.zeros((Ntrans,2))
    transient_map = np.zeros((xlen, ylen))

    for i in range(Ntrans):

        x0.append(np.random.randint(xlen))
        y0.append(np.random.randint(ylen))

        while True:
            LCidx = np.random.randint(7000)
            LC = transientsdata[LCidx]
            if np.max(LC) != 0: break #This ensure np.max is no zero and eliminate Nan
        LC /= np.max(LC)
        transfluxdensity = 10**(np.random.rand()-0.5) ## right now log-uniform between 0.3 and 3 ## change as you see fit

        ## apply the shape of the lightcurve
        
        layer = transfluxdensity*gaussian2d(xx,yy,x0[-1],y0[-1])*LC[np.newaxis,np.newaxis,:tlen]

        ## apply extrinsic variability:

        idx = np.random.randint(2000)
        phase = np.random.randint(608)
        amp = 10**((np.log10(trans_scin_range[1]/trans_scin_range[0]))*np.random.rand()+np.log10(trans_scin_range[0]))
        #print(amp)

        layer*= 1+amp*(varlib[idx][phase:])[np.newaxis,np.newaxis,:tlen]

        transient_field += layer

        transient_map[y0[-1]][x0[-1]] = 1
    #gaussian_transient_map = make_gaussian_heatmap(transient_map)
    positions[:,0] = np.array(x0)
    positions[:,1] = np.array(y0)

    return transient_field,positions,transient_map

def genfields(Nfields=2, xlen=100, ylen=100, tlen=608, noiseamp=0.1, transient_toggle = True):
    '''
    Return: combined_field: Nfield * xlen * ylen * tlen, X input for NN
            target_maps:    Nfield * xlen * ylen, y for NN
            bgpositions:    Nfield * Nbgnd * 2, location of all background
            transientpositions: Nfield * Ntrans * 2, location of all transients

    '''
    combined_fields = np.empty((Nfields, xlen, ylen, tlen), dtype=np.float16)
    target_maps     = np.empty((Nfields, xlen, ylen),       dtype=np.float16)
    bgpositions, transientspositions = [], []

    for i in range(Nfields):
        print("Generating Image Index:", i)
        Nbgnd  = np.random.randint(1, 25)
        if(transient_toggle):
            Ntrans = np.random.randint(25)
        else: 
            Ntrans = 0
        bg, noise, bgposition = genbackground(xlen, ylen, tlen, Nbgnd, noiseamp)
        transient, tpos, target_map = gentransients(xlen, ylen, tlen, Ntrans)

        combined_fields[i] = bg + noise + transient   # written in place, float32
        target_maps[i]     = target_map
        bgpositions.append(bgposition)
        transientspositions.append(tpos)
    #bgpositions = np.array(bgpositions)
    #transientspositions = np.array(transientspositions)
    return combined_fields, target_maps, bgpositions, transientspositions

def normalize_np(data, sample_frac=0.05, eps=1e-8):
    """
    Computes mean/std from a random subsample of `data` (cheap, statistically
    sufficient at this dataset size), then normalizes `data` in-place.
    Runs entirely on the original contiguous numpy array -- no torch,
    no permute -- so there's no ambiguity about whether a reduction over
    a huge strided view silently materializes a full-size temporary.
    """
    n = data.shape[0]
    sample_size = max(1, int(n * sample_frac))
    idx = np.random.choice(n, size=sample_size, replace=False)
    sample = data[idx].astype(np.float32)   # small, bounded copy just for stats
    mean = sample.mean()
    std  = sample.std()
    del sample

    data -= mean            # in-place on the big contiguous array, no new allocation
    data /= (std + eps)     # in-place, no new allocation
    return data, mean, std

def to_torch_nchw(data):
    """Convert an already-normalized NHWC numpy array into an NCHW torch view."""
    data = torch.from_numpy(data)   # no copy, shares the (already normalized) numpy memory
    data = data.squeeze(0)
    if data.dim() == 4:
        data = data.permute(0, 3, 1, 2)   # view only -- normalization is already done
    return data

def gaussianize_target(data, sigma=1):
    N, H, W = data.shape
    out = np.empty((N, H, W), dtype=np.float16)
    for i in range(N):
        out[i] = make_gaussian_heatmap(data[i], sigma)
    return out