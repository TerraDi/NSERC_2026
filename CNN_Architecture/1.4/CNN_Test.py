import unittest
import numpy as np
import pytest
#import tools_solutions as tools
#import import_ipynb
#from IPython import get_ipython
import Functions
import numpy as np
import torch


def test():
    print("Hello from Test python file")

def test_genfields_shape():
    Nfields = np.random.randint(1, 5)
    correct_shape_combined = (Nfields, 100, 100, 608)
    correct_shape_target = (Nfields, 100, 100)
    combined_fields, target_maps, bgpositions, transientspositions = Functions.genfields(Nfields=Nfields)
    
    #print(np.shape(combined_fields))
    assert np.shape(combined_fields) == correct_shape_combined
    assert np.shape(target_maps) == correct_shape_target

def test_nan_inf():
    Nfields = np.random.randint(1, 5)
    #correct_shape_combined = (Nfields, 100, 100, 608)
    #correct_shape_target = (Nfields, 100, 100)
    combined_fields, target_maps, bgpositions, transientspositions = Functions.genfields(Nfields=Nfields)
    
    assert np.isnan(combined_fields).any().item() == False
    assert np.isinf(combined_fields).any().item() == False
    assert np.isnan(target_maps).any().item() == False
    assert np.isinf(target_maps).any().item() == False
    #assert np.isposinf(combined_fields) == False
    #assert np.isnan(bgpositions).any().item() == False
    #assert np.isinf(bgpositions).any().item() == False
    #assert np.isnan(transientspositions).any().item() == False
    #assert np.isinf(transientspositions).any().item() == False



test()
test_genfields_shape()
test_nan_inf()
print("Reached here")