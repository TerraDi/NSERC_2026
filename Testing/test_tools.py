import unittest
import numpy as np
import pytest
import tools

# Create a random array
n = 4
rand_array = np.random.normal(size=(n,n))

def test_square():
    print(rand_array)

    # Test the square function to see if it
    # returns an array of the right size
    output_arr = tools.square(rand_array)
    assert output_arr.shape == (n, n)

    # Test the square function to make sure that squaring
    # a matrix containing only 1s and 0s returns the same thing
    test_arr = np.diag(np.ones(n))
    output_arr = tools.square(test_arr)
    for i in range(n):
        for j in range(n):
            assert output_arr[i,j] == test_arr[i,j]

    # Test the square function for a known input/output
    test_arr = np.array([1, 2])
    output_arr = tools.square(test_arr)
    assert output_arr[0] == 1
    assert output_arr[1] == 4


def test_get_pi():
    # See if the get_pi function really returns pi
    alleged_pi = tools.get_pi()
    assert alleged_pi == pytest.approx(3.141592653589)

def test_picky():
    # Make sure that tools.picky raises an error if the
    # wrong type is inputted
    pytest.raises(TypeError, tools.picky, 'hey')
