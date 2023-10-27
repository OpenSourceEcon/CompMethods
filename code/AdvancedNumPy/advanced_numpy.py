# advanced_numpy.py
"""Python Essentials: Advanced NumPy.
<Name>
<Class>
<Date>
"""
import numpy as np
from sympy import isprime
from matplotlib import pyplot as plt


def prob1(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    raise NotImplementedError("Problem 1 Incomplete")


def prob2(arr_list):
    """return all arrays in arr_list as one 3-dimensional array
    where the arrays are padded with zeros appropriately."""
    raise NotImplementedError("Problem 2 Incomplete")


def prob3(func, A):
    """Time how long it takes to run func on the array A in two different ways,
    where func is a universal function.
    First, use array broadcasting to operate on the entire array element-wise.
    Second, use a nested for loop, operating on each element individually.
    Return the ratio showing how many times faster array broadcasting is than
    using a nested for loop, averaged over 10 trials.

    Parameters:
            func -- array broadcast-able numpy function
            A -- nxn array to operate on
    Returns:
            num_times_faster -- float
    """
    raise NotImplementedError("Problem 3 Incomplete")


def prob4(A):
    """Divide each row of 'A' by the row sum and return the resulting array.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    raise NotImplementedError("Problem 4 Incomplete")


# this is provided for problem 5
def LargestPrime(x, show_factorization=False):
    # account for edge cases.
    if x == 0 or x == 1:
        return np.nan

    # create needed variables
    forced_break = False
    prime_factors = []  # place to store factors of number
    factor_test_arr = np.arange(1, 11)

    while True:
        # a factor is never more than half the number
        if np.min(factor_test_arr) > (x // 2) + 1:
            forced_break = True
            break
        if isprime(x):  # if the checked number is prime itself, stop
            prime_factors.append(x)
            break

        # check if anythin gin the factor_test_arr are factors
        div_arr = x / factor_test_arr
        factor_mask = div_arr - div_arr.astype(int) == 0
        divisors = factor_test_arr[factor_mask]
        if divisors.size > 0:  # if divisors exist...
            if (
                divisors[0] == 1 and divisors.size > 1
            ):  # make sure not to select 1
                i = 1
            elif (
                divisors[0] == 1 and divisors.size == 1
            ):  # if one is the only one don't pick it
                factor_test_arr = factor_test_arr + 10
                continue
            else:  # othewise take the smallest divisor
                i = 0

            # if divisor was found divide number by it and
            # repeat the process
            x = int(x / divisors[i])
            prime_factors.append(divisors[i])
            factor_test_arr = np.arange(1, 11)
        else:  # if no number was found increase the test_arr
            # and keep looking for factors
            factor_test_arr = factor_test_arr + 10
            continue

    if show_factorization:  # show entire factorization if desired
        print(prime_factors)
    if forced_break:  # if too many iterations break
        print(f"Something wrong, exceeded iteration threshold for value: {x}")
        return 0
    return max(prime_factors)


def prob5(arr, naive=False):
    """Return an array where every number is replaced be the largest prime
    in its factorization. Implement two methods. Switching between the two
    is determined by a bool.

    Example:
        >>> A = np.array([15, 41, 49, 1077])
        >>> prob4(A)
        array([5,41,7,359])
    """
    raise NotImplementedError("Problem 5 Incomplete")


def prob6(x, y, z, A, optimize=False, split=True):
    """takes three vectors and a matrix and performs
    (np.outer(x,y)*z.reshape(-1,1))@A on them using einsum."""
    raise NotImplementedError("Problem 6 part 1 Incomplete")


def naive6(x, y, z, A):
    """uses normal numpy functions to do what prob5 does"""
    raise NotImplementedError("Problem 6 part 2 Incomplete")


def prob7():
    """Times and creates plots that generate the difference in
    speeds between einsum and normal numpy functions
    """
    raise NotImplementedError("Problem 7 Incomplete")
