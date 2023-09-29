(Chap_Numpy)=


# NumPy

ACME materials link...


# Exercises

TODO: update excerises to be more relevant to OG-Core. e.g., matrix represents savings.  Then we do some array operations to multiple savings by the "population distribution" matrix omega.  Then sum to get total savings, accounting for population weights.

1. Create a Numpy array `b` (defined this as the savings of 2 types of agents over 5 periods):
   $$ \[
M=
  \begin{bmatrix}
    1 & 2 & 3 & 4 & 5 \\
    3 & 4 & 5 & 6 & 7
  \end{bmatrix}
\].  Use the `shape` method of Numpy arrays to print the shape of this matrix.  Use array slicing to print the first row of `A`.  Use array slicing to print the second column of `A`.  Use array slicing to print the first two rows and the last three columns of `A`.
2. Reshape the matrix `A` unto a 5x2 matrix.  Assign this new matrix to name `B`.  Use matrix multiplication to find `C`, which is the dot product of `A` and `B`.
3. Multiply the matrix `A` by itself element-wise (Hadamard product).
4. In one line, create a matrix of zeros that is the same size as `A`.
5. Something with `np.where`.
6. Something with appending/stacking.