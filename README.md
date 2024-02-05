# Deblurring Image - Assigment Advanced Programming Winter 2023/24

This repository contains the code for the image deblurring assignment as part of the Advanced Programming coursework. 
The assignment involves implementing a blur matrix, Tychonov-regularized reconstruction, visualizing singular values, and testing the efficiency of different solvers.

## Contents:

1. `operators.py`: Implementation of blur_matrix, tychonov_matrices, and tychonov_operators functions.
2. `inverse.ipynb`: Code for visualizing the original image, blurred noisy image, and Tychonov-regularized reconstructions using different alpha values, timing tests for different solvers - spsolve,
                    cg with tychonov_matrices, and cg with tychonov_operators.
