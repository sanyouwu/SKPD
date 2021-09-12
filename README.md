# SKPD: A General Framework of Signal Region Detection in Image Regression

## Citations

Wu, S. and Feng, L. (2021)  SKPD: A General Framework of Signal Region Detection in Image Regression".

JRSSB, revision and resubmission

## Environment and usage

The following environments are required:

- Python 3.7 (anaconda is preferable)
- k3d (for visualize tensor case, not necessarily required)

The usage refer to the **Examples.ipynb**

##  Matrix example shows

An illustration of estimated coefficients $ \hat{C} \in\mathbb{R}^{128\times 128}$ in the linear model simulation with $n=1000$ and noise level $\sigma = 1$. From left to right columns: True signals; Matrix Regression; Tensor regression; STGP; our 1-term SKPD; 3-term SKPD.

![matrix_examples](https://github.com/SanyouWu/SKPD/blob/main/matrix_examples.png)

## Human brain examples

An illustration of estimated tensor coefficients when the true signal is “one-ball” (top three rows) and “two-balls” (bottom three rows) with $n= 1500$ and noise level $\sigma = 3$.The first row is true signal in a brain MRI template and its slice from the coronal, sagittal and horizontal sections. The second and third rows are one-term SKPD and 3-terms SKPD results respectively.



![brain_region_one_ball](https://github.com/SanyouWu/SKPD/blob/main/brain_region_one_ball.png)

![brain_region_2_balls](https://github.com/SanyouWu/SKPD/blob/main/brain_region_2_balls.png)

