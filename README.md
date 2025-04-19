# LSU Factorization in PyTorch

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a Python implementation of the LSU Factorization algorithm using PyTorch, based on the work by Gennadi Malaschonok.

LSU factorization generalizes LU and LEU decomposition, aiming to work over commutative domains and their quotient fields (though this implementation uses standard floating-point numbers).

## Overview

LSU factorization decomposes a square matrix $A$ into a product of three matrices:

$$ A = \alpha L S U $$

where:
*   $\alpha$: A scalar factor accumulated during recursion, related to determinants of sub-blocks.
*   $L$: A lower triangular matrix of full rank.
*   $U$: An upper triangular matrix of full rank.
*   $S$: A matrix belonging to the Weighted Permutations Semigroup ($S_{wp}$). Its rank $r$ is equal to the rank of $A$, and its $r$ non-zero elements are related to the inverses of products of nested non-degenerate minors of $A$.

This factorization is particularly useful because the elements of $L$ and $U$ are minors of the original matrix $A$. When computations can be performed exactly (e.g., over integers or finite fields, though this implementation uses floating-point), it avoids the numerical instability issues associated with traditional LU factorization.

The algorithm also computes auxiliary matrices $M$ and $W$, and a related matrix $\hat{S}$, which satisfy:

$$ L \hat{S} M = I $$
$$ W \hat{S} U = I $$

where $I$ is the identity matrix and $\hat{S} = \alpha_r^{-1} (\alpha S + \bar{S})$. Here, $\alpha_r$ is the final determinant factor from the recursion, and $\bar{S}$ is the complement of $S$ derived from the extended mapping (see paper for details).

## Features

*   Pure Python implementation using PyTorch tensors.
*   Implements the dichotomous recursive algorithm described in the paper (Section VI).
*   Handles square matrices where the dimension $n$ is a power of 2.
*   Computes the core $L, S, U$ factors.
*   Computes auxiliary matrices $M, W, \hat{S}, \bar{S}$ and projection matrices $I, J, \bar{I}, \bar{J}$.
*   Includes optional internal validation checks (`perform_checks=True`) to verify properties during recursion.
*   Calculates a pseudo-inverse $P = \alpha_r^{-2} W S M$.

## Requirements

*   Python (>= 3.8 recommended)
*   PyTorch

## Installation

Clone the repository:

```bash
git clone https://github.com/fl0wbar/lsu.git
cd lsu
```

## Usage

Import the `lsu_factorization` function and pass your matrix (as a PyTorch tensor) to it.

```python
import torch

from lsu import lsu_factorization, _check_lsu_final_properties

# Ensure matrix dimension is a power of 2
A = torch.tensor([[0., 0., 3., 0.],
                  [2., 0., 1., 0.],
                  [0., 0., 0., 0.],
                  [1., 4., 0., 1.]], dtype=torch.float64)

# Use float64 for better precision matching the paper's example
A = A.to(dtype=torch.float64)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A = A.to(device)

# Perform factorization (enable checks for debugging)
result = lsu_factorization(A, perform_checks=False, check_atol=1e-8)

if result:
    L, S, U, M, W, S_hat, S_bar, I, J, I_bar, J_bar, alpha_r = result
    print(f"Factorization successful! Final alpha_r = {alpha_r.item()}")

    # --- Verify Final Properties ---
    Id_n = torch.eye(A.shape[0], device=device, dtype=A.dtype)
    LSU = L @ S @ U
    LShatM = L @ S_hat @ M
    WShatU = W @ S_hat @ U

    print(f"Check A == L @ S @ U : {torch.allclose(A, LSU, atol=1e-8)}")
    print(f"Check I == L @ S_hat @ M : {torch.allclose(Id_n, LShatM, atol=1e-8)}")
    print(f"Check I == W @ S_hat @ U : {torch.allclose(Id_n, WShatU, atol=1e-8)}")

    # Check pseudo-inverse and other properties
    _check_lsu_final_properties(
        A=A, L=L, S=S, U=U, M=M, W=W,
        S_hat=S_hat, S_bar=S_bar, alpha_r=alpha_r, alpha=1.0, atol=1e-8
    )

else:
    print("Factorization failed (check input matrix dimensions or logs).")

```

## Algorithm Details

The implementation follows the recursive structure outlined in Section VI of the paper.
1.  **Base Cases:**
    *   If $A$ is numerically zero, return identity matrices and zero $S$.
    *   If $A$ is a 1x1 matrix $[a]$, compute factors directly.
2.  **Recursive Step (n >= 2):**
    *   Split $A$ into four $n/2 \times n/2$ blocks: $A_{11}, A_{12}, A_{21}, A_{22}$.
    *   Recursively call `lsu_factorization` four times on derived matrices (corresponding to steps 3.1, 3.2, 3.3, 3.4 in the paper's algorithm description).
    *   Compute intermediate matrices based on the results of the recursive calls (e.g., `$A'_{12}$`, `$A'_{21}$`, `$A''_{22}$`). Helper functions like `_ginv_swp`, `_unit_mapping`, `_extended_mapping` implement concepts from Section III of the paper.
    *   Assemble the final $L, S, U, M, W, \hat{S}, \bar{S}, I, J, \bar{I}, \bar{J}$ matrices from the block results according to equations (18) through (27).

## Mathematical Properties

The computed factors satisfy several key properties derived in the paper (Section IV):

*   **Factorization:** $A = \alpha L S U$ (where $\alpha=1$ for the initial call)
*   **Inverse Relations:** $L \hat{S} M = I$ and $W \hat{S} U = I$
*   **S_hat Definition:** $\hat{S} = \alpha_r^{-1} (\alpha S + \bar{S})$
*   **Projection Properties:**
    *   $I = E E^T$, $J = E^T E$, where $E$ is the unit mapping of $S$.
    *   $\bar{I} = I_d - I$, $\bar{J} = I_d - J$
    *   $I S J = S$
    *   $\bar{I} S = 0$, $S \bar{J} = 0$
    *   $L \bar{I} = \bar{I}$, $\bar{J} U = \bar{J}$
*   **Pseudo-Inverse:** The matrix $P = \alpha_r^{-2} W S M$ acts as a pseudo-inverse:
    *   $P A P = P$
    *   $A P A = A$
    *   Note: $P$ is generally *not* the Moore-Penrose inverse, meaning $P A \neq (P A)^T$ and $A P \neq (A P)^T$ might not hold.

## Validation

The code includes internal checks (`_lsu_checks`) that can be activated with `perform_checks=True`. These verify the mathematical properties listed above at each level of the recursion. A final check function (`_check_lsu_final_properties`) verifies the properties of the pseudo-inverse $P$.

## Example

`lsu.py` contains a runnable example using the 4x4 matrix from Section IX of the paper.

### `lsu_factorization` Parameters

*   `A` (`torch.Tensor`): The input square matrix ($n \times n$). $n$ must be a power of 2.
*   `alpha` (`float | torch.Tensor`, optional): Scaling factor from the parent recursion level. Defaults to 1.0.
*   `eps` (`float`, optional): Small tolerance for safe division and zero checks. Defaults to `1e-12`.
*   `depth` (`int`, optional): Current recursion depth (for internal logging).
*   `perform_checks` (`bool`, optional): If `True`, performs detailed validation checks within each recursive step. Useful for debugging but significantly slows down execution. Defaults to `False`.
*   `check_atol` (`float`, optional): Absolute tolerance used for floating-point comparisons during validation checks if `perform_checks` is `True`. Defaults to `1e-7`.

### `lsu_factorization` Return Values

Returns a tuple containing:
`(L, S, U, M, W, S_hat, S_bar, I, J, I_bar, J_bar, alpha_r)`
or `None` if input validation fails.

*   `L`, `S`, `U`: The core factorization matrices.
*   `M`, `W`: Inverse-related auxiliary matrices.
*   `S_hat`: Derived from $S$ and $\bar{S}$.
*   `S_bar`: Complementary matrix to $S$.
*   `I`, `J`: Projection matrices based on the non-zero structure of $S$.
*   `I_bar`, `J_bar`: Complementary projection matrices ($\bf{I}$ - $I$, $\bf{I}$ - $J$).
*   `alpha_r`: The final accumulated determinant scaling factor.


## Limitations and Notes

*   **Input Size:** The current recursive implementation requires the input matrix dimension $n$ to be a power of 2 ($n = 2^k$). Padding or blocking strategies would be needed for general sizes.
*   **Floating-Point Arithmetic:** This implementation uses standard PyTorch floating-point tensors. It does not perform exact symbolic computation as might be possible in a commutative domain setting. Numerical precision limitations apply.
*   **Complexity:** The algorithm has a time complexity related to matrix multiplication, approximately $O(n^\beta)$ where $2 < \beta \le 3$.
*   **Recursive Structure:** The algorithm recursively breaks down the matrix into four quadrants and solves subproblems.
*   **Weighted Permutation Semigroup ($Swp$):** The central matrix $S$ has at most one non-zero element per row and column.
*   **Auxiliary Matrices:** Computes $M, W$ related to inverses, and $\hat{S} = \frac{1}{\alpha_r}(\alpha S + \bar{S})$. Also computes projection matrices $I, J$ and their complements $\bar{I}, \bar{J}$.
*   **Numerical Stability:** While the original paper emphasizes computation without errors (e.g., over integers or fields), this implementation uses floating-point numbers. The structure might still offer benefits for certain matrix types compared to standard LU.


## References

1.  Malaschonok, G. (2022). *LSU factorization*. Preprint submitted to Journal of Computational Science. [SSRN: 4331134](https://ssrn.com/abstract=4331134)
2.  Malaschonok, G. (2025). *LSU factorization*. [arXiv:2503.13640 \[cs.SC\]](https://arxiv.org/abs/2503.13640) (Likely an updated version of the preprint)
3.  Malaschonok, G. (2023). LSU Factorization. In *2023 International Conference on Computational Science and Computational Intelligence (CSCI)* (pp. 472-478). IEEE. [DOI: 10.1109/CSCI62032.2023.00083](https://doi.org/10.1109/CSCI62032.2023.00083)

## License

This project is licensed under the MIT License.
