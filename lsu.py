from __future__ import annotations

import math  # For checking power of 2

import torch

# Define a small epsilon for safe division and zero checks
DEFAULT_EPS = 1e-12


def _safe_inv(x: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """
    Computes the element-wise inverse of a tensor safely, avoiding division by zero.

    For elements close to zero (within `eps`), returns `1/eps` with the original sign.
    Handles potential NaN/Inf values resulting from division by very small numbers.

    Args:
        x: Input tensor.
        eps: Threshold below which values are considered close to zero.

    Returns:
        Element-wise inverse of x, stabilized around zero.
    """
    # Use torch.where for conditional selection based on absolute value
    # Keep the original sign even for values close to zero
    # Note: This returns 1/eps for near-zero elements, preserving the scale's inverse nature
    #       rather than returning 0, which might be appropriate in other contexts.
    return torch.where(torch.abs(x) < eps, torch.sign(x) / eps, 1.0 / x)


def _ginv_swp(S: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """
    Computes the generalized inverse specific to matrices in the Swp semigroup.
    (Weighted Permutations Semigroup) as defined in Property 3 of the paper.

    As per Property 3 in the paper:
    S+ = {(b_ij): b_ij = 0 if a_ji = 0, b_ij = a_ji^-1 if a_ji != 0}.
    This involves transposing the matrix and taking the element-wise safe inverse
    of the non-zero elements of the *transposed* matrix.

    Args:
        S: Input tensor, expected to be from the Swp semigroup (at most one
           non-zero element per row/column, though not strictly enforced here).
        eps: Threshold for considering elements as non-zero during inversion.

    Returns:
        The generalized inverse S+ specific to Swp structure.
    """
    S_T = S.T
    # Apply safe inverse only to elements that were non-zero in the transpose
    inv_S_T = torch.where(torch.abs(S_T) < eps, 0.0, _safe_inv(S_T, eps))
    return inv_S_T


def _unit_mapping(S: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """
    Applies the unit mapping (Definition 1).

    Replaces all non-zero elements (abs value > eps) of the input matrix with 1.0,
    while keeping zero elements as 0.0. This corresponds to the homomorphism F* -> 1.
    The result is a matrix in Sp.
    S --> S->1 (denoted as E in the paper's recursive formulas, Eq 2).

    Args:
        S: Input matrix (typically the S matrix from LSU).
        eps: Threshold for considering elements as non-zero.

    Returns:
        Matrix E with 1s where S was non-zero, 0s otherwise.
    """
    return torch.where(torch.abs(S) > eps, 1.0, 0.0)


def _extended_mapping(S: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    """
    Applies the extended mapping (Definition 2).

    Creates S_ext by taking S and replacing the sub-block at the intersection
    of zero-rows and zero-columns with a unit block (block of ones).

    Args:
        S: Input matrix from Swp.
        eps: Threshold for considering rows/columns as zero.

    Returns:
        The extended matrix S_ext.
    """
    S_ext = S.clone()
    # Identify rows where all elements are close to zero
    is_zero_row = torch.all(torch.abs(S) < eps, dim=1)
    # Identify columns where all elements are close to zero
    is_zero_col = torch.all(torch.abs(S) < eps, dim=0)

    # Get indices of zero rows and columns
    zero_row_indices = torch.where(is_zero_row)[0]
    zero_col_indices = torch.where(is_zero_col)[0]

    # If both zero rows and zero columns exist, fill the intersection sub-block
    if zero_row_indices.numel() > 0 and zero_col_indices.numel() > 0:
        # Use broadcasting to select the sub-block and fill with 1.0
        S_ext[zero_row_indices[:, None], zero_col_indices] = 1.0

    return S_ext


torch.set_printoptions(precision=4, sci_mode=False, linewidth=140)


def _lsu_checks(A, L, S, U, M, W, S_hat, S_bar, I, J, I_bar, J_bar, alpha, context_str="init", depth=0, atol=1e-7):
    """Internal helper to perform validation checks during recursion."""
    n, device, dtype = A.shape[0], A.device, A.dtype

    atol = max(1e-6, atol) if dtype == torch.float32 else atol

    Id = torch.eye(n, device=device, dtype=dtype)
    Z = torch.zeros_like(A)
    failed_checks = []
    indent = "    " * depth

    print(f"\n{indent}--- LSU Checks @ Depth={depth}, Context='{context_str}' ---")
    print(f"{indent}    Matrix Size: {A.shape}, alpha: {alpha.item():.4g}")

    def _print_fail(chk_name, lhs_expr, lhs_val, rhs_expr, rhs_val):
        diff = torch.abs(lhs_val - rhs_val)
        max_diff = torch.max(diff).item()
        print(f"{indent}    [FAIL] Check: {chk_name}")
        print(f"{indent}           Max Diff: {max_diff:.4g}")
        # Optional: Print matrices on failure for detailed debugging
        print(
            f"{indent}           LHS ({lhs_expr}):\n{indent}           {str(lhs_val).replace(chr(10), chr(10)+indent+'           ')}"
        )
        print(
            f"{indent}           RHS ({rhs_expr}):\n{indent}           {str(rhs_val).replace(chr(10), chr(10)+indent+'           ')}"
        )
        failed_checks.append(chk_name)

    # Check 1: A = alpha * L @ S @ U (Eq 1)
    check_name = "A == alpha * L @ S @ U"
    lhs = alpha * (L @ S @ U)
    rhs = A
    if not torch.allclose(lhs, rhs, atol=atol):
        _print_fail(check_name, "alpha*(L @ S @ U)", lhs, "A", rhs)

    # Check 2: L @ S_hat @ M = Identity (Eq 1 implicitly, required for inverse)
    check_name = "L @ S_hat @ M == Id"
    lhs = L @ S_hat @ M
    rhs = Id
    if not torch.allclose(lhs, rhs, atol=atol):
        _print_fail(check_name, "L @ S_hat @ M", lhs, "Id", rhs)

    # Check 3: W @ S_hat @ U = Identity (Eq 1 implicitly, required for inverse)
    check_name = "W @ S_hat @ U == Id"
    lhs = W @ S_hat @ U
    rhs = Id
    if not torch.allclose(lhs, rhs, atol=atol):
        _print_fail(check_name, "W @ S_hat @ U", lhs, "Id", rhs)

    # Check 4: I @ S @ J = S (Property derived from Eq 2)
    check_name = "I @ S @ J == S"
    lhs = I @ S @ J
    rhs = S
    if not torch.allclose(lhs, rhs, atol=atol):
        _print_fail(check_name, "I @ S @ J", lhs, "S", rhs)

    # Check 5: I_bar @ S = 0 (Property derived from Eq 2)
    check_name = "I_bar @ S == 0"
    lhs = I_bar @ S
    rhs = Z
    if not torch.allclose(lhs, rhs, atol=atol):
        _print_fail(check_name, "I_bar @ S", lhs, "Zero", rhs)

    # Check 6: S @ J_bar = 0 (Property derived from Eq 2)
    check_name = "S @ J_bar == 0"
    lhs = S @ J_bar
    rhs = Z
    if not torch.allclose(lhs, rhs, atol=atol):
        _print_fail(check_name, "S @ J_bar", lhs, "Zero", rhs)

    # Check 7: L @ I_bar = I_bar (Eq 3)
    check_name = "L @ I_bar == I_bar"
    lhs = L @ I_bar
    rhs = I_bar
    if not torch.allclose(lhs, rhs, atol=atol):
        _print_fail(check_name, "L @ I_bar", lhs, "I_bar", rhs)

    # Check 8: J_bar @ U = J_bar (Eq 3)
    check_name = "J_bar @ U == J_bar"
    lhs = J_bar @ U
    rhs = J_bar
    if not torch.allclose(lhs, rhs, atol=atol):
        _print_fail(check_name, "J_bar @ U", lhs, "J_bar", rhs)

    if failed_checks:
        print(
            f"{indent}    !!! {len(failed_checks)} Checks Failed @ Depth={depth}, Context='{context_str}': {failed_checks}"
        )
    else:
        print(f"{indent}    +++ All Checks Passed @ Depth={depth}, Context='{context_str}' +++")
    print(f"{indent}--- End Checks @ Depth={depth} ---")


def _check_lsu_final_properties(A, L, S, U, M, W, S_hat, S_bar, alpha_r, alpha=1, context_str="final", atol=1e-7):
    """Internal helper to perform validation checks on the final result."""
    n, device, dtype = A.shape[0], A.device, A.dtype
    Id = torch.eye(n, device=device, dtype=dtype)
    failed_checks = []
    indent = "  "

    atol = max(1e-6, atol) if dtype == torch.float32 else atol

    print(f"\n--- LSU Final Property Checks, Context='{context_str}' ---")
    print(f"    Matrix Size: {A.shape}, alpha_r (det_r scale): {alpha_r.item():.4g}")

    det_r_sq_inv = _safe_inv(alpha_r**2)

    # if A is invertible, then P forms the inverse of matrix A
    P = det_r_sq_inv * (W @ S @ M)  # Pseudo-inverse (Eq before IV.B)

    # L_inv = S_hat @ M
    # U_inv = W @ S_hat

    def _print_fail(chk_name, lhs_expr, lhs_val, rhs_expr, rhs_val):
        diff = torch.abs(lhs_val - rhs_val)
        max_diff = torch.max(diff).item()
        print(f"{indent}    [FAIL] Check: {chk_name}")
        print(f"{indent}           Max Diff: {max_diff:.4g}")
        # Optional: Print matrices on failure for detailed debugging
        print(
            f"{indent}           LHS ({lhs_expr}):\n{indent}           {str(lhs_val).replace(chr(10), chr(10)+indent+'           ')}"
        )
        print(
            f"{indent}           RHS ({rhs_expr}):\n{indent}           {str(rhs_val).replace(chr(10), chr(10)+indent+'           ')}"
        )
        failed_checks.append(chk_name)

    # Check 1: A @ P = Id (Only if A is invertible)
    # AP = LSU (1/det_r^2) WSM
    check_name = "A @ P = (L @ S @ U) * (1 / det_r**2) * (W @ S @ M) = Id"

    lhs = A @ P
    rhs_1 = (L @ S @ U) * det_r_sq_inv * (W @ S @ M)
    rhs_2 = Id
    is_close_1 = torch.allclose(lhs, rhs_1, atol=atol)
    is_close_2 = torch.allclose(lhs, rhs_2, atol=atol)
    is_close_3 = torch.allclose(rhs_1, rhs_2, atol=atol)
    print(
        f"{indent}    Check [A invertibility]: {check_name:<35} | Result: {is_close_1}-{is_close_2}-{is_close_3} (atol={atol})"
    )
    if not (is_close_1 and is_close_2 and is_close_3):
        _print_fail(check_name, "A@P", lhs, "(L @ S @ U) * (1 / det_r**2) * (W @ S @ M)", rhs_1)
        _print_fail(check_name, "A@P", lhs, "Id", rhs_2)
        _print_fail(check_name, "(L @ S @ U) * (1 / det_r**2) * (W @ S @ M)", rhs_1, "Id", rhs_2)

    # Check 2: Let's check S_hat = (1/alpha_r)*(alpha*S + S_bar) instead
    check_name = "S_hat == (1/alpha_r)*(alpha*S + S_bar)"
    lhs = S_hat
    rhs = _safe_inv(alpha_r) * (alpha * S + S_bar)
    if not torch.allclose(lhs, rhs, atol=atol):
        _print_fail(check_name, "S_hat", lhs, "(1/alpha_r)*(alpha*S + S_bar)", rhs)

    # Check 3.1: P @ A @ P = P (Pseudo-inverse property)
    check_name = "P @ A @ P == P"
    lhs = P @ A @ P
    rhs = P
    if not torch.allclose(lhs, rhs, atol=atol):
        _print_fail(check_name, "P @ A @ P", lhs, "P", rhs)

    # Check 3.2: A @ P @ A = A (Pseudo-inverse property)
    check_name = "A @ P @ A == A"
    lhs = A @ P @ A
    rhs = A
    if not torch.allclose(lhs, rhs, atol=atol):
        _print_fail(check_name, "A @ P @ A", lhs, "A", rhs)

    # Check 4: P @ A = (P @ A)^T (Moore-Penrose property - NOT expected for this P)
    check_name = "P @ A == (P @ A)^T (MP check - Not Expected)"
    lhs = P @ A
    rhs = lhs.T
    is_mp_close1 = torch.allclose(lhs, rhs, atol=atol)
    print(f"{indent}    Check: {check_name:<35} | Result: {is_mp_close1} (Not Expected)")
    if not is_mp_close1:
        _print_fail(check_name, "P @ A", lhs, "(P @ A)^T", rhs)

    # No failure log needed as it's expected to fail

    # Check 5: A @ P = (A @ P)^T (Moore-Penrose property - NOT expected for this P)
    check_name = "A @ P == (A @ P)^T (MP check - Not Expected)"
    lhs = A @ P
    rhs = lhs.T
    is_mp_close2 = torch.allclose(lhs, rhs, atol=atol)
    print(f"{indent}    Check: {check_name:<35} | Result: {is_mp_close2} (Not Expected)")
    if is_mp_close2:
        _print_fail(check_name, "A @ P", lhs, "(A @ P)^T", rhs)

    # No failure log needed

    if failed_checks:
        print(f"{indent}    !!! {len(failed_checks)} Final Property Checks Failed:")
        for f_chk in failed_checks:
            print(f"{indent}    {f_chk}")
    else:
        print(f"{indent}    +++ All Expected Final Property Checks Passed +++")

    print(f"{indent}    NOTE: Check 'A@P == Id' holds only if A is invertible.")
    print(f"{indent}    NOTE: Checks 'P@A@P == P' and 'A@P@A == A' should always hold.")
    print(f"{indent}    NOTE: Checks 'P@A == (P@A)^T' and 'A@P == (A@P)^T' (Moore-Penrose) are NOT expected to hold.")
    print(
        f"{indent}    NOTE: -- since P is a pseudo-inverse, but is not the Moore-Penrose generalized inverse for the matrix A."
    )
    print(f"{indent}--- End Final Property Checks ---")


def lsu_factorization(
    A: torch.Tensor,
    alpha: float | torch.Tensor = 1.0,
    eps: float = DEFAULT_EPS,
    depth: int = 0,
    perform_checks: bool = False,
    check_atol: float = 1e-7,
):
    """
    Performs LSU factorization A = alpha * L @ S @ U recursively based on the paper
    "LSU Factorization" by Gennadi Malaschonok (arXiv:2503.13640v1).

    This function implements the dichotomous recursive algorithm
    described in Section V and VI of the paper.
    It decomposes a square matrix A (whose dimension n must be a power of 2)
    from R^(n x n) (approximated by floating-point numbers) into:
    - L: Lower triangular matrix (rank n).
    - U: Upper triangular matrix (rank n).
    - S: A matrix from the Weighted Permutations Semigroup Swp(F) (rank r <= n),
         whose non-zero elements relate to inverses of products of nested minors of A.
    - M, W: Inverse factors such that L @ S_hat @ M = I and W @ S_hat @ U = I.
    - S_hat: Related to S and its extended mapping complement S_bar.
             S_hat = alpha_r^-1 * (alpha * S + S_bar).
    - S_bar: Complement of S from the extended mapping (S_ext - S).
    - I, J: Projection matrices derived from S (I = E@E.T, J = E.T@E where E=unit_mapping(S)).
    - I_bar, J_bar: Complementary projections (Id - I, Id - J).
    - alpha_r: The final scalar value corresponding to the determinant of the
               last non-degenerate subproblem, scaled by intermediate alphas.

    This algorithm generalizes LU/LEU decomposition for matrices over commutative domains
    and their quotient fields. It decomposes A into a lower triangular matrix L,
    an upper triangular matrix U, and a weighted permutation matrix S (from Swp semigroup).
    S ∈ S_wp(F), of rank 'r', with 'r' non-zero elements equal to
    (alpha*det1)^-1 , (det1*det2)^-1, .., (detr-1*detr)^-1,
    Here, det_r, det_r-1, .., det_1 is a sequence consisting of 'r' nested nondegenerate
    minors of the matrix A, which have sizes r, r-1, .., 1, respectively.
    It also computes auxiliary matrices M and W related to inverses.

    Key Properties (see paper Section IV.A, Eq 1-3, IV.B):
    - A = alpha * (L @ S @ U)
    - L @ S_hat @ M = Identity
    - W @ S_hat @ U = Identity

    - S_hat = (1/alpha_r) * (alpha * S + S_bar), where alpha_r is the final determinant factor.

    - S_bar = extended_mapping(S) - S
    - E = S -> 1 (unit mapping)
    - I = E @ E^T
    - J = E^T @ E
    - I_bar = Id - I
    - J_bar = Id - J
    - I @ S @ J = S
    - I_bar @ S = zero
    - S @ J_bar = zero
    - L @ I_bar = I_bar
    - J_bar @ U = J_bar

    - If A is invertible, P = (1/alpha_r^2) * W @ S @ M is the inverse of A.
    - If A is not invertible, P is a pseudo-inverse satisfying P@A@P=P and A@P@A=A.

    --  If matrix A is invertible, then matrix
        P = (1 / det_r**2) * (W @ S @ M), is the inverse matrix for A
        --  L^-1 = S_hat @ M
        --  U^-1 = W @ S_hat
        --  S_hat = det_r * S
        so, A @ P = (L @ S @ U) * (1 / det_r**2) * (W @ S @ M) = Id

    --  If matrix A is not invertible, then one can easily check that two identities
        --  (P @ A @ P) = P and (A @ P @ A) = A
        hold. But, in the general case, identities P @ A = (P @ A)^T, A @ P = (A @ P)^T do not hold.
        Thus, in this case, matrix P is a pseudo-inverse, but is not the
        Moore-Penrose generalized inverse for the matrix A.

    Args:
        A: The input square matrix (n x n). n must be a power of 2.
        alpha: The scaling factor from the parent recursion level.
               Defaults to 1.0 for the initial call. Represents the determinant
               of the top-left block processed in the parent call.
        eps: Small value for safe division and zero checks.
        depth: Current recursion depth (for logging/debugging).
        perform_checks: If True, performs validation checks at each step (slow).
        check_atol: Absolute tolerance for floating-point comparisons in checks.

    Returns:
        Tuple: (L, S, U, M, W, S_hat, S_bar, I, J, I_bar, J_bar, alpha_r)
        Returns None if input validation fails.

    """
    device = A.device
    dtype = A.dtype
    n = A.shape[0]

    # Ensure alpha is a tensor for consistent operations
    if not isinstance(alpha, torch.Tensor):
        alpha = torch.as_tensor(alpha, device=device, dtype=dtype)

    # --- Input Validation ---
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        print(f"ERROR (Depth {depth}): Input matrix A must be square. Shape: {A.shape}")
        return None
    if n == 0:
        print(f"ERROR (Depth {depth}): Input matrix A cannot be empty.")
        return None
    # Check if n is a power of 2 (more efficient check)
    is_power_of_two = (n > 0) and ((n & (n - 1)) == 0)
    if not is_power_of_two and n > 1:
        print(f"ERROR (Depth {depth}): Matrix dimension n={n} must be a power of 2 " f"for this recursive algorithm.")
        # Future improvement: Implement padding/blocking for non-power-of-2 sizes.
        return None

    # construct functions for handling base cases of zeros and scalars
    def _case_zero(A, alpha, eps, depth):
        # print(f"DEBUG (Depth {depth}): Handling Zero Matrix Case")
        alpha_r = alpha  # set final determinant factor to input alpha
        # --------------------------------------------------
        Id = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        Z = torch.zeros_like(A)
        # --------------------------------------------------
        S = Z.clone()
        I = Z.clone()
        J = Z.clone()
        # --------------------------------------------------
        I_bar = Id.clone()
        J_bar = Id.clone()
        # --------------------------------------------------
        L = Id.clone()
        U = Id.clone()
        M = Id.clone() * alpha
        W = Id.clone() * alpha
        # --------------------------------------------------
        S_bar = Id.clone()  # S_bar = Ext(0) - 0 = I - 0 = I
        S_hat = Id.clone() * _safe_inv(alpha, eps)  # S_hat = ar^-1(a*0 + S_bar) = a^-1 * I
        # --------------------------------------------------
        if perform_checks:
            _lsu_checks(
                A=A,
                L=L,
                S=S,
                U=U,
                M=M,
                W=W,
                S_hat=S_hat,
                S_bar=S_bar,
                I=I,
                J=J,
                I_bar=I_bar,
                J_bar=J_bar,
                alpha=alpha,
                context_str=f"_case_zero-{depth=}",
                depth=depth,
                atol=check_atol,
            )
        # --------------------------------------------------
        return L, S, U, M, W, S_hat, S_bar, I, J, I_bar, J_bar, alpha_r

    # --- Base Case 1: A is the zero matrix (numerically) ---
    if torch.allclose(A, torch.zeros_like(A), atol=eps):
        return _case_zero(A, alpha, eps, depth)

    # --- Base Case 2: n = 1 (Scalar Case) ---
    def _case_scalar(A, alpha, eps, depth):
        # Proceed with n=1 calculation as per paper VI-(2)
        # print(f"DEBUG (Depth {depth}): Handling Scalar Case, a={a_scalar.item():.4g}")
        # final determinant factor (det_r) is the element itself
        alpha_r = A  # NOTE: paper says alpha_r = a for n=1 case
        # --------------------------------------------------
        L = A.clone()
        U = A.clone()
        M = A.clone()  # Paper VI-(2) M=[a]
        W = A.clone()  # Paper VI-(2) W=[a]
        # --------------------------------------------------
        S = _safe_inv(alpha * A, eps)
        S_hat = _safe_inv(A, eps) ** 2
        # --------------------------------------------------
        # Let's verify S_hat = ar^-1(a*S+S_bar)
        # S_bar = Ext(S) - S. If S!=0, Ext(S)=S => S_bar=0
        # S_hat = a^-1 * (alpha * (alpha*a)^-1 + 0) = a^-1 * a^-1 = a^-2. Matches.
        Iu, Z = torch.ones_like(A), torch.zeros_like(A)
        I = Iu.clone()  # E = [1] => I = E@E.T = [1]
        J = Iu.clone()  # J = E.T@E = [1]
        I_bar = Z.clone()  # I_bar = Id - I = [1] - [1] = [0]
        J_bar = Z.clone()  # J_bar = Id - J = [1] - [1] = [0]
        S_bar = Z.clone()  # Calculated above
        # --------------------------------------------------
        if perform_checks:
            _lsu_checks(
                A=A,
                L=L,
                S=S,
                U=U,
                M=M,
                W=W,
                S_hat=S_hat,
                S_bar=S_bar,
                I=I,
                J=J,
                I_bar=I_bar,
                J_bar=J_bar,
                alpha=alpha,
                context_str=f"_case_scalar-{depth=}",
                depth=depth,
                atol=check_atol,
            )
        # --------------------------------------------------
        return L, S, U, M, W, S_hat, S_bar, I, J, I_bar, J_bar, alpha_r

    if n == 1:
        # --------------------------------------------------
        # If the single element is close to zero, treat as zero matrix case
        if torch.abs(A[0, 0]) < eps:
            # print(f"DEBUG (Depth {depth}): Handling Near-Zero Scalar Case")
            return _case_zero(A, alpha, eps, depth)
        return _case_scalar(A, alpha, eps, depth)

    # --- Recursive Step (n >= 2) ---
    n_half = n // 2  # Midpoint for splitting

    # Split A into 4 blocks (views, no copies yet)
    A11, A12 = A[:n_half, :n_half], A[:n_half, n_half:]
    A21, A22 = A[n_half:, :n_half], A[n_half:, n_half:]

    # First step : factorization of the upper left block (A11)

    # --- Step 3.1: Recurse on A11 (Top-Left Block) ---
    # Result provides factorization for A11: A11 = alpha * L11 @ S11 @ U11
    # alpha_k is the determinant scalar output for the A11 subproblem.
    lsu_A11 = lsu_factorization(
        A=A11,
        alpha=alpha,
        eps=eps,
        depth=depth + 1,
        perform_checks=perform_checks,
        check_atol=check_atol,
    )
    if lsu_A11 is None:
        return None  # Propagate error
    L11, S11, U11, M11, W11, S_hat11, S_bar11, I11, J11, I_bar11, J_bar11, alpha_k = lsu_A11
    # alpha_k is det_k (determinant factor of A11 block relative to alpha)

    # checks for LSU factorization of A11
    if perform_checks:
        _lsu_checks(
            A=A11,
            L=L11,
            S=S11,
            U=U11,
            M=M11,
            W=W11,
            S_hat=S_hat11,
            S_bar=S_bar11,
            I=I11,
            J=J11,
            I_bar=I_bar11,
            J_bar=J_bar11,
            alpha=alpha,
            context_str=f"A11-depth-{depth + 1}",
            depth=depth + 1,
            atol=check_atol,
        )

    # --- Calculate intermediate matrices A'_12, A'_21 ---
    # Note: Paper uses A'12, A''12 etc. Renaming for clarity.

    A_0_12 = M11 @ A12
    A_0_21 = A21 @ W11

    A_1_12 = alpha_k * (S_hat11 @ A_0_12)
    A_1_21 = alpha_k * (A_0_21 @ S_hat11)

    A_2_12 = (S_bar11 @ A_0_12) * _safe_inv(alpha, eps)
    A_2_21 = (A_0_21 @ S_bar11) * _safe_inv(alpha, eps)

    # Second step : factorization of the lower left block (A21)

    # --- Step 3.2: Recurse on A''_21 (Lower-Left derived block) ---
    # Factorizes A''_21 = alpha_k * L21 @ S21 @ U21
    # alpha_l is the determinant scalar output for this subproblem.
    lsu_A_2_21 = lsu_factorization(
        A=A_2_21,
        alpha=alpha_k,
        eps=eps,
        depth=depth + 1,
        perform_checks=perform_checks,
        check_atol=check_atol,
    )
    if lsu_A_2_21 is None:
        return None
    L21, S21, U21, M21, W21, S_hat21, S_bar21, I21, J21, I_bar21, J_bar21, alpha_l = lsu_A_2_21
    # alpha_l is det_l (determinant factor of A''_21 relative to alpha_k)

    if perform_checks:
        _lsu_checks(
            A=A_2_21,
            L=L21,
            S=S21,
            U=U21,
            M=M21,
            W=W21,
            S_hat=S_hat21,
            S_bar=S_bar21,
            I=I21,
            J=J21,
            I_bar=I_bar21,
            J_bar=J_bar21,
            alpha=alpha_k,
            context_str=f"A_2_21-depth-{depth + 1}",
            depth=depth + 1,
            atol=check_atol,
        )

    # Third step : factorization of the upper left block (A12)

    # --- Step 3.3: Recurse on A''_12 (Top-Right derived block) ---
    # Factorizes A''_12 = alpha_k * L12 @ S12 @ U12
    # alpha_m is the determinant scalar output for this subproblem.
    lsu_A_2_12 = lsu_factorization(
        A=A_2_12,
        alpha=alpha_k,
        eps=eps,
        depth=depth + 1,
        perform_checks=perform_checks,
        check_atol=check_atol,
    )
    if lsu_A_2_12 is None:
        return None
    L12, S12, U12, M12, W12, S_hat12, S_bar12, I12, J12, I_bar12, J_bar12, alpha_m = lsu_A_2_12
    # alpha_m is det_m (determinant factor of A''_12 relative to alpha_k)

    if perform_checks:
        _lsu_checks(
            A=A_2_12,
            L=L12,
            S=S12,
            U=U12,
            M=M12,
            W=W12,
            S_hat=S_hat12,
            S_bar=S_bar12,
            I=I12,
            J=J12,
            I_bar=I_bar12,
            J_bar=J_bar12,
            alpha=alpha_k,
            context_str=f"A_2_12-depth-{depth + 1}",
            depth=depth + 1,
            atol=check_atol,
        )

    # --- Calculate intermediate determinant factors and matrix A''_22 ---

    # --- Calculate terms for the final recursion (Step 3.4) ---
    # lambda and alpha_s (scalars for next recursion level)
    lambda_factor = alpha_l * _safe_inv(alpha_k, eps)  # lambda = det_l / det_k
    alpha_s = lambda_factor * alpha_m  # alpha_s = det_s = lambda * det_m

    A_0_22 = A_1_21 @ _ginv_swp(S11, eps) @ A_1_12
    A_1_22 = (((alpha * alpha_k**2) * A22) - A_0_22) * _safe_inv(alpha * alpha_k, eps)
    A_2_22 = S_bar21 @ M21 @ A_1_22 @ W12 @ S_bar12
    A_3_22 = A_2_22 * _safe_inv((alpha_k**2 * alpha), eps)

    # Fourth step : factorization of the lower right block (A22)

    # --- Step 3.4: Recurse on A'''_22 (Bottom-Right derived block) ---
    # Factorizes A'''_22 = alpha_s * L22 @ S22 @ U22
    # alpha_r is the final determinant scalar output for the overall problem A.
    lsu_A_3_22 = lsu_factorization(
        A=A_3_22,
        alpha=alpha_s,
        eps=eps,
        depth=depth + 1,
        perform_checks=perform_checks,
        check_atol=check_atol,
    )
    if lsu_A_3_22 is None:
        return None
    L22, S22, U22, M22, W22, S_hat22, S_bar22, I22, J22, I_bar22, J_bar22, alpha_r = lsu_A_3_22
    # alpha_r is the final determinant factor (relative to alpha_s)

    if perform_checks:
        _lsu_checks(
            A=A_3_22,
            L=L22,
            S=S22,
            U=U22,
            M=M22,
            W=W22,
            S_hat=S_hat22,
            S_bar=S_bar22,
            I=I22,
            J=J22,
            I_bar=I_bar22,
            J_bar=J_bar22,
            alpha=alpha_s,
            context_str=f"A_3_22-depth-{depth + 1}",
            depth=depth + 1,
            atol=check_atol,
        )

    # --- Assemble the final matrices L, S, U, M, W, S_hat ---

    # Calculate projection matrices J_lambda_12, I_lambda_12
    J_lambda_12 = lambda_factor * J12 + J_bar12
    I_lambda_12 = lambda_factor * I12 + I_bar12

    # Calculate L_tilde_12, U_tilde_12
    L_tilde_12 = L12 @ I_lambda_12
    U_tilde_12 = J_lambda_12 @ U12

    # Calculate L3 block
    term1_L3 = (A21 @ W11 @ I11) * _safe_inv(alpha_k, eps)
    term2_L3 = (S_bar21 @ M21 @ A_1_22 @ W12 @ I12) * _safe_inv(alpha_m * alpha_k * alpha, eps)
    L3 = term1_L3 + term2_L3

    # Calculate U2 block
    term1_U2 = (J11 @ M11 @ A12) * _safe_inv(alpha_k, eps)
    term2_U2 = (J21 @ M21 @ A_1_22) * _safe_inv(alpha_l * alpha, eps)
    U2 = term1_U2 + term2_U2

    # Assemble L
    L = torch.zeros_like(A)
    L[:n_half, :n_half] = L11 @ L_tilde_12
    L[n_half:, :n_half] = L3
    L[n_half:, n_half:] = L21 @ L22

    # Assemble S
    S = torch.zeros_like(A)
    S12_scaled = _safe_inv(lambda_factor**2, eps) * S12
    S[:n_half, :n_half] = S11
    S[:n_half, n_half:] = S12_scaled
    S[n_half:, :n_half] = S21
    S[n_half:, n_half:] = S22

    # Assemble U
    U = torch.zeros_like(A)
    U[:n_half, :n_half] = U21 @ U11
    U[:n_half, n_half:] = U2
    U[n_half:, n_half:] = U22 @ U_tilde_12

    # --- Calculate S_bar, S_hat, E, I, J, I_bar, J_bar ---
    S_ext = _extended_mapping(S, eps)
    # compute S_bar from S using the S_ext mapping relation
    S_bar = S_ext - S  # Complementary mapping (Def 3)
    S_hat = _safe_inv(alpha_r, eps) * ((alpha * S) + S_bar)

    # compute E using the unit mapping on S
    E = _unit_mapping(S, eps)  # Def 1
    I = E @ E.T
    J = E.T @ E
    Id_n = torch.eye(n, device=device, dtype=dtype)
    I_bar = Id_n - I
    J_bar = Id_n - J

    # --- Assemble final matrices M, W ---
    # These are complex derivations involving inverses and block manipulations.

    # compute inverse of I_lmbda_12 using _safe_inv since I and J belongs to group of diagonal matrices
    I_lambda_12_inv = torch.diag(_safe_inv(torch.diag(I_lambda_12), eps))  # Only works if diagonal

    L3_prime = L3 @ I_lambda_12_inv @ S_hat12 @ M12 @ S_hat11 @ M11
    U2_prime = W11 @ S_hat11 @ W21 @ S_hat21 @ U2

    EpSbT = (E + S_bar).T
    E_p_11, E_p_12 = EpSbT[:n_half, :n_half], EpSbT[:n_half, n_half:]
    E_p_21, E_p_22 = EpSbT[n_half:, :n_half], EpSbT[n_half:, n_half:]

    # S_p blocks - coefficients alpha, alpha_l, alpha_k, alpha_s
    # in the papers there is a mistake, that S_prime matrix has a accidental transpose in it
    S_p_11 = (alpha * E[:n_half, :n_half]) + (alpha * S_bar11)
    S_p_12 = (alpha_l * E[:n_half, n_half:]) + (alpha * S_bar12)
    S_p_21 = (alpha_k * E[n_half:, :n_half]) + (alpha * S_bar21)
    S_p_22 = (alpha_s * E[n_half:, n_half:]) + (alpha * S_bar22)

    # M assembly
    M_mat_b1_11 = E_p_11 @ S_bar12 @ M12 @ S_p_11 @ M11
    M_mat_b1_12 = E_p_12 @ S_bar22 @ M22 @ S_p_21 @ M21
    M_mat_b1_21 = E_p_21 @ S_p_12 @ M12 @ S_bar11 @ M11
    M_mat_b1_22 = E_p_22 @ S_p_22 @ M22 @ S_bar21 @ M21

    M_mat_b1 = torch.cat(
        (
            torch.cat([M_mat_b1_11, M_mat_b1_12], dim=1),
            torch.cat([M_mat_b1_21, M_mat_b1_22], dim=1),
        ),
        dim=0,
    )

    Id_n_half = torch.eye(n_half, device=device, dtype=dtype)

    M_mat_b2 = torch.cat(
        (
            torch.cat([(alpha_r * _safe_inv(alpha_m * alpha_k, eps)) * Id_n_half, torch.zeros_like(L3_prime)], dim=1),
            torch.cat([(-_safe_inv(alpha_l, eps)) * L3_prime, _safe_inv(alpha_l, eps) * Id_n_half.clone()], dim=1),
        ),
        dim=0,
    )

    M = (M_mat_b1 @ M_mat_b2) * _safe_inv(alpha, eps)

    # W assembly
    W_mat_b1 = torch.cat(
        (
            torch.cat(
                [(alpha_r * _safe_inv(alpha_k * alpha_l, eps)) * Id_n_half, (-_safe_inv(alpha_m, eps)) * U2_prime],
                dim=1,
            ),
            torch.cat([torch.zeros_like(U2_prime), _safe_inv(alpha_m, eps) * Id_n_half.clone()], dim=1),
        ),
        dim=0,
    )

    # W_mat_b2 block structure
    W_mat_b2_11 = W11 @ S_p_11 @ W21 @ S_bar21 @ E_p_11
    W_mat_b2_12 = W11 @ S_bar11 @ W21 @ S_p_21 @ E_p_12
    W_mat_b2_21 = W12 @ S_p_12 @ W22 @ S_bar22 @ E_p_21
    W_mat_b2_22 = W12 @ S_bar12 @ W22 @ S_p_22 @ E_p_22

    W_mat_b2 = torch.cat(
        (
            torch.cat([W_mat_b2_11, W_mat_b2_12], dim=1),
            torch.cat([W_mat_b2_21, W_mat_b2_22], dim=1),
        ),
        dim=0,
    )

    # Final W scaling, similar logic to M
    W = (W_mat_b1 @ W_mat_b2) * _safe_inv(alpha, eps)

    # --- Final Checks (Optional) ---
    if perform_checks:
        _lsu_checks(
            A=A,
            L=L,
            S=S,
            U=U,
            M=M,
            W=W,
            S_hat=S_hat,
            S_bar=S_bar,
            I=I,
            J=J,
            I_bar=I_bar,
            J_bar=J_bar,
            alpha=alpha,
            context_str=f"A-depth-{depth}",
            depth=depth,
            atol=check_atol,
        )

    return L, S, U, M, W, S_hat, S_bar, I, J, I_bar, J_bar, alpha_r


# --- Example Usage ---
if __name__ == "__main__":
    # Set default dtype and device
    torch.set_default_dtype(torch.float64)  # Use float64 for better precision as in example
    # Use CUDA if available, otherwise CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("--- Using CUDA device ---")
    else:
        device = torch.device("cpu")
        print("--- Using CPU device ---")

    # Example 1: Matrix from Section VIII of the paper
    A_paper = torch.tensor(
        [[0, 0, 3, 0], [2, 0, 1, 0], [0, 0, 0, 0], [1, 4, 0, 1]], dtype=torch.get_default_dtype(), device=device
    )

    print("\n--- Running LSU Factorization on Paper Example ---")
    print("Input Matrix A:")
    print(A_paper)

    # Run factorization
    # Set perform_checks=True for detailed step-by-step validation (slower)
    result_paper = lsu_factorization(A_paper, perform_checks=True)

    if result_paper:
        L_p, S_p, U_p, M_p, W_p, S_hat_p, S_bar_p, I_p, J_p, I_bar_p, J_bar_p, alpha_r_p = result_paper

        print(f"\n--- Factorization Complete ---")
        print(f"Final alpha_r = {alpha_r_p.item():.6g}")  # Expected: 24

        # --- Print Key Results ---
        print("\nMatrix L (Lower Triangular):")
        print(L_p)
        print("\nMatrix S (Weighted Permutation):")
        print(S_p)
        print("\nMatrix U (Upper Triangular):")
        print(U_p)
        print("\nMatrix M (Inverse Related):")
        print(M_p)
        print("\nMatrix W (Inverse Related):")
        print(W_p)
        print("\nMatrix S_hat (Scaled S + S_bar):")
        print(S_hat_p)

        # --- Verification Checks (Final Result) ---
        print("\n--- Final Verification ---")
        # 1. Check A = alpha * L @ S @ U (alpha is 1.0 for initial call)
        LSU = L_p @ S_p @ U_p
        print(f"Check 1: A == L @ S @ U : {torch.allclose(A_paper, LSU, atol=1e-8)}")
        if not torch.allclose(A_paper, LSU, atol=1e-8):
            print("   LSU Product:\n", LSU)
            print("   Original A:\n", A_paper)
            print("   Max Difference:", torch.max(torch.abs(A_paper - LSU)).item())

        # 2. Check Identity = L @ S_hat @ M
        LShatM = L_p @ S_hat_p @ M_p
        Id_n = torch.eye(A_paper.shape[0], device=device, dtype=torch.get_default_dtype())
        print(f"Check 2: Id == L @ S_hat @ M : {torch.allclose(Id_n, LShatM, atol=1e-8)}")
        if not torch.allclose(Id_n, LShatM, atol=1e-8):
            print("   LShatM Product:\n", LShatM)
            print("   Max Difference:", torch.max(torch.abs(Id_n - LShatM)).item())

        # 3. Check Identity = W @ S_hat @ U
        WShatU = W_p @ S_hat_p @ U_p
        print(f"Check 3: Id == W @ S_hat @ U : {torch.allclose(Id_n, WShatU, atol=1e-8)}")
        if not torch.allclose(Id_n, WShatU, atol=1e-8):
            print("   WShatU Product:\n", WShatU)
            print("   Max Difference:", torch.max(torch.abs(Id_n - WShatU)).item())

        # --- Check Final Properties ---
        _check_lsu_final_properties(
            A=A_paper,
            L=L_p,
            S=S_p,
            U=U_p,
            M=M_p,
            W=W_p,
            S_hat=S_hat_p,
            S_bar=S_bar_p,
            alpha_r=alpha_r_p,
        )

    else:
        print("\n--- Factorization Failed ---")

    # Example 2: A simple invertible matrix
    # A_inv = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
    # print("\n--- Running LSU Factorization on Invertible Example ---")
    # print("Input Matrix A:\n", A_inv)
    # result_inv = lsu_factorization(A_inv, perform_checks=False)
    # if result_inv:
    #      L_i, S_i, U_i, M_i, W_i, S_hat_i, _, _, _, _, _, alpha_r_i = result_inv
    #      print(f"\nFinal alpha_r = {alpha_r_i.item():.6g}") # Expected: det(A) = -2
    #      _check_lsu_final_properties(A=A_inv, L=L_i, S=S_i, U=U_i, M=M_i, W=W_i,
    #                                  S_hat=S_hat_i, alpha_r=alpha_r_i)
