import cv2
import numpy as np
import pdb

def skew(v):
    """
    Converts a 3-element vector to a 3x3 skew-symmetric matrix.
    This is the matrix representation of the cross product.
    [v_x] u = v x u
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def sampson_distance(p1, p2, F_prime):
    """
    Calculates the Sampson distance for a single point correspondence.
    This implements Equation (83) from the paper.
    """
    # Convert points to homogeneous coordinates
    u_k_minus_1 = np.array([p1[0], p1[1], 1])
    u_k = np.array([p2[0], p2[1], 1])
    
    # S matrix from Eq (84) / (B2)
    S = np.array([[1, 0, 0], [0, 1, 0]])

    # Epipolar constraint residual (numerator)
    numerator = (u_k_minus_1.T @ F_prime @ u_k)**2

    # Denominator terms
    term1 = S @ F_prime @ u_k
    term2 = S @ F_prime.T @ u_k_minus_1
    denominator = np.linalg.norm(term1)**2 + np.linalg.norm(term2)**2

    if denominator == 0:
        return np.inf

    return numerator / denominator


def RANSAC(p1, p2, C, M_Ck_Ck_minus_1, num_iterations=100, threshold=3.0, min_matches=6):
    """
    Filters a set of putative matches using the RANSAC algorithm.
    """
    best_inliers_indices = []
    if len(p1) < min_matches:
        return np.array([]), np.array([])
        
    for _ in range(num_iterations):
        # 1. Randomly sample a minimal set of correspondences
        indices = np.random.choice(len(p1), min_matches, replace=False)
        p1_sample, p2_sample = p1[indices], p2[indices]
        
        # 2. Estimate a model (initial guess for s') from the sample
        s_prime_candidate = calculate_vo_initial_guess(p1_sample, p2_sample, C, M_Ck_Ck_minus_1)
        if np.all(s_prime_candidate == 0): continue
        
        # Form the nondimensionalized fundamental matrix F' (Eq. 44)
        F_prime_candidate = np.linalg.inv(C).T @ M_Ck_Ck_minus_1.T @ skew(s_prime_candidate) @ np.linalg.inv(C)
        
        # 3. Find inliers by checking all points against the model
        current_inliers_indices = []
        for i in range(len(p1)):
            dist = sampson_distance(p1[i], p2[i], F_prime_candidate)
            if dist < threshold:
                current_inliers_indices.append(i)
        
        # 4. Update the best model if the current one has more inliers
        if len(current_inliers_indices) > len(best_inliers_indices):
            best_inliers_indices = current_inliers_indices

    p1_inliers = p1[best_inliers_indices]
    p2_inliers = p2[best_inliers_indices]
    
    return p1_inliers, p2_inliers


def match_features(img1_path, img2_path, C, M_Ck_Ck_minus_1, det_method='orb', num_iterations=100, threshold=3.0, min_matches=6):
    """
    Detects and matches features between two images using SURF.

    Args:
        img1_path (str): Path to the first image.
        img2_path (str): Path to the second image.
        C (np.ndarray): 3x3 camera calibration (intrinsics) matrix.
        M_Ck_Ck_minus_1 (np.ndarray): 3x3 rotation matrix from camera frame k-1 to k.
        det_method (str): Feature detector method ('sift' or 'orb').
        num_iterations (int): Number of iterations for RANSAC.
        threshold (float): Sampson distance threshold for inlier selection.
        min_matches (int): Minimum number of matches required for RANSAC.

    Returns:
        tuple: A tuple containing two numpy arrays (p1, p2) of corresponding
               feature points in the first and second image, respectively.
    """
    # Load images in grayscale for feature detection
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load one or both images. Check paths.")

    # Initialize SIFT detector.
    if det_method == 'sift':
        detector = cv2.SIFT_create()
    elif det_method == 'orb':
        detector = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE)
    else:
        raise ValueError("Invalid detector specified. Use 'sift' or 'orb'.")

    # Find keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # Use BFMatcher with KNN to find the two best matches for each descriptor
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    # Check if matches is not None and is iterable
    if matches is not None:
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    # Extract the coordinates of the good matches
    p1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    p2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    # RANSAC
    p1_inliers, p2_inliers = RANSAC(p1, p2, C, M_Ck_Ck_minus_1, num_iterations=num_iterations, threshold=threshold, min_matches=min_matches)

    print(f"Found {len(p1_inliers)} inliers after RANSAC among {len(p1)} initial matches.")
    # plot the inliers
    # img_matches = cv2.drawMatches(img1, kp1, img2, kp2, p1_inliers, p2_inliers, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow('Matches', img_matches)
    # # pdb.set_trace()
    # cv2.waitKey(3000)
    # cv2.destroyAllWindows()

    return p1_inliers, p2_inliers

def calculate_vo_initial_guess(matched_points1, matched_points2, C, M_Ck_Ck_minus_1):
    """
    Implements a simplified version of Algorithm 1 (up to step 9) from the paper
    to calculate the initial biased guess for the direction of motion.

    This function uses a direct least-squares method. The paper cautions that this
    produces a biased estimate and recommends the full algorithm for accuracy.

    Args:
        matched_points1 (np.ndarray): Nx2 array of keypoints from the first image.
        matched_points2 (np.ndarray): Nx2 array of keypoints from the second image.
        C (np.ndarray): 3x3 camera calibration (intrinsics) matrix.
        M_Ck_Ck_minus_1 (np.ndarray): 3x3 rotation matrix from camera frame k-1 to k.

    Returns:
        np.ndarray: The estimated 3x1 direction-of-motion unit vector (initial guess).
    """
    # Step 2 - Compute C inverse
    C_inv = np.linalg.inv(C)
    n_points = len(matched_points1)

    # Convert 2D pixel coordinates to 3D homogeneous coordinates
    u_k_minus_1_h = np.hstack([matched_points1, np.ones((n_points, 1))])
    u_k_h = np.hstack([matched_points2, np.ones((n_points, 1))])

    # Pre-compute transformed coordinates
    C_inv_u_k = (C_inv @ u_k_h.T).T

    # Step 3-6 (Simplified loop and matrix formation)
    # We only need to compute Gamma_i for the initial guess.
    gamma_matrices = []
    for i in range(n_points):
        # Current point coordinates
        u_k = C_inv_u_k[i]

        # Equation (53): h_i^T
        # This equation represents a single row of the H matrix in Eq (42)
        h_i_T = u_k_minus_1_h[i].T @ C_inv.T @ M_Ck_Ck_minus_1.T @ skew(u_k)

        # Equation (69): Gamma_i = h_i * h_i^T
        gamma_i = np.outer(h_i_T.T, h_i_T)
        gamma_matrices.append(gamma_i)

    # Step 7 (Finalizing H^T * H)
    # According to Eq (41) and (42), the matrix H is formed by stacking h_i^T rows.
    # Therefore, H^T * H is the sum of the outer products of its rows (h_i * h_i^T),
    # which is the sum of our Gamma matrices.
    H_T_H = np.sum(gamma_matrices, axis=0)

    # Step 8-9 (SVD to find the solution in the null space)
    # The direction of motion s' is the vector that minimizes H*s', which lies in
    # the null space of H. This corresponds to the eigenvector associated with the
    # smallest eigenvalue of H^T*H. We find this using SVD.
    # Equation (47): SVD of H (or H^T*H)
    _, _, V_h_T = np.linalg.svd(H_T_H)
    s_prime = V_h_T[-1, :]

    return s_prime


def cheirality_test(s_prime, p1, p2, C, M_Ck_Ck_minus_1):
    """
    Resolves the sign ambiguity of s' by ensuring landmarks are in front of the camera.
    This implements the triangulation method described in Appendix A.
    """
    C_inv = np.linalg.inv(C)
    
    u_k_minus_1_h = np.array([p1[0], p1[1], 1])
    u_k_h = np.array([p2[0], p2[1], 1])
    
    x_k_minus_1 = C_inv @ u_k_minus_1_h
    x_k = C_inv @ u_k_h

    # Construct the linear system from Equation (A8)
    A = np.vstack([skew(x_k), skew(M_Ck_Ck_minus_1 @ x_k_minus_1)])
    b = np.hstack([np.zeros(3), -skew(M_Ck_Ck_minus_1 @ x_k_minus_1) @ s_prime])

    l_prime_k, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # Check if the z-component of the triangulated point is positive.
    if l_prime_k[2] < 0:
        return -s_prime
    else:
        return s_prime


def refine_direction_mle(p1, p2, C, M_Ck_Ck_minus_1, s_prime_initial, sigma_uv=1.0, max_iter=5):
    """
    Refines the direction of motion using the unbiased Maximum Likelihood Estimator.
    This implements the iterative part of Algorithm 1 (steps 12-19).
    """
    s_prime_current = s_prime_initial
    n_points = len(p1)
    
    if n_points < 2:
        return s_prime_current

    C_inv = np.linalg.inv(C)
    s_prime = s_prime_initial
    n_points = len(p1)
    
    u_k_minus_1_h = np.hstack([p1, np.ones((n_points, 1))])
    u_k_h = np.hstack([p2, np.ones((n_points, 1))])
    
    C_inv_u_k_minus_1 = (C_inv @ u_k_minus_1_h.T).T
    C_inv_u_k = (C_inv @ u_k_h.T).T
    
    S = np.array([[1., 0., 0.], [0., 1., 0.]])
    R_u = (sigma_uv**2) * (S.T @ S) # Eq (52) & (B8)

    
    gamma_list = []
    xi_list = []
    for i in range(n_points):
        # Calculate Gamma_i from Eq (69)
        h_i_T = u_k_minus_1_h[i].T @ C_inv.T @ M_Ck_Ck_minus_1.T @ skew(C_inv_u_k[i]) # Eq (53)
        Gamma_i = np.outer(h_i_T.T, h_i_T)
        gamma_list.append(Gamma_i)
        
        # Calculate Xi_i from Eq (70)
        dh_du_k_minus_1 = -skew(C_inv_u_k[i]) @ M_Ck_Ck_minus_1 @ C_inv # Eq (58)
        dh_du_k = skew(M_Ck_Ck_minus_1 @ C_inv_u_k_minus_1[i]) @ C_inv # Eq (59)
        Xi_i = (dh_du_k_minus_1 @ R_u @ dh_du_k_minus_1.T) + (dh_du_k @ R_u @ dh_du_k.T)
        xi_list.append(Xi_i)

    delta_s = np.inf
    i_iter = 0
    s_prime_previous = np.copy(s_prime_current)

    while i_iter < max_iter and delta_s > 1e-6:
        F_s_prime_sum_term = np.zeros((3, 3))
        X_second_term_sum = np.zeros((3, 3))

        for i in range(n_points):
            Gamma_i = gamma_list[i]
            Xi_i = xi_list[i]
            
            den_s_Xi_s = s_prime_current.T @ Xi_i @ s_prime_current
            if den_s_Xi_s < 1e-9: continue

            # Term for Fisher Information Matrix (Eq. 74)
            F_s_prime_sum_term += Gamma_i / den_s_Xi_s
            
            # Term for bias correction (second part of Eq. 77)
            num_s_Gamma_s = s_prime_current.T @ Gamma_i @ s_prime_current
            X_second_term_sum += (num_s_Gamma_s / (den_s_Xi_s**2)) * Xi_i

        # Construct matrix X from Eq (77)
        X_matrix = F_s_prime_sum_term - X_second_term_sum
        
        # Solve for new s' using SVD
        U, D, Vt = np.linalg.svd(X_matrix)
        s_prime_current = Vt[-1, :]

        # Check for convergence
        delta_s = np.linalg.norm(s_prime_current - s_prime_previous)
        s_prime_previous = np.copy(s_prime_current)
        i_iter += 1
    
    # Construct pseudoinverse of D, setting the smallest singular value to 0
    D_inv = np.zeros((3, 3))
    D_inv[0, 0] = 1 / D[0] if D[0] > 1e-9 else 0
    D_inv[1, 1] = 1 / D[1] if D[1] > 1e-9 else 0
    
    # Reconstruct the covariance matrix R_s' from Eq (82)
    R_s_prime = Vt.T @ D_inv @ U.T
        
    return s_prime, R_s_prime