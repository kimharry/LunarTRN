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

def match_features(img1_path, img2_path, det_method='orb'):
    """
    Detects and matches features between two images using SURF.

    Args:
        img1_path (str): Path to the first image.
        img2_path (str): Path to the second image.

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

    print(f"Found {len(good_matches)} good matches.")
    # plot the matches
    # img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow('Matches', img_matches)
    # # pdb.set_trace()
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()

    return p1, p2

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