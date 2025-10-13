# -*- coding: utf-8 -*-
"""
Main script to run the visual odometry experiment using COLMAP data.

This script performs the following steps:
1.  Reads the COLMAP reconstruction using the `pycolmap` library.
2.  Iterates through pairs of consecutive registered images.
3.  For each pair:
    a. Matches features using the specified algorithm.
    b. Calculates the ground truth relative rotation and direction of motion
       from the COLMAP poses.
    c. Executes the simplified visual odometry algorithm.
    d. Compares the estimated direction with the ground truth and prints the error.
"""
import os
import numpy as np
import time
import pdb
from colmap_utils import load_colmap_model
from visual_odometry import match_features, calculate_vo_initial_guess, refine_direction_mle
import matplotlib.pyplot as plt

def main():
    # Configuration
    colmap_project_path = './colmap-251010'
    images_path = os.path.join(colmap_project_path, 'images')
    sparse_model_path = os.path.join(colmap_project_path, 'sparse/0')

    print("Loading COLMAP data using pycolmap")
    model = load_colmap_model(sparse_model_path)
    if model is None:
        print("Failed to load model. Exiting.")
        return

    # Sort images by name for constant testing
    sorted_image_ids = sorted(model.images.keys(), key=lambda id: model.images[id].name)
    print(f"Found {len(sorted_image_ids)} images to process.")

    # Get Camera Intrinsics
    img_id = sorted_image_ids[0]
    image = model.images[img_id]
    cam_id = image.camera_id
    cam = model.cameras[cam_id]
    params = cam.params

    C = np.array([[params[0], 0,         params[2]], 
                  [0,         params[1], params[3]], 
                  [0,         0,         1]])

    print("Camera Intrinsics (C):\n", C)
    # pdb.set_trace()

    print("\nProcessing Image Pairs")

    det_method = 'orb'
    angular_errors_initial = []
    angular_errors_final = []
    calc_times = []
    ground_truth = []
    estimated = []
    covariances = []
    for i in range(len(sorted_image_ids) - 1):
        start_time = time.time()
        img_id1 = sorted_image_ids[i]
        img_id2 = sorted_image_ids[i+1]
        image1 = model.images[img_id1]
        image2 = model.images[img_id2]

        print(f"\nProcessing pair: '{image1.name}' and '{image2.name}'")

        # Match features between the two images
        img1_path = os.path.join(images_path, image1.name)
        img2_path = os.path.join(images_path, image2.name)

        # Calculate Camera Poses
        rigid1 = image1.cam_from_world() # Rigid3D
        rigid2 = image2.cam_from_world() # Rigid3D

        # R is the rotation matrix from world to camera
        R1 = rigid1.rotation.matrix()
        R2 = rigid2.rotation.matrix()

        # t is the translation vector from world to camera
        t1 = rigid1.translation
        t2 = rigid2.translation

        # The relative rotation from frame C_{k-1} to C_k is R_k * R_{k-1}^T
        # This is our M matrix for the algorithm.
        M_Ck_Ck_minus_1 = R2 @ R1.T

        p1, p2 = match_features(img1_path, img2_path, C, M_Ck_Ck_minus_1, det_method=det_method)

        if len(p1) < 10:
            print("Not enough matches found. Skipping pair.")
            continue

        # Execute the Visual Odometry Algorithm
        s_prime_initial = calculate_vo_initial_guess(p1, p2, C, M_Ck_Ck_minus_1)
        s_prime_est, R_s_prime = refine_direction_mle(p1, p2, C, M_Ck_Ck_minus_1, s_prime_initial)

        estimated.append(s_prime_est)
        covariances.append(R_s_prime)

        end_time = time.time()
        calc_times.append(end_time - start_time)
        
        # Ground Truth Direction of Motion
        # The camera center in world coordinates is calculated as: C = -R^T * t
        cam_center1_world = -R1.T @ t1
        cam_center2_world = -R2.T @ t2

        # The translation vector (s) in world coordinates is the difference in camera centers
        s_world = cam_center2_world - cam_center1_world
        
        # Transform the world displacement vector into the frame of the second camera (C_k) to get the true direction of motion.
        s_prime_true = (R2 @ s_world) / np.linalg.norm(R2 @ s_world)
        s_prime_true = s_prime_true.flatten()
        ground_truth.append(s_prime_true)

        # Calculate Angular Error
        angular_error_initial_rad = np.arccos(np.clip(np.dot(s_prime_initial, s_prime_true), -1.0, 1.0))
        angular_error_initial_deg = np.rad2deg(angular_error_initial_rad)
        angular_errors_initial.append(angular_error_initial_deg)

        angular_error_final_rad = np.arccos(np.clip(np.dot(s_prime_est, s_prime_true), -1.0, 1.0))
        angular_error_final_deg = np.rad2deg(angular_error_final_rad)
        angular_errors_final.append(angular_error_final_deg)

        print(f"  > True Direction (s'): {np.round(s_prime_true, 4)}")
        print(f"  > Initial Guess (s'): {np.round(s_prime_initial, 4)}")
        print(f"  > Est. Direction (s'): {np.round(s_prime_est, 4)}")
        print(f"  > Covariance (R_s'):\n {R_s_prime}")
        print(f"  > Angular Error (Initial): {angular_error_initial_deg:.4f} degrees")
        print(f"  > Angular Error (Final): {angular_error_final_deg:.4f} degrees")
        print(f"  > Calculation Time: {end_time - start_time:.2f} seconds\n")

    print(f"\nAverage Angular Error (Initial): {np.mean(angular_errors_initial):.4f} degrees")
    print(f"Standard Deviation: {np.std(angular_errors_initial):.4f} degrees")
    print(f"Average Angular Error (Final): {np.mean(angular_errors_final):.4f} degrees")
    print(f"Standard Deviation: {np.std(angular_errors_final):.4f} degrees")
    print(f"Average Calculation Time: {np.mean(calc_times):.4f} seconds")
    print(f"Standard Deviation of Calculation Time: {np.std(calc_times):.4f} seconds")


if __name__ == '__main__':
    main()