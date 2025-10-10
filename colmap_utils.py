import pycolmap
import numpy as np

def load_colmap_model(sparse_model_path):
    """
    Loads a COLMAP sparse reconstruction model from a directory.

    Args:
        sparse_model_path (str): The path to the directory containing the
                                 COLMAP sparse model files (cameras.bin,
                                 images.bin, points3D.bin).

    Returns:
        pycolmap.Reconstruction: A pycolmap Reconstruction object containing
                                 all cameras, images, and 3D points.
                                 Returns None if the path is invalid.
    """
    try:
        model = pycolmap.Reconstruction(sparse_model_path)
        print("Successfully loaded COLMAP model.")
        print(f"  - Contains {len(model.cameras)} cameras.")
        print(f"  - Contains {len(model.images)} images.")
        return model
    except Exception as e:
        print(f"Error loading COLMAP model from '{sparse_model_path}': {e}")
        return None