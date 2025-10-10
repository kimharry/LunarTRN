import os
import numpy as np
from colmap_utils import load_colmap_model


def main():
    colmap_project_path = './colmap-251010'
    sparse_model_path = os.path.join(colmap_project_path, 'sparse/0')
    model = load_colmap_model(sparse_model_path)
    if model is None:
        print("Failed to load model. Exiting.")
        return

    # Sort images by name for constant testing
    sorted_image_ids = sorted(model.images.keys(), key=lambda id: model.images[id].name)
    print(f"Found {len(sorted_image_ids)} images to process.")

    C = []
    for i in range(len(sorted_image_ids)):
        image_id = sorted_image_ids[i]
        image = model.images[image_id]
        cam = model.cameras[image.camera_id]
        C.append(cam.params)
    C = np.array(C)
    np.savetxt('camera_params_info.txt', C, fmt='%s')

if __name__ == '__main__':
    main()