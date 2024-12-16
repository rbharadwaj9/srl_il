import os
import h5py
import numpy as np
from PIL import Image
import sys

def extract_and_save_images(top_folder):
    """
    Extract the first image from the dataset "/observations/images/oakd_side_view/color" in each HDF5 file
    and save it as an image in a new folder named <top_folder>_images.

    Args:
        top_folder (str): Path to the top-level folder containing child folders with HDF5 files.
    """
    # Create the output folder
    output_folder = f"{top_folder}_images"
    os.makedirs(output_folder, exist_ok=True)

    # Traverse the directory tree
    for root, _, files in os.walk(top_folder):
        for file_name in files:
            if file_name.endswith(".h5"):
                h5_file_path = os.path.join(root, file_name)
                try:
                    with h5py.File(h5_file_path, 'r') as hdf:
                        # Access the dataset
                        dataset_path = "oakd_side_view/color"
                        if dataset_path in hdf:
                            dataset = hdf[dataset_path]

                            first_key = list(dataset.keys())[0]
                            # Extract the first image along the 0th dimension

                            image_data = np.array(dataset.get(first_key))

                            # Convert to uint8 if necessary
                            if image_data.dtype != np.uint8:
                                image_data = image_data.astype(np.uint8)

                            # Save the image
                            image = Image.fromarray(image_data)
                            output_image_path = os.path.join(
                                output_folder,
                                f"{os.path.basename(top_folder)}_{file_name.replace('.h5', '.png')}"
                            )
                            image.save(output_image_path)
                            print(f"Saved image: {output_image_path}")
                        else:
                            print(f"Dataset {dataset_path} not found in file {h5_file_path}.")
                except Exception as e:
                    print(f"Error processing file {h5_file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 script_name.py <top_folder>")
        sys.exit(1)

    # Get the top folder from the command-line argument
    top_folder = sys.argv[1]

    # Call the function to extract and save images
    extract_and_save_images(top_folder)
