import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

def segment_knee_image(knee_img):
    # Apply Otsu's thresholding to segment knee
    threshold_value = threshold_otsu(knee_img)
    knee_mask = knee_img > threshold_value

    # Label connected components
    labeled_mask = label(knee_mask)

    # Get properties of connected components
    regions = regionprops(labeled_mask)

    # Find the largest connected component (knee area)
    knee_area = None
    max_area = 0
    for region in regions:
        if region.area > max_area:
            max_area = region.area
            knee_area = region

    # Create a binary mask for the largest connected component (knee)
    knee_mask = np.zeros_like(labeled_mask)
    knee_mask[labeled_mask == knee_area.label] = 1

    # Remove everything else in the image outside the knee mask
    segmented_knee_img = knee_img * knee_mask

    return segmented_knee_img

# Path to the directory containing knee DICOM images
input_dir = './Knee_dataset/series-00000/'
output_dir = './KneeOutput/'

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Get a list of all files in the input directory
dicom_files = [file for file in os.listdir(input_dir) if file.endswith('.dcm')]

# Process each knee image
for file_name in dicom_files:
    # Load the DICOM image
    dicom_data = pydicom.dcmread(os.path.join(input_dir, file_name))
    knee_img = dicom_data.pixel_array

    # Segment the knee image
    segmented_knee_img = segment_knee_image(knee_img)

    # Save the segmented knee image
    output_file_path = os.path.join(output_dir, f'segmented_knee_{file_name.replace(".dcm", ".png")}')
    plt.imsave(output_file_path, segmented_knee_img, cmap='gray')

    # You can add more features or processing steps here using segmented_knee_img variable
    # For example, calculate statistics or perform further processing on the segmented knee region.
    # Additional code for more features can be added in this loop.
