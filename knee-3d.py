import os
import numpy as np
import pydicom
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.measure import marching_cubes

# Define the input directory containing knee DICOM images
input_dir = './Knee_dataset/series-00002/'

# Get a list of all files in the input directory
dicom_files = [file for file in os.listdir(input_dir) if file.endswith('.dcm')]

# Define the spacing between slices
slice_spacing = 5  # You can adjust this value based on your preference and dataset characteristics

# Process each knee image and store the segmented images with spacing in the list
segmented_knee_images = []
for file_name in dicom_files:
    # Load the DICOM image
    dicom_data = pydicom.dcmread(os.path.join(input_dir, file_name))
    knee_img = dicom_data.pixel_array

    # Segment the knee image
    threshold_value = threshold_otsu(knee_img)
    knee_mask = knee_img > threshold_value
    labeled_mask = label(knee_mask)
    regions = regionprops(labeled_mask)
    max_area = 0
    knee_area = None
    for region in regions:
        if region.area > max_area:
            max_area = region.area
            knee_area = region
    knee_mask = np.zeros_like(labeled_mask)
    knee_mask[labeled_mask == knee_area.label] = 1
    segmented_knee_img = knee_img * knee_mask

    # Add empty slices as spacing
    for _ in range(slice_spacing):
        segmented_knee_images.append(np.zeros_like(segmented_knee_img))

    # Append the segmented knee image to the list
    segmented_knee_images.append(segmented_knee_img)

# Stack segmented knee images into a 3D volume
stacked_volume = np.stack(segmented_knee_images, axis=-1)

# Apply marching cubes algorithm to extract mesh vertices and faces
vertices, faces, _, _ = marching_cubes(stacked_volume, level=0.3, spacing=(0.5, 0.5, 0.5))

# Define a function to save the mesh as an OBJ file
def save_mesh(vertices, faces, filename):
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")  # Add 1 to each index to match OBJ format

# Specify the output directory and filename for the OBJ mesh
obj_output_dir = './MeshOutput/'
os.makedirs(obj_output_dir, exist_ok=True)
obj_filename = os.path.join(obj_output_dir, 'knee_mesh2.obj')

# Save the mesh as an OBJ file
save_mesh(vertices, faces, obj_filename)
