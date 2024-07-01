import cv2
import numpy as np
import pandas as pd
import os
import sys
from glob import glob
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from datetime import datetime  # Import datetime module

def process_images(image_folder):
    # Load images
    X_paths = sorted(glob(os.path.join(image_folder, '*.jpg')))
    X = [cv2.imread(path) for path in X_paths]

    # Normalize images
    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
    axis_norm = (0, 1)   # normalize channels independently
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))

    model = StarDist2D(None, name='grayscale_paramecium', basedir='models')

    # Process each image
    shape_data_list = []
    centroid_data_list = []
    cell_counts = []

    for idx, img in enumerate(X):
        frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        # Normalize the grayscale image
        img_norm = normalize(frame, 1, 99.8, axis=axis_norm)
        
        # Predict instances using the StarDist model
        labels, details = model.predict_instances(img_norm)

        # Convert the 'labels' image to uint8 for contour extraction
        labels_uint8 = labels.astype(np.uint8)
        
        # Find contours in the 'labels' image
        contours, _ = cv2.findContours(labels_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to grayscale

        # Prepare data structures
        shape_data = []
        centroid_data = []
        id_counter = 0

        # Process each contour found
        for contour in contours:
            # Calculate the bounding box for each contour
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate the centroid of the bounding box
            centroid_x = x + w // 2
            centroid_y = y + h // 2

            # Append contour coordinates to list
            for point in contour:
                xC, yC = point[0]
                shape_data.append({"ID": id_counter, "x": xC, "y": yC, "image_name": os.path.basename(X_paths[idx])})

            # Append centroid coordinates to list
            centroid_data.append({"ID": id_counter, "x": centroid_x, "y": centroid_y, "image_name": os.path.basename(X_paths[idx])})

            id_counter += 1

        # Append lists to main lists
        shape_data_list.extend(shape_data)
        centroid_data_list.extend(centroid_data)

        # Count centroids and store in cell_counts
        cell_counts.append(len(centroid_data))

        # Draw contours and centroids on the original image
        for cnt in contours:
            cv2.drawContours(img, [cnt], 0, (0, 0, 0), 2)
            cv2.drawContours(img, [cnt], 0, (0, 165, 255), 1)
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.circle(img, (cx, cy), 3, (0, 0, 0), cv2.FILLED)
            cv2.circle(img, (cx, cy), 2, (0, 0, 255), cv2.FILLED)

        # Save annotated image with a timestamp prefix
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"output_{now}_{os.path.splitext(os.path.basename(X_paths[idx]))[0]}.jpg"
        cv2.imwrite(output_filename, img)

    # Convert lists to DataFrames
    shape_data_df = pd.DataFrame(shape_data_list)
    centroid_data_df = pd.DataFrame(centroid_data_list)

    # Save shape_data and centroid_data DataFrames to CSV files with timestamp
    shape_data_filename = f"shape_data_{now}.csv"
    centroid_data_filename = f"centroid_data_{now}.csv"
    shape_data_df.to_csv(shape_data_filename, index=False)
    centroid_data_df.to_csv(centroid_data_filename, index=False)

    # Print cell counts per image and compute average
    print("\nCell Counts per Image:")
    for i, count in enumerate(cell_counts):
        print(f"{os.path.basename(X_paths[i])}: {count} cells")
    average_count = sum(cell_counts) / len(cell_counts)
    print(f"\nAverage Cell Count: {average_count:.2f}")

    print(f"\nProcessing complete. Data saved to {shape_data_filename} and {centroid_data_filename}.")

def main():
    if len(sys.argv) != 2:
        print("Usage: python count_cells.py <image_folder>, e.g. python count_cells.py ./images/blurry_output/")
        sys.exit(1)

    image_folder = sys.argv[1]
    if not os.path.isdir(image_folder):
        print(f"Error: {image_folder} is not a valid directory.")
        sys.exit(1)

    process_images(image_folder)

if __name__ == "__main__":
    main()
