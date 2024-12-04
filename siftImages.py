import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("Usage: python siftImages.py imagefile1 [imagefile2 imagefile3 ...]")
        sys.exit(1)
    image_files = sys.argv[1:]
    if len(image_files) == 1:
        # Task One
        task_one(image_files[0])
    else:
        # Task Two
        task_two(image_files)

def task_one(image_file):
    """
    Perform Task One: Detect and display SIFT keypoints on a single image.
    """
    # Load the image
    img = cv2.imread(image_file)
    if img is None:
        print(f"Error loading image {image_file}")
        sys.exit(1)
    # Rescale the image properly to VGA size (480x600 pixels)
    scaled_img = rescale_image(img, 480, 600)
    # Extract SIFT keypoints from the luminance Y component
    keypoints, descriptors = extract_sift_features(scaled_img)
    # Draw keypoints on the image
    img_with_keypoints = draw_keypoints(scaled_img.copy(), keypoints)
    # Display the original image and image with highlighted keypoints
    display_images(scaled_img, img_with_keypoints)
    # Output the number of detected keypoints
    print(f"# of keypoints in {image_file} is {len(keypoints)}")


def task_two(image_files):
    """
    Perform Task Two: Compare multiple images using a Bag-of-Words model constructed from SIFT descriptors.
    """
    keypoints_list = []       # List to store keypoints for each image
    descriptors_list = []     # List to store descriptors for each image
    num_keypoints_list = []   # List to store number of keypoints in each image
    total_keypoints = 0       # Total number of keypoints in all images
    images = []
    # For each image
    for image_file in image_files:
        # Load the image
        img = cv2.imread(image_file)
        if img is None:
            print(f"Error loading image {image_file}")
            sys.exit(1)
        images.append(img)
        # Rescale the image
        scaled_img = rescale_image(img, 480, 600)
        # Extract SIFT keypoints and descriptors
        keypoints, descriptors = extract_sift_features(scaled_img)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
        num_keypoints_list.append(len(keypoints))
        total_keypoints += len(keypoints)
        print(f"# of keypoints in {image_file} is {len(keypoints)}")
    print(f"Total # of keypoints of all images is {total_keypoints}")
    print()
    # Now, for K=5%, 10%, 20% of total number of keypoints
    K_percentages = [0.05, 0.10, 0.20]
    for K_percentage in K_percentages:
        K = int(np.ceil(K_percentage * total_keypoints))
        if K <= 0:
            K = 1
        # Cluster the descriptors into K clusters
        # We need to stack all descriptors into one array
        all_descriptors = np.vstack(descriptors_list)
        # Perform k-means clustering
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 100, 0.1)
        attempts = 10
        flags = cv2.KMEANS_PP_CENTERS
        # Ensure K does not exceed the number of descriptors
        if K > len(all_descriptors):
            K = len(all_descriptors)
        compactness, labels, centers = cv2.kmeans(all_descriptors.astype(np.float32), K, None, criteria, attempts, flags)
        # For each image, construct histogram of visual words
        start_idx = 0
        histograms = []
        for descriptors in descriptors_list:
            num_descriptors = descriptors.shape[0]
            labels_for_image = labels[start_idx:start_idx+num_descriptors]
            hist, _ = np.histogram(labels_for_image, bins=np.arange(K+1))
            histograms.append(hist)
            start_idx += num_descriptors
        # Normalize histograms
        histograms = [hist / np.sum(hist) for hist in histograms]
        # Calculate X^2 distance between histograms
        num_images = len(image_files)
        dissimilarity_matrix = np.zeros((num_images, num_images))
        for i in range(num_images):
            for j in range(i, num_images):
                chi_square = chi_square_distance(histograms[i], histograms[j])
                dissimilarity_matrix[i, j] = chi_square
                dissimilarity_matrix[j, i] = chi_square  # Since the matrix is symmetric
        # Output the dissimilarity matrix
        print_dissimilarity_matrix(dissimilarity_matrix, image_files, K, total_keypoints, K_percentage)

def rescale_image(img, max_height, max_width):
    """
    Rescale the image properly to a size comparable to VGA size (480x600), keeping aspect ratio.
    Applies a low-pass filter to reduce artifacts.
    """
    # Get current dimensions
    height, width = img.shape[:2]
    # Compute scaling factors
    height_scale = max_height / height
    width_scale = max_width / width
    # Use the smaller scaling factor to keep aspect ratio
    scale = min(height_scale, width_scale)
    new_width = int(width * scale)
    new_height = int(height * scale)
    # Determine if scaling up or scaling down
    if scale < 1.0:
        # Scaling down
        # Adjusted kernel size calculation
        kernel_size = int(2 * np.log(1 / scale) + 1)
        # Ensure kernel size is odd and at least 3, and limit to a maximum value
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size > 7:
            kernel_size = 7
        # Apply Gaussian blur to the image
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        # Use INTER_AREA interpolation for shrinking
        interpolation_method = cv2.INTER_AREA
    elif scale > 1.0:
        # Scaling up
        # Use INTER_CUBIC interpolation for enlarging
        interpolation_method = cv2.INTER_CUBIC
    else:
        # Scale == 1.0, no resizing needed
        return img
    # Rescale the image using the appropriate interpolation method
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=interpolation_method)
    return resized_img

def extract_sift_features(img):
    """
    Extract SIFT keypoints and descriptors from the luminance Y component of the image.
    """
    # Convert image to YCrCb color space
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # Extract Y component
    Y_channel = ycrcb[:, :, 0]
    # Create SIFT detector
    sift = cv2.SIFT_create()
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(Y_channel, None)
    return keypoints, descriptors

def draw_keypoints(img, keypoints):
    """
    Draw keypoints on the image.
    For each keypoint, draw a cross '+' at the location, a circle around it proportional to its scale,
    and a line indicating its orientation.
    """
    for kp in keypoints:
        x, y = kp.pt
        x = int(round(x))
        y = int(round(y))
        size = kp.size
        angle = kp.angle
        # Draw a cross '+'
        color = (0, 255, 0)  # Green color
        length = 5  # Length of the cross arms
        cv2.line(img, (x - length, y), (x + length, y), color, 1)
        cv2.line(img, (x, y - length), (x, y + length), color, 1)
        # Draw a circle around the keypoint
        radius = int(round(size / 2))
        cv2.circle(img, (x, y), radius, color, 1)
        # Draw a line indicating the orientation from '+' to the circle
        angle_rad = np.radians(angle)
        dx = int(round(radius * np.cos(angle_rad)))
        dy = int(round(radius * np.sin(angle_rad)))
        cv2.line(img, (x, y), (x + dx, y + dy), color, 1)
    return img

def display_images(img1, img2):
    """
    Display the original image and image with highlighted keypoints side by side using Matplotlib without margins.
    The width of the viewing window is constrained to be between 1280 and 1680 pixels.
    """
    # Ensure both images have the same height
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if h1 != h2:
        # Resize img2 to match img1's height while maintaining aspect ratio
        new_width = int(w2 * h1 / h2)
        img2 = cv2.resize(img2, (new_width, h1), interpolation=cv2.INTER_AREA)
    # Convert images from BGR to RGB
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # Concatenate images horizontally
    combined_img = np.hstack((img1_rgb, img2_rgb))
    # Get the dimensions of the combined image
    height, width = combined_img.shape[:2]
    # Ensure the width of the viewing window is between 1280 and 1680 pixels
    desired_min_width = 1280
    desired_max_width = 1680
    if width < desired_min_width:
        # Upscale the combined image to have width = desired_min_width
        scale_factor = desired_min_width / width
        new_width = desired_min_width
        new_height = int(height * scale_factor)
        combined_img = cv2.resize(combined_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        height, width = combined_img.shape[:2]
    elif width > desired_max_width:
        # Downscale the combined image to have width = desired_max_width
        scale_factor = desired_max_width / width
        new_width = desired_max_width
        new_height = int(height * scale_factor)
        combined_img = cv2.resize(combined_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        height, width = combined_img.shape[:2]
    # Calculate figure size to match the image size exactly
    dpi = 100  # Dots per inch
    fig_size = (width / dpi, height / dpi)
    # Create a figure without margins or padding
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # Left, bottom, width, height (0 to 1)
    ax.axis('off')
    ax.imshow(combined_img)
    plt.show()

def chi_square_distance(hist1, hist2):
    """
    Calculate the chi-squared distance between two histograms.
    """
    eps = 1e-10  # small value to prevent division by zero
    chi_sq = 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + eps))
    return chi_sq

def print_dissimilarity_matrix(matrix, image_files, K, total_keypoints, K_percentage):
    """
    Print the dissimilarity matrix in a readable format with aligned columns.
    """
    num_images = len(image_files)
    percentage = int(K_percentage * 100)
    print(f"K={percentage}%*{total_keypoints}={K}")
    print("Dissimilarity Matrix")
    # Prepare headers
    image_names = [get_image_name(f) for f in image_files]
    # Determine the maximum width needed for image names and data
    max_name_width = max(len(name) for name in image_names + ["Dissimilarity Matrix"])
    col_width = 13  # Add padding
    # Create header row
    header_row = ' ' * (max_name_width + 2)  # Space for row labels
    for name in image_names:
        header_row += f"{name:<{col_width}}"
    print(header_row)
    # Print each row
    for i in range(num_images):
        row_label = f"{image_names[i]:<{max_name_width}}  "
        row = row_label
        for j in range(num_images):
            if i == j:
                value = 0.0
            else:
                value = matrix[i, j]
            row += f"{value:<{col_width}.4f}"
        print(row)
    print()

def get_image_name(filepath):
    """
    Extract the image file name from the file path.
    """
    # Replace backslashes with forward slashes
    filepath = filepath.replace('\\', '/')
    # Split the path and get the last part
    filename = filepath.rsplit('/', 1)[-1]
    return filename

if __name__ == "__main__":
    main()
