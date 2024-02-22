import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from skimage import io
import cv2


def corner_detect(image, n_corners, smooth_std, window_size):
    """
    Detect corners on a given image

    Args:
        image: 2D grayscale image on which to detect corners
        n_corners: Total number of corners to be extracted
        smooth_std: Standard deviation of the Gaussian smoothing kernel
        window_size: Window size for Gaussian smoothing kernel,
                     corner detection, and nonmaximum suppresion

    Returns:
        minor_eig_image: The minor eigenvalue image (same shape as image)
        corners: Detected corners (in x-y coordinates) in a numpy array of shape (n_corners, 2)
    """
    # Smooth the image
    smoothed = gaussian_filter(image, sigma=smooth_std)

    # Compute gradients using Sobel operators
    k_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    k_y = k_x.T
    g_x = convolve2d(smoothed, k_x, mode='same', boundary='symm')
    g_y = convolve2d(smoothed, k_y, mode='same', boundary='symm')

    # Compute the squares of the gradients and their product
    I_x_2 = g_x**2
    I_y_2 = g_y**2
    I_x_y = g_x * g_y

    # Allocate the minor eigenvalue image
    minor_eig_image = np.zeros_like(image, dtype=float)

    # Compute the structure tensor for each pixel (or in a window around it) and its eigenvalues
    half_win = window_size // 2
    for i in range(half_win, image.shape[0] - half_win):
        for j in range(half_win, image.shape[1] - half_win):
            # Compute sums within the window
            S_xx = np.sum(I_x_2[i-half_win:i+half_win+1, j-half_win:j+half_win+1])
            S_yy = np.sum(I_y_2[i-half_win:i+half_win+1, j-half_win:j+half_win+1])
            S_xy = np.sum(I_x_y[i-half_win:i+half_win+1, j-half_win:j+half_win+1])
            
            # Structure tensor
            M = np.array([[S_xx, S_xy], [S_xy, S_yy]])
            
            # Eigenvalues
            eigenvalues = np.linalg.eigvalsh(M)
            minor_eig_image[i, j] = np.min(eigenvalues)


            
    def nonmax_suppression(image, window_size):
        padded = np.pad(image, ((window_size//2, window_size//2), (window_size//2, window_size//2)), mode='constant', constant_values=np.min(image))
        suppressed = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                local_window = padded[i:i+window_size, j:j+window_size]
                if image[i, j] == np.max(local_window):
                    suppressed[i, j] = image[i, j]
        return suppressed

    suppressed_image = nonmax_suppression(minor_eig_image, window_size)
    corner_indices = np.argpartition(-suppressed_image.ravel(), n_corners)[:n_corners]
    y, x = np.unravel_index(corner_indices, suppressed_image.shape)
    corners = np.column_stack((x, y))

    return minor_eig_image, corners

def show_image(img, title, color_BGR=False, corners=None):
    plt.figure(figsize=(10, 5))
    if color_BGR:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    
    if not corners is None:
        plt.scatter(corners[:, 0], corners[:, 1], s=25, edgecolors='r', facecolors='none')
    plt.title(title)
    plt.axis('off')
    plt.show()

def sift_feature_matching(imgs, kp1, kp2, threshold=1):
    sift = cv2.SIFT_create()


    _, des1 = sift.compute(imgs[0], kp1)
    _, des2 = sift.compute(imgs[1], kp2)

    # Match descriptors using Brute Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)

    result = cv2.drawMatches(imgs[0], kp1, imgs[1], kp2, good_matches[:20], None, flags = 2)

    return result, good_matches
    

def transform_images_homography(imgs, H):
    def calculate_size_and_offset(H, img1, img2):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        
        # Transform corners of img1 using H
        transformed_corners1 = cv2.perspectiveTransform(corners1, H)
        
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        
        all_corners = np.concatenate((transformed_corners1, corners2), axis=0)
        
        # Find min and max extents
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        # Translate everything so that min x, y are 0, 0
        translate_dist = [-x_min, -y_min]
        
        # Output width and height
        output_width = x_max - x_min
        output_height = y_max - y_min
        
        return output_width, output_height, translate_dist

    output_width, output_height, translate_dist = calculate_size_and_offset(H, imgs[0], imgs[1])

    # Translation matrix
    translate_H = np.array([[1, 0, translate_dist[0]], [0, 1, translate_dist[1]], [0, 0, 1]])

    # Warp img1 with the new dimensions and translation
    warped_img1 = cv2.warpPerspective(imgs[0], translate_H.dot(H), (output_width, output_height))

    output_image = warped_img1.copy()

    output_image[translate_dist[1]:translate_dist[1]+imgs[1].shape[0], translate_dist[0]:translate_dist[0]+imgs[1].shape[1]] = imgs[1]

    return output_image
    

def generate_panoramic(img_paths, n_corners=200, smooth_std=.1, window_size=7):
    imgs = []
    eig_imgs = []
    corners = []
    colored_imgs = []
    for img_path in img_paths:
        img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE)
        colored_img = cv2.imread(img_path)
        colored_img = cv2.resize(colored_img, (min(img.shape[0], 500), min(img.shape[1], 500)))
        img = cv2.resize(img, (min(img.shape[0], 500), min(img.shape[1], 500)))
        imgs.append(img)
        colored_imgs.append(colored_img)
        #img = io.imread('im' + str(i) + '.png')
        minor_eig_image, corners_vals = corner_detect(imgs[-1], n_corners, smooth_std, window_size)
        eig_imgs.append(minor_eig_image)
        corners.append(corners_vals)

    kp1 = [cv2.KeyPoint(x=float(corner[0]), y=float(corner[1]), size=20) for corner in corners[0]]
    kp2 = [cv2.KeyPoint(x=float(corner[0]), y=float(corner[1]), size=20) for corner in corners[1]]

    _, good_matches = sift_feature_matching(imgs, kp1, kp2)
    print(len(good_matches))

    # Estimate Homography matrix, using RANSAC for fundamental matrix for transformation
    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt


    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Transform two images into panoramic using estimated Homography matrix:
    panoramic = transform_images_homography(colored_imgs, H)

    return panoramic



if __name__ == '__main__':
    img1 = 'img1.png'
    img2 = 'img2.png'
    pano = generate_panoramic([img1, img2])
    
    # Save to panoramic.png
    plt.imsave('panoramic.png', cv2.cvtColor(pano, cv2.COLOR_BGR2RGB))





    


