import pdb
import glob
import cv2
import os
import numpy as np
import random

class PanaromaStitcher():
    def __init__(self):
        # SIFT for feature detection and matching
        self.sift = cv2.SIFT_create()
        # FLANN parameters for fast feature matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def detect_and_match_features(self, img1, img2):
        # Convert images to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and compute descriptors
        keypoints1, descriptors1 = self.sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = self.sift.detectAndCompute(gray2, None)
        
        # Match features using FLANN
        matches = self.flann.knnMatch(descriptors1, descriptors2, k=2)
        
        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
                
        return keypoints1, keypoints2, good_matches

    def homography(self, data_points):
        matrix_A = []
        for point in data_points:
            src_x, src_y = point[0], point[1]
            dst_X, dst_Y = point[2], point[3]
            matrix_A.append([src_x, src_y, 1, 0, 0, 0, -dst_X * src_x, -dst_X * src_y, -dst_X])
            matrix_A.append([0, 0, 0, src_x, src_y, 1, -dst_Y * src_x, -dst_Y * src_y, -dst_Y])
    
        matrix_A = np.array(matrix_A)
        u_matrix, s_values, vh_matrix = np.linalg.svd(matrix_A)
        homography_matrix = (vh_matrix[-1, :].reshape(3, 3))
        homography_matrix /= homography_matrix[2, 2]
        return homography_matrix
    
    def ransac(self, valid_points, num_iterations=1000):
        optimal_inliers = []
        resulting_H = None
        threshold = 5
        for _ in range(num_iterations):
            selected_points = random.sample(valid_points, k=4)
            H_matrix = self.compute_homography(selected_points)
            inlier_points = []
            for point in valid_points:
                homogeneous_point = np.array([point[0], point[1], 1]).reshape(3, 1)
                corresponding_point = np.array([point[2], point[3], 1]).reshape(3, 1)
                transformed_point = np.dot(H_matrix, homogeneous_point)
                transformed_point /= transformed_point[2]
                distance = np.linalg.norm(corresponding_point - transformed_point)
    
                if distance < threshold:
                    inlier_points.append(point)
    
            if len(inlier_points) > len(optimal_inliers):
                optimal_inliers = inlier_points
                resulting_H = H_matrix
    
        return resulting_H


    def find_homography(self, keypoints1, keypoints2, good_matches):
        if len(good_matches) < 4:
            return None
            
        # Extract points from keypoints and matches
        good_pts = []
        for match in good_matches:
            pt1 = keypoints1[match.queryIdx].pt
            pt2 = keypoints2[match.trainIdx].pt
            good_pts.append([pt1[0], pt1[1], pt2[0], pt2[1]])
            
        # Find homography matrix using custom RANSAC implementation
        H = self.ransac(good_pts)
        
        return H

    def warp_images(self, img1, img2, H):
        # Get dimensions
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Create points for corners of img1
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        
        # Transform corners of img1
        corners1_trans = cv2.perspectiveTransform(corners1, H)
        corners = np.concatenate((corners2, corners1_trans), axis=0)
        
        # Find dimensions of new image
        [x_min, y_min] = np.int32(corners.min(axis=0).ravel())
        [x_max, y_max] = np.int32(corners.max(axis=0).ravel())
        
        # Translation matrix
        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]], 
                                [0, 1, translation_dist[1]], 
                                [0, 0, 1]])
        
        # Warp images
        output_img = cv2.warpPerspective(img1, H_translation.dot(H),
                                       (x_max-x_min, y_max-y_min))
        
        # Place img2 on the panorama
        output_img[translation_dist[1]:h2+translation_dist[1],
                  translation_dist[0]:w2+translation_dist[0]] = img2
                  
        return output_img

    def make_panaroma_for_images_in(self, path):
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))
        
        if len(all_images) < 2:
            raise ValueError("Need at least 2 images to create a panorama")
            
        # Read the first image
        base_image = cv2.imread(all_images[0])
        homography_matrix_list = []
        
        # Process all images in sequence
        for i in range(1, len(all_images)):
            # Read next image
            next_image = cv2.imread(all_images[i])
            
            # Detect and match features
            keypoints1, keypoints2, good_matches = self.detect_and_match_features(base_image, next_image)
            
            # Find homography
            H = self.find_homography(keypoints1, keypoints2, good_matches)
            if H is None:
                print(f"Warning: Could not find good homography for image {i}")
                continue
                
            homography_matrix_list.append(H)
            
            # Warp and combine images
            try:
                base_image = self.warp_images(base_image, next_image, H)
            except cv2.error as e:
                print(f"Error warping image {i}: {e}")
                continue
        
        # Crop the final panorama to remove black borders
        gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        base_image = base_image[y:y+h, x:x+w]
        
        return base_image, homography_matrix_list
