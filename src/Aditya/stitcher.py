import pdb
import glob
import cv2
import os
from src.JohnDoe import some_function
from src.JohnDoe.some_folder import folder_func
from typing import List, Tuple, Optional
import numpy as np
import random

random.seed(1000)

class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, directory_path):
        image_folder = directory_path
        images_for_stitching = sorted(glob.glob(image_folder + os.sep + '*'))
        print('Found {} Images for stitching'.format(len(images_for_stitching)))

        transformations = []

        if len(images_for_stitching) < 2:
            print('Need at least 2 images to stitch')
            return None
        else:
            scale_factor = 0.25 if len(images_for_stitching) >= 6 else 0.6
            print('Image size is reduced to:', scale_factor, "times the original size")
            final_image = cv2.resize(cv2.imread(images_for_stitching[0]), (0, 0), fx=scale_factor, fy=scale_factor)
            for index in range(1, 5):
                img_left = final_image
                img_right = cv2.resize(cv2.imread(images_for_stitching[index]), (0, 0), fx=scale_factor, fy=scale_factor)
                final_image, transformation_matrix = self.stitch_images(img_left, img_right)
                transformations.append(transformation_matrix)
                print('Stitching Done for Image {} with panorama'.format(index))
        stitched_output = final_image

        return stitched_output, transformations


    def apply_warp(self, img, transform, output_dimensions):
        img_width, img_height = output_dimensions
        output_image = np.zeros((img_height, img_width, 3) if len(img.shape) == 3 else (img_height, img_width), dtype=img.dtype)
        img_input_height, img_input_width = img.shape[:2]
        y_indices, x_indices = np.meshgrid(np.arange(img_height), np.arange(img_width), indexing='ij')
        homogenous_points = np.stack([x_indices, y_indices, np.ones_like(x_indices)], axis=-1)
        points_to_transform = homogenous_points.reshape(-1, 3)

        inverse_transform = np.linalg.inv(transform)
        transformed_points = points_to_transform @ inverse_transform.T
        transformed_points = transformed_points[:, :2] / transformed_points[:, 2:]
        transformed_points = transformed_points.reshape(img_height, img_width, 2)

        transformed_x = transformed_points[:, :, 0]
        transformed_y = transformed_points[:, :, 1]
        valid_pixels = (transformed_x >= 0) & (transformed_x < img_input_width - 1) & (transformed_y >= 0) & (transformed_y < img_input_height - 1)

        transformed_x = transformed_x.astype(np.int32)
        transformed_y = transformed_y.astype(np.int32)

        if len(img.shape) == 3:
            output_image[valid_pixels] = img[transformed_y[valid_pixels], transformed_x[valid_pixels]]
        else:
            output_image[valid_pixels] = img[transformed_y[valid_pixels], transformed_x[valid_pixels]]

        return output_image

    def transform_perspective(self, coordinates, transformation):
        original_shape = coordinates.shape

        if len(coordinates.shape) == 3:
            coordinates = coordinates.reshape(-1, 2)

        homogenous_coordinates = np.hstack([coordinates, np.ones((coordinates.shape[0], 1))])
        transformed_coordinates = homogenous_coordinates @ transformation.T
        transformed_coordinates = transformed_coordinates[:, :2] / transformed_coordinates[:, 2:]

        large_value_mask = np.abs(transformed_coordinates) > 1e10
        transformed_coordinates[large_value_mask] = 0

        if len(original_shape) == 3:
            transformed_coordinates = transformed_coordinates.reshape(original_shape)

        return transformed_coordinates

    def compute_ransac(self, match_points):
        best_inliers = []
        best_transform = []
        threshold_distance = 5
        for _ in range(500):
            sampled_points = random.choices(match_points, k=4)
            current_transform = self.compute_homography(sampled_points)
            current_inliers = []
            for match in match_points:
                source_point = np.array([match[0], match[1], 1]).reshape(3, 1)
                destination_point = np.array([match[2], match[3], 1]).reshape(3, 1)
                transformed_point = np.dot(current_transform, source_point)
                transformed_point = transformed_point / transformed_point[2]
                distance = np.linalg.norm(destination_point - transformed_point)

                if distance < threshold_distance:
                    current_inliers.append(match)

            if len(current_inliers) > len(best_inliers):
                best_inliers, best_transform = current_inliers, current_transform
        return best_transform

    def stitch_images(self, img_left, img_right):
        kp_left, desc_left, kp_right, desc_right = self.extract_keypoints(img_left, img_right)
        matched_points = self.find_matches(kp_left, kp_right, desc_left, desc_right)
        homography_matrix = self.compute_ransac(matched_points)

        img_right_height, img_right_width = img_right.shape[:2]
        img_left_height, img_left_width = img_left.shape[:2]

        corner_points_right = np.float32([[0, 0], [0, img_right_height], [img_right_width, img_right_height], [img_right_width, 0]]).reshape(-1, 1, 2)
        corner_points_left = np.float32([[0, 0], [0, img_left_height], [img_left_width, img_left_height], [img_left_width, 0]]).reshape(-1, 1, 2)
        transformed_corners = self.transform_perspective(corner_points_left, homography_matrix)
        all_corners = np.concatenate((corner_points_right, transformed_corners), axis=0)

        [min_x, min_y] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [max_x, max_y] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        translate_transform = np.array([[1, 0, (-min_x)], [0, 1, (-min_y)], [0, 0, 1]]).dot(homography_matrix)

        final_output = self.apply_warp(img_left, translate_transform, (max_x - min_x, max_y - min_y))
        final_output[(-min_y):img_right_height + (-min_y), (-min_x):img_right_width + (-min_x)] = img_right
        result = final_output
        return result, homography_matrix

    def compute_homography(self, coordinate_pairs):
        matrix_A = []
        for coordinate_pair in coordinate_pairs:
            x1, y1 = coordinate_pair[0], coordinate_pair[1]
            x2, y2 = coordinate_pair[2], coordinate_pair[3]
            matrix_A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
            matrix_A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])

        matrix_A = np.array(matrix_A)
        _, _, vh = np.linalg.svd(matrix_A)
        transform_matrix = vh[-1, :].reshape(3, 3)
        transform_matrix = transform_matrix / transform_matrix[2, 2]
        return transform_matrix

    def extract_keypoints(self, img_left, img_right):
        keypoints_left, descriptors_left = cv2.SIFT_create().detectAndCompute(img_left, None)
        keypoints_right, descriptors_right = cv2.SIFT_create().detectAndCompute(img_right, None)
        return keypoints_left, descriptors_left, keypoints_right, descriptors_right

    def find_matches(self, kp_left, kp_right, desc_left, desc_right):
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = matcher.knnMatch(desc_left, desc_right, k=2)

        filtered_matches = []
        for match1, match2 in matches:
            if match1.distance < 0.75 * match2.distance:
                left_coordinates = kp_left[match1.queryIdx].pt
                right_coordinates = kp_right[match1.trainIdx].pt
                filtered_matches.append([left_coordinates[0], left_coordinates[1], right_coordinates[0], right_coordinates[1]])

        return filtered_matches
