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

class PanaromaStitcher:
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, folder_path: str) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
        image_paths = sorted(glob.glob(os.path.join(folder_path, '*')))
        print(f'Found {len(image_paths)} images for stitching')

        if len(image_paths) < 2:
            print('Need at least 2 images to stitch')
            return None, []
        
        downscale_factor = 0.25 if len(image_paths) >= 6 else 0.6
        print(f'Reducing image size by a factor of: {downscale_factor}')
        
        panorama_image = cv2.resize(cv2.imread(image_paths[0]), (0, 0), fx=downscale_factor, fy=downscale_factor)
        transformation_matrices = []

        for idx in range(1, min(len(image_paths), 5)):
            image_left = panorama_image
            image_right = cv2.resize(cv2.imread(image_paths[idx]), (0, 0), fx=downscale_factor, fy=downscale_factor)
            panorama_image, homography_matrix = self.stitch_images(image_left, image_right)
            transformation_matrices.append(homography_matrix)
            print(f'Stitching completed for image {idx} in panorama')

        return panorama_image, transformation_matrices

    def warp_image(self, image: np.ndarray, matrix: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
        width, height = output_size
        output_image = np.zeros((height, width, 3) if len(image.shape) == 3 else (height, width), dtype=image.dtype)
        
        y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        homogenous_points = np.stack([x_indices, y_indices, np.ones_like(x_indices)], axis=-1)
        
        inv_transform = np.linalg.inv(matrix)
        transformed_points = homogenous_points.reshape(-1, 3) @ inv_transform.T
        transformed_points = transformed_points[:, :2] / transformed_points[:, 2:]
        transformed_points = transformed_points.reshape(height, width, 2)

        x_transformed = transformed_points[:, :, 0].astype(np.int32)
        y_transformed = transformed_points[:, :, 1].astype(np.int32)
        
        valid_pixels = (
            (x_transformed >= 0) & (x_transformed < image.shape[1]) &
            (y_transformed >= 0) & (y_transformed < image.shape[0])
        )
        
        output_image[valid_pixels] = image[y_transformed[valid_pixels], x_transformed[valid_pixels]]
        return output_image

    def apply_perspective_transform(self, points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        original_shape = points.shape

        if points.ndim == 3:
            points = points.reshape(-1, 2)

        homogenous_points = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed_points = homogenous_points @ matrix.T
        transformed_points = transformed_points[:, :2] / transformed_points[:, 2:]

        large_value_mask = np.abs(transformed_points) > 1e10
        transformed_points[large_value_mask] = 0

        return transformed_points.reshape(original_shape) if original_shape[0] > 1 else transformed_points

    def compute_ransac(self, matched_points: List[Tuple[float]]) -> np.ndarray:
        best_inliers = []
        best_transform = []
        threshold = 5
        
        for _ in range(500):
            sample = random.choices(matched_points, k=4)
            transform_matrix = self.calculate_homography(sample)
            inliers = []
            
            for match in matched_points:
                source = np.array([match[0], match[1], 1]).reshape(3, 1)
                destination = np.array([match[2], match[3], 1]).reshape(3, 1)
                projected = transform_matrix @ source
                projected /= projected[2]
                
                if np.linalg.norm(destination - projected) < threshold:
                    inliers.append(match)

            if len(inliers) > len(best_inliers):
                best_inliers, best_transform = inliers, transform_matrix
                
        return best_transform

    def stitch_images(self, img_left: np.ndarray, img_right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        kp_left, desc_left, kp_right, desc_right = self.extract_keypoints_and_descriptors(img_left, img_right)
        matched_points = self.match_keypoints(kp_left, kp_right, desc_left, desc_right)
        homography = self.compute_ransac(matched_points)

        img_right_h, img_right_w = img_right.shape[:2]
        img_left_h, img_left_w = img_left.shape[:2]
        
        right_corners = np.float32([[0, 0], [0, img_right_h], [img_right_w, img_right_h], [img_right_w, 0]]).reshape(-1, 1, 2)
        left_corners = np.float32([[0, 0], [0, img_left_h], [img_left_w, img_left_h], [img_left_w, 0]]).reshape(-1, 1, 2)
        transformed_corners = self.apply_perspective_transform(left_corners, homography)
        all_corners = np.concatenate((right_corners, transformed_corners), axis=0)

        [min_x, min_y] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [max_x, max_y] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]]) @ homography
        output = self.warp_image(img_left, translation, (max_x - min_x, max_y - min_y))
        output[-min_y:img_right_h - min_y, -min_x:img_right_w - min_x] = img_right

        return output, homography

    def calculate_homography(self, pairs: List[Tuple[float]]) -> np.ndarray:
        A = []
        for x1, y1, x2, y2 in pairs:
            A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
            A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
        
        _, _, vh = np.linalg.svd(A)
        homography = vh[-1].reshape(3, 3)
        return homography / homography[2, 2]

    def extract_keypoints_and_descriptors(self, img_left: np.ndarray, img_right: np.ndarray) -> Tuple:
        sift = cv2.SIFT_create()
        kp_left, desc_left = sift.detectAndCompute(img_left, None)
        kp_right, desc_right = sift.detectAndCompute(img_right, None)
        return kp_left, desc_left, kp_right, desc_right

    def match_keypoints(self, kp_left, kp_right, desc_left, desc_right) -> List[Tuple[float]]:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        matches = matcher.knnMatch(desc_left, desc_right, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                left_pt = kp_left[m.queryIdx].pt
                right_pt = kp_right[m.trainIdx].pt
                good_matches.append((left_pt[0], left_pt[1], right_pt[0], right_pt[1]))

        return good_matches
