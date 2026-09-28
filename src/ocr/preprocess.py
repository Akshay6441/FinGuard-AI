"""
OpenCV Document Preprocessing Pipeline for FinGuard AI OCR.
Provides deskewing, denoising, adaptive thresholding, and text region contour extraction.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class DocumentPreprocessor:
    """
    OpenCV document enhancement and normalization pipeline.
    """

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return image

    @staticmethod
    def denoise(gray: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filter to reduce camera sensor noise while keeping text edges sharp.
        """
        return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    @staticmethod
    def binarize(gray: np.ndarray) -> np.ndarray:
        """
        Adaptive thresholding and Otsu binarization for clean high-contrast text.
        """
        # Otsu thresholding after subtle Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def estimate_skew_angle(binary: np.ndarray) -> float:
        """
        Estimate rotation skew angle of text lines using minimum area bounding box of text contours.
        """
        # Invert so text is white
        inv = cv2.bitwise_not(binary)
        coords = np.column_stack(np.where(inv > 0))
        if len(coords) < 50:
            return 0.0

        angle = cv2.minAreaRect(coords)[-1]
        # Normalize angle
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        # If angle is extreme, clamp to 0
        if abs(angle) > 40:
            return 0.0
        return float(angle)

    @staticmethod
    def deskew(image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image to correct skew angle.
        """
        if abs(angle) < 0.2:
            return image

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(
            image, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return deskewed

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Runs complete preprocessing: Grayscale -> Denoise -> Binarize -> Deskew.
        Returns (deskewed_color, deskewed_binary, estimated_angle).
        """
        gray = self.to_grayscale(image)
        denoised = self.denoise(gray)
        binary = self.binarize(denoised)
        angle = self.estimate_skew_angle(binary)

        if abs(angle) >= 0.2:
            deskewed_color = self.deskew(image, angle)
            deskewed_binary = self.deskew(binary, angle)
        else:
            deskewed_color = image
            deskewed_binary = binary

        return deskewed_color, deskewed_binary, angle
