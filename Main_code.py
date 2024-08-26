import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2


def local_logarithmic_transform(image, block_size, overlap):
    result = np.zeros_like(image, dtype=np.float32)

    for i in range(0, image.shape[0] - overlap, block_size - overlap):
        for j in range(0, image.shape[1] - overlap, block_size - overlap):
            block = image[i:i + block_size, j:j + block_size]
            mean_intensity = np.mean(block)
            block_min = np.min(block)
            if block_min < 0:
                block = block - block_min
            epsilon = 1e-8
            block = block + epsilon
            result[i:i + block_size, j:j + block_size] = np.log(1 + block - mean_intensity)

    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    result = result.astype(np.uint8)

    return result


def histogram_equalization(img):
    equalized_img = cv2.equalizeHist(img)
    return equalized_img


def log_transform(img, eta=1, gamma=1, lmbda=1):
    img_fft = fft2(img)
    phase = np.angle(img_fft)
    log_magnitude = gamma * np.log(eta * np.abs(img_fft) + lmbda)
    transformed_img_fft = np.exp(log_magnitude + 1j * phase)
    result_img = np.abs(ifft2(transformed_img_fft)).astype(np.uint8)

    return result_img


def clane_enhancement(image, block_size, overlap):
    result = np.zeros_like(image, dtype=np.float32)

    for i in range(0, image.shape[0] - overlap, block_size - overlap):
        for j in range(0, image.shape[1] - overlap, block_size - overlap):
            block = image[i:i + block_size, j:j + block_size]
            log_transformed_block = log_transform(block)
            equalized_block = histogram_equalization(log_transformed_block)
            result[i:i + block_size, j:j + block_size] = equalized_block

    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    result = result.astype(np.uint8)

    return result


def calculate_eme(image):
    k1, k2 = image.shape
    eme = np.max((1 / (k1 * k2)) * np.sum(20 * np.log10(image / np.min(image) + 1e-10)))
    return eme if eme > 0 else 1e-10


def apply_image_enhancement(image, block_size, overlap):
    local_log_transformed_image = local_logarithmic_transform(image, block_size, overlap)

    enhanced_image1 = local_log_transformed_image
    enhanced_image2 = clane_enhancement(image, block_size, overlap)
    enhanced_image3 = clane_enhancement(image, block_size * 2, overlap * 2)
    enhanced_image4 = clane_enhancement(image, block_size * 4, overlap * 4)

    eme1 = calculate_eme(enhanced_image1)
    eme2 = calculate_eme(enhanced_image2)
    eme3 = calculate_eme(enhanced_image3)
    eme4 = calculate_eme(enhanced_image4)

    total_eme = eme1 + eme2 + eme3 + eme4

    if total_eme > 0:
        weight1 = eme1 / total_eme
        weight2 = eme2 / total_eme
        weight3 = eme3 / total_eme
        weight4 = eme4 / total_eme
    else:
        weight1 = weight2 = weight3 = weight4 = 0

    weighted_average_image = (
        enhanced_image1 * weight1 +
        enhanced_image2 * weight2 +
        enhanced_image3 * weight3 +
        enhanced_image4 * weight4
    ).astype(np.uint8)

    return enhanced_image1, enhanced_image2, enhanced_image3, enhanced_image4, weighted_average_image


thermal_image = cv2.imread('thermal_image.jpg', cv2.IMREAD_GRAYSCALE)

block_size = 8
overlap = 2

enhanced_image1, enhanced_image2, enhanced_image3, enhanced_image4, weighted_average_image = \
    apply_image_enhancement(thermal_image, block_size, overlap)

cv2.imshow('Thermal Image', thermal_image)
cv2.imshow('Enhanced Image 1', enhanced_image1)
cv2.imshow('Enhanced Image 2', enhanced_image2)
cv2.imshow('Enhanced Image 3', enhanced_image3)
cv2.imshow('Enhanced Image 4', enhanced_image4)
cv2.imshow('Weighted Average Image', weighted_average_image)

cv2.waitKey(0)
cv2.destroyAllWindows()