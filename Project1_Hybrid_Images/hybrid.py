import sys
import cv2
import numpy as np
import math


def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # Initialize data
    length, width = kernel.shape
    k1, k2 = int((length - 1) / 2), int((width - 1) / 2)
    kernel_matrix = np.asmatrix(kernel).reshape((length * width,1))
    # For grey scale image
    if len(img.shape) == 2:
        L, W = img.shape
        pixel_count = 0
        pixel_matrix = np.zeros((L * W, length * width))
        padding_img = np.zeros((L + k1 * 2, W + k2 * 2))
        padding_img[k1:k1 + L, k2:k2 + W] = img
        for i in range(L):
            for j in range(W):
                neighbor = padding_img[i:(length + i), j:(width + j)].reshape((length * width,))
                pixel_matrix[pixel_count] = neighbor
                pixel_count += 1
        return np.dot(pixel_matrix, kernel_matrix).reshape((L, W))

    # For RGB image
    if len(img.shape) == 3:
        L, W, C = img.shape
        result_per_channel = np.zeros((L, W, C))
        for k in range(C):
            pixel_count = 0
            pixel_matrix = np.zeros((L * W, length * width))
            padding_img = np.zeros((L + k1 * 2, W + k2 * 2, C))
            padding_img[k1:k1 + L, k2:k2 + W, k] = img[:, :, k]
            for i in range(L):
                for j in range(W):
                    neighbor = padding_img[i:(length + i), j:(width + j),k].reshape((length * width,))
                    pixel_matrix[pixel_count] = neighbor
                    pixel_count += 1
            result_per_channel[:, :, k] = np.dot(pixel_matrix, kernel_matrix).reshape((L, W, 1))
        return result_per_channel

    else:
        print("Input image is not identifiable.")


def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    flipped_kernel = kernel[::-1, ::-1]
    return cross_correlation_2d(img, flipped_kernel)


def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    norm = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            x = (i - (norm.shape[0] - 1) / 2)
            y = (j - (norm.shape[1] - 1) / 2)
            norm[i, j] = np.exp(-((x) ** 2 + (y) ** 2) / float(2 * sigma ** 2))
    kernel_blur = norm / np.sum(norm)
    return kernel_blur


def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return convolve_2d(img, gaussian_blur_kernel_2d(sigma, size, size))


def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return img - low_pass(img, sigma, size)


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
                        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)
