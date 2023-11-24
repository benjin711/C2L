from typing import Tuple
import unittest

import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import v2

from c2l.utils.augmentor import CustomRandomCrop, CustomRandomHorizontalFlip, \
    CustomRandomResizedCrop, CustomRandomRotation, CustomRandomVerticalFlip, \
    Augmentor


def get_dummy_data(H: int, W: int, num: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        H: the height of the image
        W: the width of the image
        num: the number of points in the point cloud
    Returns:
        the image, point cloud, and camera matrix
    """
    # Create checkerboard image
    tile_size = 120
    white = np.ones((tile_size, tile_size), np.uint8) * 255
    black = np.zeros((tile_size, tile_size), np.uint8)
    img = np.block([[white, black], [black, white]])
    img = np.tile(
        img[None, ...],
        (
            3,
            np.ceil(H / (2 * tile_size)).astype(np.int32),
            np.ceil(W / (2 * tile_size)).astype(np.int32),
        )
    )[:, :H, :W]

    # Create point cloud
    rng = np.random.default_rng(0)
    pcl = (rng.random((num, 4), np.float32) - 0.5) * 100

    # Create camera matrix
    K = np.eye(3, dtype=np.float32)
    K[0, 2] = 960
    K[1, 2] = 540
    K[0, 0] = 718
    K[1, 1] = 718

    return img, pcl, K


def draw_square(
    img: np.ndarray,
    pixel: np.ndarray,
    tile_size: int,
    color: Tuple[int, int, int]
) -> np.ndarray:
    """
    Args:
        img: the image to draw on
        pixel: the pixel to draw around
        tile_size: the size of the square
        color: the color of the square
    Returns:
        the image with the square drawn on it
    """
    h_range = (int(pixel[1]) - tile_size // 2, int(pixel[1]) + tile_size // 2)
    w_range = (int(pixel[0]) - tile_size // 2, int(pixel[0]) + tile_size // 2)
    img[:, h_range[0]:h_range[1], w_range[0]:w_range[1]] = np.array(color)[:, None, None]

    return img


def save_images(img1: np.ndarray, img2: np.ndarray, img3: np.ndarray, save_path: str) -> None:
    """
    Args:
        img1: the first image
        img2: the second image
        img3: the third image
        save_path: the path to save the images to
    """
    # Create a subplot with 1 row and 3 columns
    _, axes = plt.subplots(1, 3)

    # Display the first image in the first subplot
    img1 = np.moveaxis(img1, 0, -1)
    axes[0].imshow(img1)
    axes[0].set_title('Image 1')

    # Display the second image in the second subplot
    img2 = np.moveaxis(img2, 0, -1)
    axes[1].imshow(img2)
    axes[1].set_title('Image 2')

    # Display the third image in the third subplot
    img3 = np.moveaxis(img3, 0, -1)
    axes[2].imshow(img3)
    axes[2].set_title('Image 3')

    # Show the plot
    plt.savefig(save_path)


def transform_pixel(pixel: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Args:
        pixel: the pixel to transform
        K: the camera matrix
    Returns:
        the transformed pixel
    """
    pixel = np.array([pixel[0], pixel[1], 1], np.float32)
    pixel = K @ pixel
    pixel = pixel / pixel[2]
    pixel = pixel[:2]
    pixel = pixel.astype(np.int32)

    return pixel


class TestCustomRandomCrop(unittest.TestCase):

    def test_call(self):
        H_pre, W_pre = 1080, 1920
        H_post, W_post = 320, 960

        # Setup object of scrutiny
        crc = CustomRandomCrop(
            size=(H_post, W_post),
        )

        img, pcl, K = get_dummy_data(H_pre, W_pre, 100)

        # Draw red patch at center location
        mid_pixel = np.array([W_pre // 2, H_pre // 2, 1], np.int32)
        img = draw_square(img, mid_pixel, 90, (255, 0, 0))

        # Call the object of scrutiny
        img_p, pcl_p, K_p = crc(img, pcl, K)

        # Check the results
        self.assertEqual(img_p.shape, (3, H_post, W_post))
        self.assertTrue(np.isclose(pcl, pcl_p).all())

        # Manually inspect images to test that the image is indeed getting cropped and resized
        # Additionally, check that the camera matrix is being adjusted correctly. This can be
        # done by checking that "K_p @ np.linalg.inv(K) @ mid_pixel" overlaps with the transformed
        # right square.
        DEBUG = False
        if DEBUG:
            # Transform mid_pixel to the post-crop image using the adjusted camera matrix
            mid_pixel = mid_pixel.astype(np.float32)
            mid_pixel = transform_pixel(mid_pixel, K_p @ np.linalg.inv(K))

            # Draw blue square around the transformed mid_pixel
            img_p_copy = img_p.copy()
            img_p_copy = draw_square(img_p_copy, mid_pixel, 30, (0, 0, 255))

            # Save the images for visual inspection
            save_images(img, img_p, img_p_copy, 'test_custom_random_crop.png')


class TestCustomRandomResizedCrop(unittest.TestCase):

    def test_call(self):
        H_pre, W_pre = 1080, 1920
        H_post, W_post = 320, 960

        # Setup object of scrutiny
        crrc = CustomRandomResizedCrop(
            size=(H_post, W_post),
            scale=(0.5, 1.0),
            ratio=(3 * 3 / 4, 3 * 4 / 3)
        )

        img, pcl, K = get_dummy_data(H_pre, W_pre, 100)

        # Draw red patch at center location
        mid_pixel = np.array([W_pre // 2, H_pre // 2, 1], np.int32)
        img = draw_square(img, mid_pixel, 90, (255, 0, 0))

        # Call the object of scrutiny
        img_p, pcl_p, K_p = crrc(img, pcl, K)

        # Check the results
        self.assertEqual(img_p.shape, (3, H_post, W_post))
        self.assertTrue(np.isclose(pcl, pcl_p).all())

        # Manually inspect images to test that the image is indeed getting cropped and resized
        # Additionally, check that the camera matrix is being adjusted correctly. This can be
        # done by checking that "K_p @ np.linalg.inv(K) @ mid_pixel" overlaps with the transformed
        # right square.
        DEBUG = False
        if DEBUG:
            # Transform mid_pixel to the post-crop image using the adjusted camera matrix
            mid_pixel = mid_pixel.astype(np.float32)
            mid_pixel = transform_pixel(mid_pixel, K_p @ np.linalg.inv(K))

            # Draw blue square around the transformed mid_pixel
            img_p_copy = img_p.copy()
            img_p_copy = draw_square(img_p_copy, mid_pixel, 30, (0, 0, 255))

            # Save the images for visual inspection
            save_images(img, img_p, img_p_copy, 'test_custom_random_resized_crop.png')


class TestCustomRandomRotation(unittest.TestCase):

    def test_call(self):
        H, W = 1080, 1920

        # Setup object of scrutiny
        crr = CustomRandomRotation(
            degrees=45,
        )

        img, pcl, K = get_dummy_data(H, W, 100)

        # Draw red patch at non-center location
        not_center = np.array([W // 2, H // 2, 1], np.int32)
        not_center[0] -= 100
        not_center[1] -= 200
        img = draw_square(img, not_center, 90, (255, 0, 0))

        # Call the object of scrutiny
        img_p, pcl_p, K_p = crr(img, pcl, K)

        # Check the results
        self.assertEqual(img_p.shape, (3, H, W))
        self.assertTrue(np.isclose(pcl, pcl_p).all())

        # Manually inspect images to test that the image is indeed getting rotated
        # Additionally, check that the camera matrix is being adjusted correctly. This can be
        # done by checking that "K_p @ np.linalg.inv(K) @ mid_pixel" overlaps with the transformed
        # right square.
        DEBUG = False
        if DEBUG:
            # Transform not_center to the post-crop image using the adjusted camera matrix
            not_center = transform_pixel(not_center, K_p @ np.linalg.inv(K))

            # Draw blue square around the transformed mid_pixel
            img_p_copy = img_p.copy()
            img_p_copy = draw_square(img_p_copy, not_center, 40, (0, 0, 255))

            # Save the images for visual inspection
            save_images(img, img_p, img_p_copy, 'test_custom_random_rotation.png')


class TestCustomRandomHorizontalFlip(unittest.TestCase):

    def test_call(self):
        H, W = 1080, 1920

        # Setup object of scrutiny
        crhf = CustomRandomHorizontalFlip(
            p=1.0,
        )

        img, pcl, K = get_dummy_data(H, W, 100)
        K[0, 2] -= 100
        K[1, 2] -= 200

        # Draw red patch at non-center location
        not_center = np.array([W // 2, H // 2, 1], np.int32)
        not_center[0] -= 300
        not_center[1] -= 200
        img = draw_square(img, not_center, 90, (255, 0, 0))

        # Call the object of scrutiny
        img_p, pcl_p, K_p = crhf(img, pcl, K)

        # Check the results
        self.assertEqual(img_p.shape, (3, H, W))
        self.assertTrue(np.isclose(pcl, pcl_p).all())

        # Manually inspect images to test that the image is indeed getting flipped
        # Additionally, check that the camera matrix is being adjusted correctly. This can be
        # done by checking that "K_p @ np.linalg.inv(K) @ mid_pixel" overlaps with the transformed
        # right square.
        DEBUG = False
        if DEBUG:
            # Transform non-center pixel to the post-crop image using the adjusted camera matrix
            not_center = not_center.astype(np.float32)
            not_center = transform_pixel(not_center, K_p @ np.linalg.inv(K))

            # Draw blue square around the transformed mid_pixel
            img_p_copy = img_p.copy()
            img_p_copy = draw_square(img_p_copy, not_center, 40, (0, 0, 255))

            # Save the images for visual inspection
            save_images(img, img_p, img_p_copy, 'test_custom_random_horizontal_flip.png')


class TestCustomRandomVerticalFlip(unittest.TestCase):

    def test_call(self):
        H, W = 1080, 1920

        # Setup object of scrutiny
        crhf = CustomRandomVerticalFlip(
            p=1.0,
        )

        img, pcl, K = get_dummy_data(H, W, 100)
        K[0, 2] -= 100
        K[1, 2] -= 200

        # Draw red patch at non-center location
        not_center = np.array([W // 2, H // 2, 1], np.int32)
        not_center[0] -= 100
        not_center[1] -= 300
        img = draw_square(img, not_center, 90, (255, 0, 0))

        # Call the object of scrutiny
        img_p, pcl_p, K_p = crhf(img, pcl, K)

        # Check the results
        self.assertEqual(img_p.shape, (3, H, W))
        self.assertTrue(np.isclose(pcl, pcl_p).all())

        # Manually inspect images to test that the image is indeed getting flipped
        # Additionally, check that the camera matrix is being adjusted correctly. This can be
        # done by checking that "K_p @ np.linalg.inv(K) @ mid_pixel" overlaps with the transformed
        # right square.
        DEBUG = False
        if DEBUG:
            # Transform mid_pixel to the post-crop image using the adjusted camera matrix
            not_center = not_center.astype(np.float32)
            not_center = transform_pixel(not_center, K_p @ np.linalg.inv(K))

            # Draw blue square around the transformed mid_pixel
            img_p_copy = img_p.copy()
            img_p_copy = draw_square(img_p_copy, not_center, 40, (0, 0, 255))

            # Save the images for visual inspection
            save_images(img, img_p, img_p_copy, 'test_custom_random_vertical_flip.png')


class TestAugmentor(unittest.TestCase):

    def setUp(self) -> None:
        custom_augmentors = [
            CustomRandomResizedCrop(
                size=(320, 960),
                scale=(0.5, 1.0),
                ratio=(3 * 3 / 4, 3 * 4 / 3)
            ),
            CustomRandomRotation(
                degrees=45,
            ),
            CustomRandomHorizontalFlip(
                p=1.0,
            ),
            CustomRandomVerticalFlip(
                p=1.0,
            ),
        ]
        color_augmentors = [
            v2.Grayscale(num_output_channels=3),
            v2.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.5
            ),
        ]
        self.aug = Augmentor(
            custom_augmentors=custom_augmentors,
            color_augmentors=color_augmentors
        )

    def test_call(self):
        H_pre, W_pre = 1080, 1920

        img, pcl, K = get_dummy_data(H_pre, W_pre, 100)

        # Draw red patch at non-center location
        not_center = np.array([W_pre // 2, H_pre // 2, 1], np.int32)
        not_center[0] -= 100
        not_center[1] -= 200
        img = draw_square(img, not_center, 90, (255, 0, 0))

        # Call the object of scrutiny
        img_p, pcl_p, K_p = self.aug(img, pcl, K)

        # Check the results
        self.assertEqual(img_p.shape, (3, 320, 960))
        self.assertTrue(np.isclose(pcl, pcl_p).all())

        # Manually inspect images to test that the image is indeed getting cropped and resized
        # Additionally, check that the camera matrix is being adjusted correctly. This can be
        # done by checking that "K_p @ np.linalg.inv(K) @ mid_pixel" overlaps with the transformed
        # right square.
        DEBUG = False
        if DEBUG:
            # Transform not_center to the post-crop image using the adjusted camera matrix
            not_center = not_center.astype(np.float32)
            not_center = transform_pixel(not_center, K_p @ np.linalg.inv(K))

            # Draw blue square around the transformed mid_pixel
            img_p_copy = img_p.copy()
            img_p_copy = draw_square(img_p_copy, not_center, 30, (0, 0, 255))

            # Save the images for visual inspection
            save_images(img, img_p, img_p_copy, 'test_augmentor.png')
