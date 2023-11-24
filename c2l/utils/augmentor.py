from typing import Any, Sequence, Tuple, Union, Protocol
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CustomAugmentor(Protocol):
    """
    Custom augmentator interface. Augmentators are callables that perform data
    augmentation on an image, a pcl and a K matrix tuple. The point cloud
    is assumed to be in the camera frame.
    """

    def __call__(
        self,
        img: np.ndarray,
        pcl: np.ndarray,
        K: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            img (3, H, W): the image to be augmented
            pcl: the point cloud
            K: the camera matrix of the image
        Returns:
            the pcl and augmented image and camera matrix
        """
        ...  # pylint: disable=unnecessary-ellipsis


class CustomRandomCrop:
    """
    Custom random crop. This class is a wrapper for the torchvision
    RandomCrop class. It is used to make sure that the camera matrix
    is updated accordingly.
    """

    def __init__(
        self,
        size: Union[int, Sequence[int]],
    ) -> None:
        """
        Args:
            size: the size of the crop (H, W)
            padding: the padding on each border (L, T, R, B)
            pad_if_needed: whether the image should be padded if it is smaller than the crop
            fill: the value to be filled in the padded areas
            padding_mode: the padding mode
        """
        self.size = size

    def __call__(
        self,
        img: np.ndarray,
        pcl: np.ndarray,
        K: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            img (3, H, W): the image to be cropped
            pcl: the point cloud
            K: the camera matrix of the image
        Returns:
            the pcl and adjusted image and camera matrix
        """
        K = K.copy()

        img = torch.tensor(img)

        # Apply the random crop
        i, j, h, w = v2.RandomCrop.get_params(
            img,
            output_size=self.size
        )
        img = TF.crop(img, i, j, h, w)

        # Adjust for the crop
        K_p = np.array([
            [1, 0, -j],
            [0, 1, -i],
            [0, 0, 1]
        ])
        K = np.dot(K_p, K)

        # Convert the image back to numpy
        img = np.array(img)

        return img, pcl, K


class CustomRandomResizedCrop:
    """
    Custom random resized crop. This class is a wrapper for the torchvision
    RandomResizedCrop class. It is used to make sure that the camera matrix
    is updated accordingly.
    """

    def __init__(
        self,
        size: Union[int, Sequence[int]],
        scale: Tuple[float, float],
        ratio: Tuple[float, float],
        interpolation: Union[v2.InterpolationMode, int] = v2.InterpolationMode.BILINEAR
    ) -> None:
        """
        Args:
            size: the size of the crop (H, W)
            scale: the range of the size of the area of the original image to be cropped
            ratio: the range of the aspect ratio of the original image to be cropped
            interpolation: the interpolation mode
        """
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(
        self,
        img: np.ndarray,
        pcl: np.ndarray,
        K: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            img (3, H, W): the image to be cropped
            pcl: the point cloud
            K: the camera matrix of the image
        Returns:
            the pcl and adjusted image and camera matrix
        """
        K = K.copy()

        img = torch.tensor(img)

        # Apply the random crop
        i, j, h, w = v2.RandomResizedCrop.get_params(
            img,
            scale=self.scale,
            ratio=self.ratio,
        )
        img = TF.resized_crop(img, i, j, h, w, self.size, self.interpolation, True)

        # Adjust for the crop
        K_p = np.array([
            [1, 0, -j],
            [0, 1, -i],
            [0, 0, 1]
        ])
        K = np.dot(K_p, K)

        # Adjust for the resize
        K_pp = np.array([
            [img.shape[2] / w, 0, 0],
            [0, img.shape[1] / h, 0],
            [0, 0, 1]
        ])
        K = np.dot(K_pp, K)

        # Convert the image back to numpy
        img = np.array(img)

        return img, pcl, K


class CustomRandomRotation:
    """
    Custom random rotation around the image center. This class is a wrapper 
    for the torchvision RandomRotation class. It is used to make sure that the camera matrix
    is updated accordingly.
    """

    def __init__(
        self,
        degrees: Union[float, Sequence],
        interpolation: Union[v2.InterpolationMode, int] = v2.InterpolationMode.BILINEAR
    ) -> None:
        """
        Args:
            degrees: the range of the degrees of the rotation
            interpolation: the interpolation mode
        """
        if isinstance(degrees, (int, float)):
            degrees = (-degrees, degrees)

        self.degrees = degrees
        self.interpolation = interpolation

    def __call__(
        self,
        img: np.ndarray,
        pcl: np.ndarray,
        K: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            img (3, H, W): the image to be rotated
            pcl: the point cloud
            K: the camera matrix of the image
        Returns:
            the pcl and adjusted image and camera matrix
        """
        K = K.copy()

        # Convert the image to PIL
        img = torch.tensor(img)

        # Apply the random rotation
        angle = v2.RandomRotation.get_params(self.degrees)
        img = TF.rotate(img, angle, self.interpolation)

        # Adjust for the rotation around the center
        K_p = np.array([
            [1, 0, -img.shape[2] / 2],
            [0, 1, -img.shape[1] / 2],
            [0, 0, 1]
        ])
        K = np.dot(K_p, K)

        K_pp = np.array([
            [np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle)), 0],
            [-np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0],
            [0, 0, 1]
        ])
        K = np.dot(K_pp, K)

        K_ppp = np.array([
            [1, 0, img.shape[2] / 2],
            [0, 1, img.shape[1] / 2],
            [0, 0, 1]
        ])
        K = np.dot(K_ppp, K)

        # Convert the image back to numpy
        img = np.array(img)

        return img, pcl, K


class CustomRandomHorizontalFlip:
    """
    Custom random horizontal flip. This class is a wrapper for the torchvision
    RandomHorizontalFlip class. It is used to make sure that the camera matrix
    is updated accordingly.
    """

    def __init__(
        self,
        p: float = 0.5
    ) -> None:
        """
        Args:
            p: the probability of the image being flipped
        """
        self.p = p

    def __call__(
        self,
        img: np.ndarray,
        pcl: np.ndarray,
        K: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            img (3, H, W): the image to be flipped
            pcl: the point cloud
            K: the camera matrix of the image
        Returns:
            the pcl and adjusted image and camera matrix
        """
        K = K.copy()

        # Convert the image to PIL
        img = torch.tensor(img)

        # Apply the random flip
        if torch.rand(1) < self.p:
            img = TF.hflip(img)

            # Adjust for the flip
            K_p = np.array([
                [-1, 0, img.shape[2]],
                [0, 1, 0],
                [0, 0, 1]
            ])
            K = np.dot(K_p, K)

        # Convert the image back to numpy
        img = np.array(img)

        return img, pcl, K


class CustomRandomVerticalFlip:
    """
    Custom random vertical flip. This class is a wrapper for the torchvision
    RandomVerticalFlip class. It is used to make sure that the camera matrix
    is updated accordingly.
    """

    def __init__(
        self,
        p: float = 0.5
    ) -> None:
        """
        Args:
            p: the probability of the image being flipped
        """
        self.p = p

    def __call__(
        self,
        img: np.ndarray,
        pcl: np.ndarray,
        K: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            img (3, H, W): the image to be flipped
            pcl: the point cloud
            K: the camera matrix of the image
        Returns:
            the pcl and adjusted image and camera matrix
        """
        K = K.copy()

        # Convert the image to PIL
        img = torch.tensor(img)

        # Apply the random flip
        if torch.rand(1) < self.p:
            img = TF.vflip(img)

            # Adjust for the flip
            K_p = np.array([
                [1, 0, 0],
                [0, -1, img.shape[1]],
                [0, 0, 1]
            ])
            K = np.dot(K_p, K)

        # Convert the image back to numpy
        img = np.array(img)

        return img, pcl, K


class Augmentor:
    """
    Augmentor interface. Augmentors are callables that perform data
    augmentation on an image, a pcl and a K matrix tuple. The point cloud
    is assumed to be in the camera frame.
    """

    def __init__(
        self,
        custom_augmentors: Sequence[CustomAugmentor],
        color_augmentors: Sequence[Any],
    ) -> None:
        """
        Args:
            custom_augmentors: the custom_augmentors to be applied
            color_augmentors: the color augmentors to be applied
        """

        self.custom_augmentors = v2.Compose(custom_augmentors) if custom_augmentors else None
        self.color_augmentors = v2.RandomChoice(color_augmentors) if color_augmentors else None

        # ImageNet normalization
        self.normalization = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, inplace=True),
        ])

    def __call__(
        self,
        img: np.ndarray,
        pcl: np.ndarray,
        K: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            img (3, H, W): the image to be augmented
            pcl: the point cloud
            K: the camera matrix of the image
        Returns:
            the pcl and augmented image and camera matrix
        """
        # Apply the custom augmentors
        if self.custom_augmentors:
            img, pcl, K = self.custom_augmentors(img, pcl, K)

        # Apply the color augmentors
        img = torch.tensor(img)

        if self.color_augmentors:
            img = self.color_augmentors(img)

        img = self.normalization(img)
        img = np.array(img)

        return img, pcl, K
