import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix


class TranslationSampler:

    dim = 3

    def __init__(
        self,
        low: float,
        high: float,
    ) -> None:
        self.low = low
        self.high = high

    def __call__(self, n: int = 1, device: str = 'cpu'):
        """
        Uniform sampling from a cube
        """
        return self.low + (self.high - self.low) * \
            torch.rand((n, TranslationSampler.dim), device=device)


class RotationSampler:

    def __init__(self, angle: float) -> None:
        """
        Args:
            angle (float): angle in radians
        """
        assert angle >= 0, "angle must be non-negative"
        assert angle <= np.pi, "angle must be less than or equal pi"
        self.angle = angle

    def __call__(self, n: int = 1, device: str = 'cpu'):
        """
        Uniform sampling of rotation axis on the unit sphere
        https://mathworld.wolfram.com/SpherePointPicking.html.
        Uniform sampling of rotation angle in [-angle, angle].

        Args:
            n (int, optional): number of samples. Defaults to 1.
            device (str, optional): device to sample on. Defaults to 'cpu'.

        Returns:
            torch.Tensor: random rotation matrices of shape (n, 3, 3)
        """
        axis = torch.rand((n, 3), device=device)
        axis = axis / torch.norm(axis, dim=1, keepdim=True)
        angle = 2 * self.angle * torch.rand((n, 1), device=device) - self.angle

        return axis_angle_to_matrix(angle * axis)


class TransformationSampler:

    def __init__(
        self,
        translation_sampler: TranslationSampler,
        rotation_sampler: RotationSampler,
    ) -> None:
        self.translation_sampler = translation_sampler
        self.rotation_sampler = rotation_sampler

    def __call__(self, n: int = 1, device: str = 'cpu'):
        """
        Args:
            n (int, optional): number of samples. Defaults to 1.
            device (str, optional): device to sample on. Defaults to 'cpu'.

        Returns:
            torch.Tensor: random transformation matrices of shape (n, 4, 4)
        """
        translation = self.translation_sampler(n, device)
        rotation = self.rotation_sampler(n, device)
        transformation = torch.eye(4, device=device).repeat(n, 1, 1)
        transformation[:, :3, :3] = rotation
        transformation[:, :3, 3] = translation

        return transformation
