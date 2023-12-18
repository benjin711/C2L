import unittest

import numpy as np
import torch

from c2l.utils.transformation_sampler import (RotationSampler,
                                              TransformationSampler,
                                              TranslationSampler)


class TestTranslationSampler(unittest.TestCase):

    def setUp(self) -> None:
        self.sampler = TranslationSampler(-1, 1)

    def test_init(self):
        self.assertEqual(self.sampler.low, -1)
        self.assertEqual(self.sampler.high, 1)

    def test_call(self):
        samples = self.sampler(10)
        self.assertEqual(samples.shape, (10, 3))
        self.assertTrue((samples >= -1).all())
        self.assertTrue((samples <= 1).all())


class TestRotationSampler(unittest.TestCase):

    def setUp(self) -> None:
        self.sampler = RotationSampler(90)
        self.sampler_zero = RotationSampler(0)

    def test_init(self):
        self.assertEqual(self.sampler.angle, np.pi / 2)

    def test_call(self):
        samples = self.sampler(2)
        self.assertEqual(samples.shape, (2, 3, 3))
        self.assertTrue(
            torch.allclose(
                samples.transpose(1, 2) @ samples,
                torch.eye(3)[None, ...].repeat(2, 1, 1),
                rtol=1e-5, atol=1e-5
            )
        )

        samples = self.sampler_zero(2)
        self.assertEqual(samples.shape, (2, 3, 3))
        self.assertTrue(
            torch.allclose(
                samples.transpose(1, 2) @ samples,
                torch.eye(3)[None, ...].repeat(2, 1, 1),
                rtol=1e-5, atol=1e-5
            )
        )


class TestTransformationSampler(unittest.TestCase):

    def setUp(self) -> None:
        self.sampler = TransformationSampler(
            TranslationSampler(-.15, 0.15),
            RotationSampler(10 / 180 * np.pi)
        )

    def test_call(self):
        samples = self.sampler(2)
        self.assertEqual(samples.shape, (2, 4, 4))
