import unittest
from unittest.mock import Mock

import numpy as np
import torch
from c2l.datasets.c2l_dataset_wrapper import C2LDatasetWrapper
from c2l.datasets.c2l_dataclasses import C2LDataSample


class TestC2LDatasetWrapper(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)

        # Mock dataset
        self.mock_dataset = Mock()

        T_cam_velo = np.array([
            [0.0138,  0.8051, -0.5930, -0.3636],
            [-0.8044,  0.3612,  0.4716,  0.5556],
            [0.5939,  0.4705,  0.6526, -0.8707],
            [0.0000,  0.0000,  0.0000,  1.0000]
        ], dtype=np.float32)

        mock_pcl = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float32)

        self.mock_sample = C2LDataSample(
            pcl=mock_pcl,
            img=rng.integers(0, 255, size=(3, 2, 4), dtype=np.uint8),
            K=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32),
            T=T_cam_velo,
            metadata={'test': 'test'}
        )
        self.mock_dataset.get_sample.return_value = self.mock_sample

        # Mock augmentator
        self.mock_augmentator = Mock(side_effect=lambda img, pcl, K: (img, pcl, K))

        # Mock transformation sampler
        self.mock_transformation_sampler = Mock()

        self.mock_transformation = torch.tensor([
            [-0.2641,  0.6158,  0.7423, -0.0923],
            [0.6632,  0.6748, -0.3238,  0.5900],
            [-0.7003,  0.4068, -0.5866, -0.7989],
            [0.0000,  0.0000,  0.0000,  1.0000]
        ], dtype=torch.float32)

        self.mock_transformation_sampler.return_value = self.mock_transformation

        # Create dataset wrapper for testing
        self.dataset_wrapper = C2LDatasetWrapper(
            self.mock_dataset, self.mock_augmentator, self.mock_transformation_sampler)

    def test_getitem(self):
        sample = self.dataset_wrapper[0]

        mock_pcl = self.mock_sample.pcl
        T_cam_velo = self.mock_sample.T

        # Check that the intensity channel is not modified
        self.assertTrue(np.isclose(sample.pcl[:, 3], mock_pcl[:, 3]).all())

        # Check that the pcl is transformed from velo to cam to prior frame
        pcl = np.hstack((mock_pcl[:, :3], np.ones((mock_pcl.shape[0], 1))))
        pcl_p = np.dot(T_cam_velo, pcl.T).T
        pcl_pp = np.dot(self.mock_transformation.numpy(), pcl_p.T).T
        self.assertTrue(np.isclose(sample.pcl[:, :3], pcl_pp[:, :3]).all())

        # Check that the image, K, and metadata are not modified
        self.assertTrue(np.isclose(sample.img, self.mock_sample.img).all())
        self.assertTrue(np.isclose(sample.K, self.mock_sample.K).all())
        self.assertEqual(sample.metadata, self.mock_sample.metadata)

        # Check that the transformation is updated to prior to cam
        T_p = np.linalg.inv(self.mock_transformation)
        self.assertTrue(np.isclose(sample.T, T_p).all())

        self.mock_dataset.get_sample.assert_called_once_with(0)
        self.mock_transformation_sampler.assert_called_once_with()
