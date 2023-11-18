import unittest
from pathlib import Path
import numpy as np
from c2l.datasets.kitti_odometry_dataset import KittiOdometry


class TestKittiOdometryDataset(unittest.TestCase):

    def setUp(self) -> None:
        self.dataset = KittiOdometry(
            "/home/benjin/Data/kitti_odometry/dataset")

    def test_init(self):
        # Check that data has been loaded correctly
        self.assertEqual(set(self.dataset.seq_to_calib.keys()),
                         set(range(KittiOdometry.num_seq)))

        for seq_id in range(KittiOdometry.num_seq):
            self.assertEqual(
                set(self.dataset.seq_to_calib[seq_id].keys()),
                set(['P_rect_00', 'P_rect_10', 'P_rect_20', 'P_rect_30',
                     'T_cam0_velo', 'T_cam1_velo', 'T_cam2_velo', 'T_cam3_velo',
                     'K_cam0', 'K_cam1', 'K_cam2', 'K_cam3'])
            )

            # Check that the calibration matrices have the correct shape
            self.assertEqual(
                self.dataset.seq_to_calib[seq_id]['P_rect_00'].shape, (3, 4))
            self.assertEqual(
                self.dataset.seq_to_calib[seq_id]['P_rect_10'].shape, (3, 4))
            self.assertEqual(
                self.dataset.seq_to_calib[seq_id]['P_rect_20'].shape, (3, 4))
            self.assertEqual(
                self.dataset.seq_to_calib[seq_id]['P_rect_30'].shape, (3, 4))
            self.assertEqual(
                self.dataset.seq_to_calib[seq_id]['T_cam0_velo'].shape, (4, 4))
            self.assertEqual(
                self.dataset.seq_to_calib[seq_id]['T_cam1_velo'].shape, (4, 4))
            self.assertEqual(
                self.dataset.seq_to_calib[seq_id]['T_cam2_velo'].shape, (4, 4))
            self.assertEqual(
                self.dataset.seq_to_calib[seq_id]['T_cam3_velo'].shape, (4, 4))
            self.assertEqual(
                self.dataset.seq_to_calib[seq_id]['K_cam0'].shape, (3, 3))
            self.assertEqual(
                self.dataset.seq_to_calib[seq_id]['K_cam1'].shape, (3, 3))
            self.assertEqual(
                self.dataset.seq_to_calib[seq_id]['K_cam2'].shape, (3, 3))
            self.assertEqual(
                self.dataset.seq_to_calib[seq_id]['K_cam3'].shape, (3, 3))

        # Check a single data sample
        sample = self.dataset.data[0]
        self.assertIsInstance(sample.pcl, Path)
        self.assertIsInstance(sample.img, Path)
        self.assertIsInstance(sample.K, np.ndarray)
        self.assertEqual(sample.K.shape, (3, 3))
        self.assertIsInstance(sample.T, np.ndarray)
        self.assertEqual(sample.T.shape, (4, 4))
        self.assertIsInstance(sample.metadata, dict)
        self.assertEqual(set(sample.metadata.keys()), set(
            ['seq_id', 'item_id', 'cam_id', 'timestamp', 'pose']))
        self.assertIsInstance(sample.metadata['seq_id'], int)
        self.assertIsInstance(sample.metadata['item_id'], int)
        self.assertIsInstance(sample.metadata['cam_id'], int)
        self.assertIsInstance(sample.metadata['timestamp'], str)
        self.assertIsInstance(sample.metadata['pose'], np.ndarray)
        self.assertEqual(sample.metadata['pose'].shape, (4, 4))

    def test_getitem(self):
        # Check that the __getitem__ method returns a correct sample
        sample = self.dataset[0]
        self.assertIsInstance(sample.pcl, np.ndarray)
        self.assertEqual(sample.pcl.dtype, np.float32)
        self.assertEqual(sample.pcl.shape[1], 4)
        self.assertIsInstance(sample.img, np.ndarray)
        self.assertEqual(sample.img.dtype, np.uint8)
        self.assertEqual(sample.img.shape[2], 3)
        self.assertIsInstance(sample.K, np.ndarray)
        self.assertEqual(sample.K.shape, (3, 3))
        self.assertIsInstance(sample.T, np.ndarray)
        self.assertEqual(sample.T.shape, (4, 4))
        self.assertIsInstance(sample.metadata, dict)
        self.assertEqual(set(sample.metadata.keys()), set(
            ['seq_id', 'item_id', 'cam_id', 'timestamp', 'pose']))
        self.assertIsInstance(sample.metadata['seq_id'], int)
        self.assertIsInstance(sample.metadata['item_id'], int)
        self.assertIsInstance(sample.metadata['cam_id'], int)
        self.assertIsInstance(sample.metadata['timestamp'], str)
        self.assertIsInstance(sample.metadata['pose'], np.ndarray)
        self.assertEqual(sample.metadata['pose'].shape, (4, 4))
