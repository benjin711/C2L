from pathlib import Path
from typing import Dict, List
import numpy as np
from PIL import Image
from c2l.datasets.c2l_dataclasses import C2LDataSample


class KittiOdometry:

    num_seq = 22
    last_seq_w_poses = 11

    def __init__(self, dataset_path: str) -> None:
        super().__init__()

        dataset_path = Path(dataset_path)

        # Data
        self.data = []
        self.seq_to_calib = {}

        for seq_id in range(KittiOdometry.num_seq):
            seq_path = dataset_path / "sequences" / f"{seq_id:02d}"
            pcl_path = seq_path / "velodyne"

            self.seq_to_calib[seq_id] = self._parse_calib(
                seq_path / "calib.txt")

            with open(seq_path / "times.txt", 'r', encoding='utf-8') as f:
                timestamps = f.readlines()

            if seq_id < KittiOdometry.last_seq_w_poses:
                pose_path = dataset_path / "poses" / f"{seq_id:02d}.txt"
                poses = self._parse_poses(pose_path)

            for idx, timestamp in enumerate(timestamps):
                pcl = pcl_path / f"{idx:06d}.bin"

                for cam_id in [2, 3]:
                    img = seq_path / f"image_{cam_id}" / f"{idx:06d}.png"

                    self.data.append(C2LDataSample(
                        pcl=pcl,
                        img=img,
                        K=self.seq_to_calib[seq_id][f'K_cam{cam_id}'],
                        T=self.seq_to_calib[seq_id][f'T_cam{cam_id}_velo'],
                        metadata={
                            'seq_id': seq_id,
                            'item_id': idx,
                            'cam_id': cam_id,
                            'timestamp': timestamp,
                            'pose': poses[idx] if seq_id < KittiOdometry.last_seq_w_poses else None
                        }
                    ))

    def __len__(self) -> int:
        return len(self.data)  # 87104

    def get_sample(self, idx) -> C2LDataSample:
        sample = self.data[idx]

        sample.pcl = np.fromfile(
            sample.pcl, dtype=np.float32).reshape(-1, 4)  # (N, 4)
        sample.img = np.transpose(Image.open(sample.img), (2, 0, 1))  # (H, W, 3) -> (3, H, W)

        return sample

    def _parse_calib(self, filepath: Path) -> Dict[str, np.ndarray]:
        data = {}

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                try:
                    key, value = line.split(':', 1)
                except ValueError:
                    key, value = line.split(' ', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        # Create 3x4 projection matrices
        P_rect_00 = np.reshape(data['P0'], (3, 4))
        P_rect_10 = np.reshape(data['P1'], (3, 4))
        P_rect_20 = np.reshape(data['P2'], (3, 4))
        P_rect_30 = np.reshape(data['P3'], (3, 4))

        data['P_rect_00'] = P_rect_00
        data['P_rect_10'] = P_rect_10
        data['P_rect_20'] = P_rect_20
        data['P_rect_30'] = P_rect_30

        # Compute the rectified extrinsics from cam0 to camN
        # Small values in P_rect_X0[1, 3] and P_rect_X0[2, 3] can safely be neglected
        T1 = np.eye(4)
        T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T3 = np.eye(4)
        T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

        # Compute the velodyne to rectified camera coordinate transforms
        data['T_cam0_velo'] = np.reshape(data['Tr'], (3, 4))
        data['T_cam0_velo'] = np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]])
        data['T_cam1_velo'] = T1.dot(data['T_cam0_velo'])
        data['T_cam2_velo'] = T2.dot(data['T_cam0_velo'])
        data['T_cam3_velo'] = T3.dot(data['T_cam0_velo'])

        # Compute the camera intrinsics
        data['K_cam0'] = P_rect_00[0:3, 0:3]
        data['K_cam1'] = P_rect_10[0:3, 0:3]
        data['K_cam2'] = P_rect_20[0:3, 0:3]
        data['K_cam3'] = P_rect_30[0:3, 0:3]

        # Delete the unnecessary data
        del data['P0']
        del data['P1']
        del data['P2']
        del data['P3']
        del data['Tr']

        return data

    def _parse_poses(self, filepath: Path) -> List[np.ndarray]:
        with open(filepath, 'r', encoding='utf-8') as f:
            poses = f.readlines()

        poses = [np.vstack([
            np.fromstring(pose, sep=' ').reshape(3, 4),
            [0, 0, 0, 1]
        ])
            for pose in poses
        ]

        return poses
