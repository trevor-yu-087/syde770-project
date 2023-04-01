import torch
import pandas as pd
import numpy as np

rng = np.random.default_rng()

class SmartwatchDataset(torch.utils.data.Dataset):
    def __init__(self, valid_files, sample_period=0.02):
        """
        Parameters:
        -----------
        valid_files: list of filepaths to normalized data
        """
        super().__init__()
        self.data = []
        for file in valid_files:
            df = pd.read_csv(file)
            # Resample the data if needed
            df.index = pd.to_timedelta(df["time"], unit="seconds")
            df = df.drop("time", axis=1)
            df = df.resample(f"{sample_period}S").mean()
            self.data.append(df.values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Returns tuple of (imu, mocap) at index"""
        item = self.data[index]
        imu = item[:, 0:9]  # IMU sensor data [accel, mag, gyro]
        mocap = item[:, 9:]  # Mocap data [pos, quat]
        return imu, mocap
    

class SmartwatchAugmentLstm:
    """
    Collate function to apply random augmentations to the data
        - Randomly perturb the mocap positions
        - Randomly flip sign of mocap quaternion
        - Add random noise to IMU channels
        - Random crop to the signal (if possible)
    """
    def __init__(self, position_noise=0.2, accel_eps=0.01, gyro_eps=0.01, mag_eps=0.01, max_samples=512):
        """
        Parameters:
        -----------
        position_noise: float, limits on uniform distribution [-p, p] to add position offset to mocap
        accel_eps: float, standard deviation on Gaussian noise added to accelerometer channels
        gyro_eps: float, standard deviation on Gaussian noise added to gyroscope channels
        mag_eps: float, standard deviation on Gaussian noise added to mangetometer channels
        """
        self.position_noise = position_noise
        self.accel_eps = accel_eps
        self.gyro_eps = gyro_eps
        self.mag_eps = mag_eps
        self.max_samples = max_samples

    def _random_crop(self, imu, mocap):
        """
        Apply a random crop of the signal of length self.max_samples to both inputs and labels, if able to
        Due to targets being a shifted version of decoder inputs, we need to account for one extra timepoint
        """
        n, d = imu.shape
        max_offset = n - self.max_samples - 1

        if max_offset > 0:
            offset = rng.choice(max_offset)
            inds = slice(offset, offset + self.max_samples + 1)
            return imu[inds, :], mocap[inds, :]
        else:
            return imu, mocap

    def __call__(self, data):
        """
        Parameters:
        -----------
        data: list of tuple of (imu, mocap) of length batch_size
            imu: np.ndarray, dimensions (n_samples, 9), signal data for IMU accel, gyro, and mag
            mocap: np.ndarray, dimensions (n_samples, 7), position and quaternion data from mocap

        Returns:
        --------
        collated_data: dict of torch.nn.utils.rnn.PackedSequence with keys ["encoder_inputs", "decoder_inputs", "targets"]
        """
        encoder_inputs = []
        decoder_inputs = []
        targets = []
        for (imu, mocap) in data:
            imu, mocap = self._random_crop(imu, mocap)

            n_in, d_in = imu.shape
            n_out, d_out = mocap.shape
            assert n_in == n_out, "IMU and mocap must have the same number of sequence elements"
            assert d_in == 9, f"IMU data has dimensionality {d_in} instead of 9"
            assert d_out == 7, f"Mocap data has dimensionality {d_out} instead of 7"

            # Augment XYZ positions
            offset = rng.uniform(-self.position_noise, self.position_noise, size=(1, 3))
            mocap[:, 0:3] += offset
            # Augment quaternion sign
            sign = rng.choice([-1, 1])
            mocap[:, 4:] *= sign

            accel_noise = rng.normal(loc=0, scale=self.accel_eps, size=(n_in, 3))
            gyro_noise = rng.normal(loc=0, scale=self.gyro_eps, size=(n_in, 3))
            mag_noise = rng.normal(loc=0, scale=self.mag_eps, size=(n_in, 3))

            noise = np.hstack([accel_noise, gyro_noise, mag_noise])
            imu += noise

            # Ensure targets are one timestep shifted wrt inputs
            encoder_inputs.append(torch.FloatTensor(imu[:-1, :]))
            decoder_inputs.append(torch.FloatTensor(mocap[:-1, :]))
            targets.append(torch.FloatTensor(mocap[1:, :]))

        lengths = [len(item) for item in encoder_inputs]
        inds = np.flip(np.argsort(lengths)).copy()  # PackedSequence expects lengths from longest to shortest
        lengths = torch.LongTensor(lengths)[inds]

        # Sort by lengths
        encoder_inputs = [encoder_inputs[i] for i in inds]
        decoder_inputs = [decoder_inputs[i] for i in inds]
        targets = [targets[i] for i in inds]

        encoder_inputs = torch.nn.utils.rnn.pack_sequence(encoder_inputs)
        decoder_inputs = torch.nn.utils.rnn.pack_sequence(decoder_inputs)
        targets = torch.nn.utils.rnn.pack_sequence(targets)
        collated_data = {
            "encoder_inputs": encoder_inputs,
            "decoder_inputs": decoder_inputs,
            "targets": targets
        }
        return collated_data


def get_file_lists():
    """Get list of files to pass to dataset class
    Returns:
    --------
    train_files: list of str filepaths to pre-processed train data
    test_files: list of str filepaths to pre-processed test data
    """
    import glob
    valid_files = glob.glob("/root/data/smartwatch/subjects/*/*_full.csv")
    test_subjects = [f"S{n}" for n in [5, 10, 15, 20, 25, 30]]
    test_files = [file for file in valid_files for subject in test_subjects if f"/{subject}/" in file]
    train_files = [file for file in valid_files if file not in set(test_files)]
    return train_files, test_file
