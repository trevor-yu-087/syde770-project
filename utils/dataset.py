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
        """Returns tuple of (data, label) at index"""
        item = self.data[index]
        inputs = item[:, 0:9]  # IMU sensor data [accel, mag, gyro]
        labels = item[:, 9:]  # Mocap data [pos, quat]
        return inputs, labels
    

class SmartwatchAugment:
    """
    Collate function to apply random augmentations to the data
        - Randomly perturb the mocap positions
        - Randomly flip sign of mocap quaternion
        - Add random noise to IMU channels
        - Random crop of the signal window, if possible
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

    def _random_crop(self, inputs, labels):
        """Apply a random crop of the signal of length self.max_samples to both inputs and labels, if able to"""
        n, d = inputs.shape
        max_offset = n - self.max_samples

        if max_offset > 0:
            offset = rng.choice(max_offset)
            inds = slice(offset, offset + self.max_samples)
            return inputs[inds, :], labels[inds, :]
        else:
            return inputs, labels

    def __call__(self, data):
        """
        Parameters:
        -----------
        data: list of tuple of (inputs, labels) of length batch_size
            inputs: np.ndarray, dimensions (n_samples, 9), signal data for IMU accel, gyro, and mag
            labels: np.ndarray, dimensions (n_samples, 7), position and quaternion data from mocap

        Returns:
        --------
        (inputs, labels): augmented signal data, augmented labels
        """
        x = []
        y = []
        for (inputs, labels) in data:
            inputs, labels = self._random_crop(inputs, labels)

            n_in, d_in = inputs.shape
            n_out, d_out = labels.shape
            assert n_in == n_out, "Inputs and outputs must have the same number of sequence elements"
            assert d_in == 9, f"Input has dimensionality {d_in} instead of 9"
            assert d_out == 7, f"Output has dimensionality {d_out} instead of 7"

            # Augment XYZ positions
            offset = rng.uniform(-self.position_noise, self.position_noise, size=(1, 3))
            labels[:, 0:3] += offset
            # Augment quaternion sign
            sign = rng.choice([-1, 1])
            labels[:, 4:] *= sign

            accel_noise = rng.normal(loc=0, scale=self.accel_eps, size=(n_in, 3))
            gyro_noise = rng.normal(loc=0, scale=self.gyro_eps, size=(n_in, 3))
            mag_noise = rng.normal(loc=0, scale=self.mag_eps, size=(n_in, 3))

            noise = np.hstack([accel_noise, gyro_noise, mag_noise])
            inputs += noise

            x.append(torch.FloatTensor(inputs))
            y.append(torch.FloatTensor(labels))
        lengths = [len(item) for item in x]
        inds = np.flip(np.argsort(lengths)).copy() # PackedSequence expects lengths from longest to shortest
        lengths = torch.LongTensor(lengths)[inds]

        # Sort by lengths
        x = [x[i] for i in inds]
        y = [y[i] for i in inds]

        packed_inputs = torch.nn.utils.rnn.pack_sequence(x)
        packed_labels = torch.nn.utils.rnn.pack_sequence(y)

        return packed_inputs, packed_labels
