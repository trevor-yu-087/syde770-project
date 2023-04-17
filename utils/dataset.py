import torch
import pandas as pd
import numpy as np
from torch import nn

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
    def __init__(self, position_noise=0.2, accel_eps=0.1, gyro_eps=0.1, mag_eps=0.1, max_input_samples=512, downsample_output_seq=1):
        """
        Parameters:
        -----------
        position_noise: float, limits on uniform distribution [-p, p] to add position offset to mocap
        accel_eps: float, standard deviation on Gaussian noise added to accelerometer channels
        gyro_eps: float, standard deviation on Gaussian noise added to gyroscope channels
        mag_eps: float, standard deviation on Gaussian noise added to mangetometer channels
        max_input_samples: int, maximum number of input samples
        downsample_output_seq: int, factor to downsample output sequence
        """
        self.position_noise = position_noise
        self.accel_eps = accel_eps
        self.gyro_eps = gyro_eps
        self.mag_eps = mag_eps
        self.max_input_samples = max_input_samples
        self.downsample_output_seq = downsample_output_seq
    
    def _random_crop(self, imu, mocap):
        """
        Apply a random crop of the signal of length self.max_input_samples to both inputs and labels, if able to
        Due to targets being a shifted version of decoder inputs, we need to account for one extra timepoint (after downsampling)
        """
        n, d = imu.shape
        ds = self.downsample_output_seq
        max_len = self.max_input_samples + ds
        max_offset = n - max_len

        if max_offset > 0:
            offset = rng.choice(max_offset)
            input_inds = slice(offset, offset + self.max_input_samples)
            output_inds = slice(offset, offset + max_len)
            imu, mocap = imu[input_inds, :], mocap[output_inds, :]
        else:
            cutoff = ds if n % ds == 0 else n % ds
            input_inds = slice(0, n - cutoff)
            imu = imu[input_inds, :]
        if self.downsample_output_seq > 1:
            mocap = mocap[::self.downsample_output_seq, :]
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
            assert np.ceil(n_in / self.downsample_output_seq) + 1 == n_out, f"Downsamping failed, n_in={n_in}; n_out={n_out}"
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
            encoder_inputs.append(torch.FloatTensor(imu))
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


class SmartwatchAugmentTransformer:
    """
    Collate function to apply random augmentations to the data
        - Randomly perturb the mocap positions
        - Randomly flip sign of mocap quaternion
        - Add random noise to IMU channels
        - Random crop to the signal (if possible)
    """
    def __init__(self, position_noise=0.2, accel_eps=0.1, gyro_eps=0.1, mag_eps=0.1, max_input_samples=512, downsample_output_seq=1):
        """
        Parameters:
        -----------
        position_noise: float, limits on uniform distribution [-p, p] to add position offset to mocap
        accel_eps: float, standard deviation on Gaussian noise added to accelerometer channels
        gyro_eps: float, standard deviation on Gaussian noise added to gyroscope channels
        mag_eps: float, standard deviation on Gaussian noise added to mangetometer channels
        max_input_samples: int, maximum number of input samples
        downsample_output_seq: int, factor to downsample output sequence
        """
        self.position_noise = position_noise
        self.accel_eps = accel_eps
        self.gyro_eps = gyro_eps
        self.mag_eps = mag_eps
        self.max_input_samples = max_input_samples
        self.downsample_output_seq = downsample_output_seq

    def _random_crop(self, imu, mocap):
        """
        Apply a random crop of the signal of length self.max_input_samples to both inputs and labels, if able to
        Due to targets being a shifted version of decoder inputs, we need to account for one extra timepoint (after downsampling)
        """
        n, d = imu.shape
        ds = self.downsample_output_seq
        max_len = self.max_input_samples + ds
        max_offset = n - max_len

        if max_offset > 0:
            offset = rng.choice(max_offset)
            input_inds = slice(offset, offset + self.max_input_samples)
            output_inds = slice(offset, offset + max_len)
            imu, mocap = imu[input_inds, :], mocap[output_inds, :]
        else:
            cutoff = ds if n % ds == 0 else n % ds
            input_inds = slice(0, n - cutoff)
            imu = imu[input_inds, :]
        if self.downsample_output_seq > 1:
            mocap = mocap[::self.downsample_output_seq, :]
        return imu, mocap

    def padding_mask(self, input, pad_idx=0, dim=512): 
        # Create mask which marks the zero padding values in the input by a 0
        mask = torch.zeros((dim))
        if input.shape[0] < dim:
            mask[input.shape[0]:] = 1
            return mask.bool()
        #mask = mask.float()

        return mask.bool()


    def lookahead_mask(self, shape):
        # Mask out future entries by marking them with a 1.0
        mask = 1 - torch.tril(torch.ones((shape, shape)))
        mask = mask.masked_fill(mask == 1, float('-inf'))
    
        return mask


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
            assert np.ceil(n_in / self.downsample_output_seq) + 1 == n_out, f"Downsamping failed, n_in={n_in}; n_out={n_out}"
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
            encoder_inputs.append(torch.FloatTensor(imu))
            decoder_inputs.append(torch.FloatTensor(mocap[:-1, :]))
            targets.append(torch.FloatTensor(mocap[1:, :]))

        lengths = [len(item) for item in encoder_inputs]
        inds = np.flip(np.argsort(lengths)).copy()  # PackedSequence expects lengths from longest to shortest
        lengths = torch.LongTensor(lengths)[inds]

        # Sort by lengths
        encoder_inputs = [encoder_inputs[i] for i in inds]
        decoder_inputs = [decoder_inputs[i] for i in inds]
        targets = [targets[i] for i in inds]

        decoder_lengths = [len(item) for item in decoder_inputs]

        # Pad input, if needed
        for i, length in enumerate(lengths):
            if length != self.max_input_samples or decoder_lengths[i] != self.max_input_samples:
                #print("Dim does not equal maximum number of input samples - padding sequence") 
                encoder_inputs[i] = nn.functional.pad(encoder_inputs[i], pad=(0, 0, self.max_input_samples - encoder_inputs[i].shape[0], 0), mode='constant', value=0)
                decoder_inputs[i] = nn.functional.pad(decoder_inputs[i], pad=(0, 0, self.max_input_samples - decoder_inputs[i].shape[0], 0), mode='constant', value=0)
                targets[i] = nn.functional.pad(targets[i], pad=(0, 0, self.max_input_samples - targets[i].shape[0], 0), mode='constant', value=0)


        # Padding mask for encoder
        enc_padding_mask = [self.padding_mask(input=encoder_inputs[i], dim=self.max_input_samples) for i in inds]
        enc_lookahead_mask = [self.lookahead_mask(shape=encoder_inputs[i].shape[0]) for i in inds]
        
        # Padding and look-ahead masks for decoder
        dec_in_padding_mask = [self.padding_mask(input=decoder_inputs[i], dim=self.max_input_samples) for i in inds]
        dec_in_lookahead_mask = [self.lookahead_mask(shape=decoder_inputs[i].shape[0]) for i in inds]


        encoder_inputs = torch.stack(encoder_inputs)
        decoder_inputs = torch.stack(decoder_inputs)
        targets = torch.stack(targets)

        enc_padding_mask = torch.stack(enc_padding_mask)
        enc_lookahead_mask = torch.stack(enc_lookahead_mask)
        
        dec_in_padding_mask = torch.stack(dec_in_padding_mask)
        dec_in_lookahead_mask = torch.stack(dec_in_lookahead_mask)


        collated_data = {
            "encoder_inputs": encoder_inputs,
            "decoder_inputs": decoder_inputs,
            "targets": targets,
            "encoder_padding_mask": enc_padding_mask,
            "decoder_padding_mask": dec_in_padding_mask,
            "decoder_lookahead_mask": dec_in_lookahead_mask,
            "encoder_lookahead_mask": enc_lookahead_mask
        }
        return collated_data

def get_file_lists(
        val_sub_list, 
        test_sub_list,
    ):
    """Get list of files to pass to dataset class
    Parameters:
    -----------
    val_sub_list: list of subject numbers for validation
    test_sub_list: list of subject numbers for testing
    ***Note: subjects '01' - '09' must be entered as strings in the list
    Returns:
    --------
    train_files: list of str filepaths to pre-processed train data
    val_files: list of str filepaths to pre-processed validation data
    test_files: list of str filepaths to pre-processed test data
    """
    import glob
    valid_files = glob.glob("~/scratch/syde770_processed_data/subjects/*/*_full.csv")

    val_subjects = [f"S{n}" for n in val_sub_list]
    val_files = [file for file in valid_files for subject in val_subjects if f"/{subject}/" in file]

    test_subjects = [f"S{n}" for n in test_sub_list]
    test_files = [file for file in valid_files for subject in test_subjects if f"/{subject}/" in file]

    train_files = [file for file in valid_files if file not in set(val_files + test_files)]
    return train_files, val_files, test_files
