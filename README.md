# Applications of Deep Learned Inertial Odometry for Generation of Shoulder Physiotherapy Exercise Kinematics
Sequence to sequence generation of precise kinematic data from sequential IMU data. 

## Generation and Structure of Dataset JSON
Dataset contains a total of 84 subjects performing nine shoulder physiotherapy exercises in a 'correct' and 'incorrect' way. Data was processed and temporally synchronized into one file containing time, positional X, Y, Z, quaternion orientations, and 3 channel accelerometer, gyroscope, and magnetometer data. See notebooks for data preprocessing techniques used on raw data.

The data JSON is a dictionary of file paths to the processed data, stratified by subject for training, validation, and testing:

- Total subjects: 84
    - Test: 14 subjects
    - Train/Val: 70 subjects
        - Train: 60 subjects
            - 5-fold: 12 subjects/fold
        - Val: 10 subjects

Data splits required for training/testing include 'train', 'val', 'test'. The 'train' split can also be further split into train/fold sets for k-fold cross validation.

See `013 - Generate data json.ipynb` to generate the data JSON on local machine. 

Example of data JSON:
```
{
    "train": [
        <path_to_sub_1_data_file_1>,
        <path_to_sub_1_data_file_2>,
        ...
        <path_to_sub_1_data_file_n>,
        <path_to_sub_2_data_file_1>,
        <path_to_sub_2_data_file_2>,
        ...
        <path_to_sub_2_data_file_n>,
        
        ...

        <path_to_sub_k_data_file_1>,
        <path_to_sub_k_data_file_2>,
        ...
        <path_to_sub_k_data_file_n>,
    ]
    "val: [
        <path_to_sub_n_data_file_1>,
        <path_to_sub_n_data_file_2>,
        ...
        <path_to_sub_n_data_file_n>,
    ]
    "test: [
        <path_to_sub_n_data_file_1>,
        <path_to_sub_n_data_file_2>,
        ...
        <path_to_sub_n_data_file_n>,
    ]
}
```

## How to Run:
1) Create new environment (Python 3.10) with the following mandatory packages:
    ```
    torch
    pandas
    numpy
    typer
    tensorboard
    matplotlib
    optuna
    ```
    Or use:
    ```console
    conda env create -f environment.yml
    ```
2) Prepare data.json file as outlined above
3) To run:
    - Activate corresponding virtual environment
    - Navigate to repository directory
    - Training:
        ```
        python run.py run <path_to_data_json> <path_to_output_directory> <model> 
        ```
    - Tuning:
        ```
        python tune.py 
        ```
        - Note: Edit global variables in `tune.py` for paths and model
    - Testing
        ```
        python run.py run <path_to_data_json> <path_to_checkpoints_dir> <model> 
        ```
        Note: checkpoints directory (output from training/tuning) will contain folders:
        - best
        - checkpoint
        - tensorboard

### Additional Parameters
Other additional parameters can be set by the user when training/testing:

**Training**
```
REQUIRED:
data_json: Path
save_dir: Path
model: str [ronin, lstm, cnn_lstm, transformer, cnn_transformer]

OPTIONAL:
seq_len: int                            Default: 32
teacher_force_ratio: float [0.0:1.0]    Default: 0.5
dynamic_tf: bool                        Default: False
min_tf_ratio: float                     Default: 0.5
tf_decay: float                         Default: 0.01
enable_checkpoints: bool                Default: False
```

**Testing**
```
REQUIRED:
data_json: Path
save_dir: Path
model: str [ronin, lstm, cnn_lstm, transformer, cnn_transformer]

OPTIONAL:
seq_len: int                            Default: 32
```

## Contributors:
- Trevor Yu (trevor.yu@uwaterloo.ca)
- Jonathan Chu (jh3chu@uwaterloo.ca)
- Tia Tuinstra (ttuinstr@uwaterloo.ca)