import torch

BATCH_SIZE = 4
NUM_EPOCH = 5
ENCODER_LEARNING_RATE = 1e-4
DECODER_LEARNING_RATE = 1e-4
TEACHER_FORCE_RATIO = 0.5

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'