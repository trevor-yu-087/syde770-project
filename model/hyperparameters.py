import torch

BATCH_SIZE = 256
NUM_EPOCH = 500
NUM_WORKERS = 12
ENCODER_LEARNING_RATE = 1e-4
DECODER_LEARNING_RATE = 1e-4
TEACHER_FORCE_RATIO = 1

TRANSFORMER_BATCH_SIZE= 8
TRANSFORMER_LEARNING_RATE = 5e-4
TRANSFORMER_TEACHER_FORCE_RATIO = 0.5 # must use teacher forcing otherwise memory issues - reduced seq len to run with AR
NUM_HEADS=4

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'