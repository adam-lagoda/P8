import os
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path
from lib.extract_csv_tensor import extract_data_from_tensorboard
from lib.plot_csv import plot_csv_logs

log_folder = "Torque_Reward_and_with_lost_detection_test_1_1683889158.8175397_1"

# Usage example
rollout = [
    'rollout/ep_len_mean',
    'rollout/ep_rew_mean',
    'rollout/exploration_rate',
    'train/loss'
]

log_dir = Path(__file__).parent.parent / log_folder
output_dir = Path(__file__).parent.parent / log_folder


args = extract_data_from_tensorboard(log_dir, rollout)
plot_csv_logs(*args)
print("test")
