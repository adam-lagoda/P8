import os
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pathlib import Path

def extract_csv_from_tensorboard(log_dir, output_dir):
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Get all available tags
    tags = event_acc.Tags()

    for tag in tags['scalars']:
        csv_file_path = os.path.join(output_dir, tag + '.csv')

        # Open CSV file for writing
        with open(csv_file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            # Write CSV header
            writer.writerow(['step', 'wall_time', 'value'])

            # Extract data for each scalar event and write to CSV
            for scalar_event in event_acc.Scalars(tag):
                writer.writerow([scalar_event.step, scalar_event.wall_time, scalar_event.value])

        print(f"CSV file saved for tag '{tag}': {csv_file_path}")

# Usage example
log_dir = Path(__file__).parent.joinpath("torq energy cont at lost detec\Torque_Reward_and_No_lost_detection_test_1_1683801433.881646_1")
output_dir = "csv/files"
extract_csv_from_tensorboard(log_dir, output_dir)
