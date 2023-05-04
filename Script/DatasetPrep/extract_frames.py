import cv2
from tqdm import tqdm


video_paths = [
    r"C:/Users/adaml/Desktop/dataset_yolo_v8/Videos/moving_multi_high_fog.mkv",
    r"C:/Users/adaml/Desktop/dataset_yolo_v8/Videos/moving_single_high_fog.mkv",
    r"C:/Users/adaml/Desktop/dataset_yolo_v8/Videos/moving_single_low_fog.mkv"
    ]

output_folder = r"C:/Users/adaml/Desktop/dataset_yolo_v8/images"


for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(total_frames)):
        
        ret, frame = cap.read()

        if ret:
            folder_name = video_path.split('/')[-1].split('.')[0]
            filename = fr'{output_folder}/{folder_name}/_frame_{i:05d}.jpg'             
            
            cv2.imwrite(filename, frame)
            print(f"{i} frames out of {(total_frames)}")
        else:
            break

cap.release()
