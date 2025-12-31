import os
import cv2
import shutil
import numpy as np
import pandas as pd

from tqdm import tqdm

rgb_dirs = '/home/kientt/AFOSR/afosr2022/data/rgb'
rgb_csv = '/home/kientt/AFOSR/afosr2022/data/rgb/data_RGB.csv'
of_dirs = '/home/kientt/AFOSR/afosr2022/data/of'
of_csv = '/home/kientt/AFOSR/afosr2022/data/of/data_OF.csv'
for dir_pathss in os.listdir(rgb_dirs):
    dir_paths = os.path.join(rgb_dirs, dir_pathss)
    for dir_path in os.listdir(dir_paths):
        of_dir = os.path.join(of_dirs, dir_pathss, dir_path)
        print(of_dir)
        os.makedirs(of_dir, exist_ok=True)
        rgb_dir = os.path.join(rgb_dirs, dir_pathss, dir_path)
        print(rgb_dir)
        df = pd.read_csv(rgb_csv)
        clips = df['clip'].tolist()
        labels = df['label'].tolist()
        print('of_dir:', of_dir)
        print('rgb_dir: ',rgb_dir)

        def calc_optical_flow_farneback(frame1, frame2, *args, **kwargs):
            height, width = frame1.shape
            # Optical flow is now calculated
            flow = cv2.calcOpticalFlowFarneback(frame1, frame2, *args, **kwargs)
            # Compute magnite and angle of 2D vector
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Create mask
            hsv_mask = np.zeros((height, width, 3), dtype=np.uint8)
            # Make image saturation to a maximum value
            hsv_mask[..., 1] = 255
            # Set image hue value according to the angle of optical flow
            hsv_mask[..., 0] = ang * 180 / np.pi / 2
            # Set value as per the normalized magnitude of optical flow
            hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            # Return RGB representation
            return cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

        # save video
        pbar = tqdm(zip(clips, labels), total=len(clips))
        for clip, label in pbar:
            cap = cv2.VideoCapture(os.path.join(rgb_dir, clip))
            
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fps = cap.get(cv2.CAP_PROP_FPS)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # if not cap.isOpened() or num_frames != 16:
            #     continue
            pbar.set_description(clip)

            out = cv2.VideoWriter(os.path.join(of_dir, clip), cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
            frame1 = frame2 = None
            for i in range(num_frames):
                _, frame2 = cap.read()
                frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                if i == 0:
                    frame1 = frame2
                    continue
                of_representation = calc_optical_flow_farneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                out.write(of_representation)
                frame1 = frame2
            del frame1, frame2
            cap.release()
            out.release()

        # copy csv
        shutil.copy2(rgb_csv, of_csv)