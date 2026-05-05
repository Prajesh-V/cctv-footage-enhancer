import numpy as np
import cv2
import os

def generate_test_video(path: str, duration_sec: int = 3, fps: int = 24, w: int = 320, h: int = 240):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    total_frames = duration_sec * fps
    for i in range(total_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Moving white square
        sz = 40
        x = int((i / total_frames) * (w - sz))
        y = int((i / total_frames) * (h - sz))
        cv2.rectangle(img, (x, y), (x + sz, y + sz), (255, 255, 255), -1)
        out.write(img)
    out.release()

if __name__ == '__main__':
    os.makedirs('clarityai/frontend/tests', exist_ok=True)
    generate_test_video('clarityai/frontend/tests/test1.mp4')
    generate_test_video('clarityai/frontend/tests/test2.mp4', duration_sec=5)
