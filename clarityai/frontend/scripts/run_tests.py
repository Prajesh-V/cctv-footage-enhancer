import os
import time
import requests

BASE = os.environ.get('BACKEND_BASE', 'http://localhost:8000')

def upload_video(path, upscale=2, roi=None):
    files = {'video': open(path, 'rb')}
    data = {
        'upscale_factor': str(upscale),
        'roi': roi if roi else '{}'
    }
    r = requests.post(f"{BASE}/api/jobs/video", files=files, data=data)
    return r.json()

def main():
    tests_dir = 'clarityai/frontend/tests'
    videos = [os.path.join(tests_dir, f) for f in os.listdir(tests_dir) if f.endswith('.mp4')]
    for v in videos:
        print('Uploading', v)
        res = upload_video(v)
        print('Response:', res)
        time.sleep(1)

if __name__ == '__main__':
    main()
