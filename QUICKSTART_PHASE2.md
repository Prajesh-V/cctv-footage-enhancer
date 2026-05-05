ClarityAI Phase 2 MVP — Video (Quickstart)

This guide gets you from zero to a working Phase 2 MVP: a UI to upload video, run a frame-by-frame processing pipeline with ROI support, and download the enhanced video. It uses the Phase 2 backend with a UI built in Next.js.

Prerequisites
- Python 3.11+ (for the backend)
- Node.js 14+ (for the frontend)
- FFmpeg installed and available in PATH (required for audio muxing)
- Git installed (to clone/pull and manage the repo)
- Optional: Docker + Docker Compose if you prefer containerized local dev

Code layout (relevant parts)
- Backend: clarityai/backend/
- Frontend: clarityai/frontend/
- Test helpers: clarityai/frontend/tools/generate_test_videos.py, clarityai/frontend/scripts/run_tests.py

Step 1 — Install and run the backend (Phase 2)
Option A — Local environment (recommended for quick experiments)
1) Create and activate a Python virtual environment
   - On macOS/Linux:
     python3 -m venv venv
     source venv/bin/activate
   - On Windows:
     python -m venv venv
     .\venv\Scripts\activate
2) Install requirements
   - pip install -r clarityai/backend/requirements.txt
3) Start the API server
   - uvicorn clarityai.backend.main:app --reload --port 8000
   - The backend exposes endpoints at http://localhost:8000

Option B — Docker (if you prefer containerized run)
1) From repo root run:
   - docker-compose up --build
2) Backend runs on http://localhost:8000 and frontend on http://localhost:3000 (if you start the frontend container)

Step 2 — Install and run the frontend (Phase 2 UI)
1) Install dependencies
   - cd clarityai/frontend
   - npm install
2) Start the frontend (ensure BACKEND URL is correct)
   - In Linux/macOS:
     NEXT_PUBLIC_BACKEND_BASE_URL=http://localhost:8000 npm run dev
   - In Windows (PowerShell):
     $env:NEXT_PUBLIC_BACKEND_BASE_URL="http://localhost:8000"; npm run dev
3) Open the UI in your browser: http://localhost:3000

Step 3 — Generate test videos (optional, quick tests)
1) Generate two small test videos for quick testing
   - python clarityai/frontend/tools/generate_test_videos.py
   - This creates sample videos at clarityai/frontend/tests/test1.mp4 and clarityai/frontend/tests/test2.mp4

Step 4 — Test the Phase 2 flow (UI + CLI helper)
Option A — Use the UI
1) In the browser UI, upload a video file from clarityai/frontend/tests/test1.mp4
2) Set upscale_factor (2x or 4x) and ROI (optional)
3) Submit and monitor real-time progress via the WebSocket panel
4) When finished, click Download to save the enhanced video

Option B — Use the test harness (CLI)
1) Ensure the backend is running
2) Run the test harness (downloads must be enabled by the backend in Phase 2)
   - python clarityai/frontend/scripts/run_tests.py
3) Observe backend responses and ensure a final enhanced video is produced

Step 5 — ROI quick tips
- ROI is defined in normalized coordinates [0..1] for x, y, w, h
- Example ROI: {"x":0.1, "y":0.1, "w":0.5, "h":0.5}
- ROI is applied per-frame in the video path; you’ll see a red rectangle overlay in previews

What to expect and troubleshooting
- If FFmpeg is missing, you’ll see an error indicating FFmpeg must be installed. Install FFmpeg and ensure it's in PATH.
- If the VideoWriter cannot initialize, you’ll see a descriptive error in the backend logs. Ensure the output directory is writable and the chosen codec is supported on your OS.
- WebSocket progress should update in near real-time. If not, verify CORS and the browser’s console for WebSocket connection issues.

Next steps (Phase 2 enhancements)
- Improve UI with a small ROI helper (canvas overlay, drag-to-draw ROI) in Phase 3.
- Add a robust Phase 2 video test suite that validates ROI normalization, per-frame processing, and audio muxing.

Notes
- This quickstart assumes a local dev environment. If you want a single command to boot both backend and frontend, we can add a Compose workflow or a small bootstrap script.
- If you’d like, I can also add a one-run script to initialize both servers in a single terminal with proper cross-platform support.
