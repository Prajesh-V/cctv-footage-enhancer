## ClarityAI Frontend (Phase 2 UI)

- Provides a minimal Next.js-based UI to upload videos, configure Phase 2 options (upscale, ROI), and monitor progress via WebSocket.
- It talks directly to the backend API at /api/jobs/video and /ws/jobs/{job_id} for live updates.

- Prereqs: Node.js (14+), NPM/Yarn, and a running backend at http://localhost:8000.

- Start UI:
  1) cd clarityai/frontend
  2) npm install
  3) npm run dev

- Base backend URL can be controlled by NEXT_PUBLIC_BACKEND_BASE_URL environment variable.
