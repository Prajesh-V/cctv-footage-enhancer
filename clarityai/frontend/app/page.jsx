"use client";

import React, { useState, useEffect, useRef } from 'react'

const BACKEND_BASE_URL = (typeof window !== 'undefined' && window.__BACKEND_BASE_URL) || (process?.env?.NEXT_PUBLIC_BACKEND_BASE_URL ?? 'http://localhost:8000')

export default function Page() {
  const [videoFile, setVideoFile] = useState(null)
  const [upscale, setUpscale] = useState(2)
  const [roi, setRoi] = useState({ x: 0.0, y: 0.0, w: 1.0, h: 1.0 })
  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState('idle')
  const [progress, setProgress] = useState(0)
  const [previewImageUrl, setPreviewImageUrl] = useState(null)
  const [ocrResults, setOcrResults] = useState([])
  const [plateResults, setPlateResults] = useState([])
  const wsRef = useRef(null)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [startPos, setStartPos] = useState({ x: 0, y: 0 })

  useEffect(() => {
    if (!jobId) return
    const wsUrl = (BACKEND_BASE_URL.startsWith('https') ? 'wss' : 'ws') + '://' + BACKEND_BASE_URL.replace(/^https?:\/\//, '') + `/ws/jobs/${jobId}`
    const ws = new WebSocket(wsUrl)
    
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data)
        if ('progress' in data) {
          setProgress(data.progress)
          if (data.status) setStatus(data.status)
        } else if (data.status) {
          setStatus(data.status)
        }
      } catch (e) {
        // ignore non-JSON
      }
    }
    
    wsRef.current = ws
    
    // Status Polling Fallback (in case WS closes early)
    const pollInterval = setInterval(async () => {
      if (status === 'completed' || status === 'failed' || status === 'cancelled') {
        clearInterval(pollInterval)
        return
      }
      try {
        const res = await fetch(`${BACKEND_BASE_URL}/api/jobs/${jobId}`)
        if (res.ok) {
          const data = await res.json()
          if (data.status) {
            setStatus(data.status)
            if (data.progress !== undefined) setProgress(data.progress)
          }
        }
      } catch (err) {
        console.error('Polling error:', err)
      }
    }, 3000)

    return () => {
      ws.close()
      clearInterval(pollInterval)
    }
  }, [jobId, status])

  // Fetch OCR/plate results when job completes
  useEffect(() => {
    if (status === 'completed' && jobId) {
      fetch(`${BACKEND_BASE_URL}/api/jobs/${jobId}`)
        .then(res => res.json())
        .then(data => {
          setOcrResults(data.ocr_texts || [])
          setPlateResults(data.plates || [])
        })
        .catch(() => {})
    }
  }, [status, jobId])

  async function handleFileChange(e) {
    const file = e.target.files?.[0] ?? null
    setVideoFile(file)
    setRoi({ x: 0.0, y: 0.0, w: 1.0, h: 1.0 })
    setPreviewImageUrl(null)
    
    if (file) {
      const form = new FormData()
      form.append('video', file)
      try {
        const res = await fetch(`${BACKEND_BASE_URL}/api/utils/preview-frame`, {
          method: 'POST',
          body: form
        })
        if (res.ok) {
          const blob = await res.blob()
          setPreviewImageUrl(URL.createObjectURL(blob))
        }
      } catch (err) {
        console.error('Failed to fetch preview frame:', err)
      }
    }
  }

  function handleCanvasMouseDown(e) {
    if (status !== 'idle' && status !== 'completed' && status !== 'failed') return
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width
    const y = (e.clientY - rect.top) / rect.height
    setIsDrawing(true)
    setStartPos({ x, y })
    setRoi({ x, y, w: 0, h: 0 })
  }

  function handleCanvasMouseMove(e) {
    if (!isDrawing) return
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width
    const y = (e.clientY - rect.top) / rect.height
    const w = Math.max(0, x - startPos.x)
    const h = Math.max(0, y - startPos.y)
    setRoi({ x: startPos.x, y: startPos.y, w: Math.min(w, 1 - startPos.x), h: Math.min(h, 1 - startPos.y) })
  }

  function handleCanvasMouseUp() {
    setIsDrawing(false)
  }

  async function submitVideo() {
    if (!videoFile) return
    const form = new FormData()
    form.append('video', videoFile)
    form.append('upscale_factor', String(upscale))
    form.append('roi', JSON.stringify(roi))

    const res = await fetch(`${BACKEND_BASE_URL}/api/jobs/video`, {
      method: 'POST',
      body: form,
    })
    if (!res.ok) {
      const err = await res.text()
      console.error('Video job failed to start:', err)
      return
    }
    const data = await res.json()
    const id = data.job_id || data.id
    setJobId(id)
    setStatus(data.status || 'in_progress')
    setOcrResults([])
    setPlateResults([])
  }

  async function downloadResult() {
    if (!jobId) return
    const res = await fetch(`${BACKEND_BASE_URL}/api/jobs/${jobId}/result`)
    if (res.ok) {
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `enhanced_${jobId}.mp4`
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } else {
      console.error('Failed to fetch result')
    }
  }

  return (
    <main className="main-container">
      <h1>ClarityAI</h1>

      <div className="exhibit-card">
        <div className="input-group">
          <label>I. Select Footage</label>
          <input
            type="file"
            accept="video/*"
            onChange={handleFileChange}
          />
          {videoFile && (
            <p style={{ marginTop: '0.5rem', fontSize: '0.8rem', color: 'var(--gold)', letterSpacing: '0.1em' }}>
              EXHIBIT: {videoFile.name}
            </p>
          )}
        </div>

        {videoFile && (
          <div className="input-group" style={{ marginTop: '1rem' }}>
            <label>II. Draw ROI on Preview (drag to select region)</label>
            <div className="framed-preview" style={{ width: '100%', maxWidth: '640px' }}>
              {previewImageUrl ? (
                <img
                  src={previewImageUrl}
                  style={{ width: '100%', display: 'block' }}
                  alt="Video Preview"
                />
              ) : (
                <video
                  ref={videoRef}
                  src={videoFile ? URL.createObjectURL(videoFile) : ''}
                  style={{ width: '100%', display: 'block' }}
                  muted
                  preload="metadata"
                />
              )}
              <canvas
                ref={canvasRef}
                width={640}
                height={360}
                onMouseDown={handleCanvasMouseDown}
                onMouseMove={handleCanvasMouseMove}
                onMouseUp={handleCanvasMouseUp}
                onMouseLeave={handleCanvasMouseUp}
                style={{
                  position: 'absolute',
                  top: '10px',
                  left: '10px',
                  width: 'calc(100% - 20px)',
                  height: 'calc(100% - 20px)',
                  cursor: 'crosshair',
                }}
              />
              {roi.w > 0 && roi.h > 0 && (
                <div
                  style={{
                    position: 'absolute',
                    left: `calc(10px + ${roi.x * (100 - (2000/640)) }%)`,
                    top: `calc(10px + ${roi.y * (100 - (2000/360)) }%)`,
                    width: `${roi.w * (100 - (2000/640))}%`,
                    height: `${roi.h * (100 - (2000/360))}%`,
                    border: '2px solid var(--gold)',
                    boxShadow: '0 0 15px var(--gold-glow)',
                    pointerEvents: 'none',
                  }}
                />
              )}
            </div>
            <p style={{ fontSize: '0.7rem', color: 'var(--pewter)', marginTop: '1rem', letterSpacing: '0.1em' }}>
              COORDINATES: x={roi.x.toFixed(2)}, y={roi.y.toFixed(2)}, w={roi.w.toFixed(2)}, h={roi.h.toFixed(2)}
            </p>
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '2rem', marginTop: '1.5rem' }}>
          <div className="input-group">
            <label>III. Upscale Factor</label>
            <select value={upscale} onChange={e => setUpscale(parseInt(e.target.value))}>
              <option value={2}>2x Standard</option>
              <option value={4}>4x Ultra-Luxe</option>
            </select>
          </div>
        </div>

        <div style={{ marginTop: '2rem', textAlign: 'center' }}>
          <button
            onClick={submitVideo}
            disabled={!videoFile || (status !== 'idle' && status !== 'completed' && status !== 'failed')}
          >
            {status === 'idle' ? 'Begin Enhancement' : 'New Sequence'}
          </button>
        </div>
      </div>

      {(jobId || status !== 'idle') && (
        <div className="exhibit-card" style={{ borderLeft: '4px solid var(--gold)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h3 style={{ marginBottom: '0.5rem' }}>Current Operation</h3>
              <p style={{ fontSize: '0.7rem', color: 'var(--pewter)', letterSpacing: '0.1em' }}>SERIAL: {jobId || 'PENDING'}</p>
            </div>
            <div className={`status-badge ${status === 'completed' ? 'status-completed' : ''}`}>
              {status}
            </div>
          </div>

          <div className="progress-container">
            <div className="progress-bar" style={{ width: `${progress}%` }}></div>
          </div>
          <div style={{ textAlign: 'right', marginTop: '0.5rem', fontSize: '1rem', fontWeight: 600, color: 'var(--gold)', fontFamily: 'var(--font-display)' }}>
            {progress}%
          </div>

          {(status === 'completed' || status === 'cancelled' || progress === 100) && (
            <div style={{ marginTop: '2.5rem', textAlign: 'center', animation: 'fadeIn 1s ease-out' }}>
              <button 
                onClick={downloadResult} 
                className="download-btn"
                style={{ 
                  borderStyle: 'double', 
                  borderWidth: '4px',
                  boxShadow: '0 0 20px var(--gold-glow)',
                  background: 'rgba(212, 175, 55, 0.1)'
                }}
              >
                {status === 'completed' ? 'Export Enhanced Evidence' : 'Fetch Latest Result'}
              </button>
              {status !== 'completed' && status !== 'cancelled' && (
                <p style={{ fontSize: '0.7rem', color: 'var(--pewter)', marginTop: '0.5rem' }}>
                  Processing finalized. Preparing download...
                </p>
              )}
            </div>
          )}
        </div>
      )}

      {status === 'completed' && (ocrResults.length > 0 || plateResults.length > 0) && (
        <div className="exhibit-card">
          <h3 style={{ marginBottom: '2rem' }}>Curated Results</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3rem' }}>
            {plateResults.length > 0 && (
              <div>
                <h4 style={{ fontFamily: 'var(--font-display)', color: 'var(--pewter)', fontSize: '0.8rem', letterSpacing: '0.2em', textTransform: 'uppercase', marginBottom: '1.5rem' }}>License Plate Gallery</h4>
                {plateResults.map((p, i) => (
                  <div key={i} className="result-item">
                    <h4>Exhibit {i + 1}</h4>
                    <p style={{ fontSize: '0.9rem' }}>Confidence: {(p.confidence * 100).toFixed(0)}%</p>
                    <p style={{ fontSize: '0.7rem', color: 'var(--pewter)', marginTop: '0.25rem' }}>Grid Ref: [{p.bbox?.join(', ')}]</p>
                  </div>
                ))}
              </div>
            )}
            {ocrResults.length > 0 && (
              <div>
                <h4 style={{ fontFamily: 'var(--font-display)', color: 'var(--pewter)', fontSize: '0.8rem', letterSpacing: '0.2em', textTransform: 'uppercase', marginBottom: '1.5rem' }}>Textual Artifacts</h4>
                {ocrResults.map((t, i) => (
                  <div key={i} className="result-item">
                    <h4>Artifact {i + 1}</h4>
                    <p style={{ fontSize: '1.1rem', color: 'var(--champagne)' }}>"{t.text}"</p>
                    <p style={{ fontSize: '0.7rem', color: 'var(--pewter)', marginTop: '0.25rem' }}>Quality Index: {(t.confidence * 100).toFixed(0)}%</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </main>
  )
}
