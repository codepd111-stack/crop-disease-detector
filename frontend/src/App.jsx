import { useState, useRef } from 'react'
import axios from 'axios'

const API_URL = 'http://localhost:8000'

function ConfidenceBar({ label, confidence, isTop }) {
  return (
    <div style={{ marginBottom: '10px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
        <span style={{ fontSize: '13px', color: isTop ? '#166534' : '#555', fontWeight: isTop ? 600 : 400 }}>
          {label.replace(/___/g, ' — ').replace(/_/g, ' ')}
        </span>
        <span style={{ fontSize: '13px', fontWeight: 600, color: isTop ? '#166534' : '#555' }}>
          {confidence}%
        </span>
      </div>
      <div style={{ background: '#e5e7eb', borderRadius: '99px', height: '8px', overflow: 'hidden' }}>
        <div style={{
          height: '100%',
          width: `${confidence}%`,
          background: isTop ? 'linear-gradient(90deg, #16a34a, #4ade80)' : '#9ca3af',
          borderRadius: '99px',
          transition: 'width 0.8s ease'
        }} />
      </div>
    </div>
  )
}

export default function App() {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const fileRef = useRef()

  const handleFile = (file) => {
    if (!file || !file.type.startsWith('image/')) return
    setImage(file)
    setPreview(URL.createObjectURL(file))
    setResult(null)
    setError(null)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  const handleSubmit = async () => {
    if (!image) return
    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', image)

    try {
      const res = await axios.post(`${API_URL}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setResult(res.data)
    } catch (err) {
      setError('Something went wrong. Make sure the API server is running.')
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setImage(null)
    setPreview(null)
    setResult(null)
    setError(null)
  }

  const isHealthy = result?.disease?.toLowerCase().includes('healthy')

  return (
    <div style={{ minHeight: '100vh', padding: '24px 16px', maxWidth: '520px', margin: '0 auto' }}>

      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: '32px' }}>
        <div style={{ fontSize: '40px', marginBottom: '8px' }}>🌿</div>
        <h1 style={{ fontSize: '24px', fontWeight: 700, color: '#14532d' }}>Crop Disease Detector</h1>
        <p style={{ color: '#6b7280', fontSize: '14px', marginTop: '6px' }}>
          Upload a leaf photo to diagnose disease and get treatment advice
        </p>
      </div>

      {/* Upload Area */}
      {!result && (
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileRef.current.click()}
          style={{
            border: `2px dashed ${dragging ? '#16a34a' : '#bbf7d0'}`,
            borderRadius: '16px',
            background: dragging ? '#f0fdf4' : 'white',
            padding: '32px 16px',
            textAlign: 'center',
            cursor: 'pointer',
            transition: 'all 0.2s',
            marginBottom: '16px'
          }}
        >
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            capture="environment"
            style={{ display: 'none' }}
            onChange={(e) => handleFile(e.target.files[0])}
          />
          {preview ? (
            <img
              src={preview}
              alt="preview"
              style={{ maxHeight: '240px', borderRadius: '12px', objectFit: 'cover', width: '100%' }}
            />
          ) : (
            <>
              <div style={{ fontSize: '48px', marginBottom: '12px' }}>📷</div>
              <p style={{ fontWeight: 600, color: '#15803d', fontSize: '15px' }}>
                Tap to upload or take a photo
              </p>
              <p style={{ color: '#9ca3af', fontSize: '13px', marginTop: '4px' }}>
                JPG, PNG, WEBP supported
              </p>
            </>
          )}
        </div>
      )}

      {/* Analyse Button */}
      {preview && !result && (
        <button
          onClick={handleSubmit}
          disabled={loading}
          style={{
            width: '100%',
            padding: '14px',
            background: loading ? '#86efac' : 'linear-gradient(135deg, #16a34a, #15803d)',
            color: 'white',
            border: 'none',
            borderRadius: '12px',
            fontSize: '16px',
            fontWeight: 600,
            cursor: loading ? 'not-allowed' : 'pointer',
            marginBottom: '12px',
            transition: 'all 0.2s'
          }}
        >
          {loading ? '🔍 Analysing...' : '🔬 Analyse Leaf'}
        </button>
      )}

      {/* Error */}
      {error && (
        <div style={{
          background: '#fef2f2', border: '1px solid #fecaca',
          borderRadius: '12px', padding: '14px', color: '#dc2626',
          fontSize: '14px', marginBottom: '16px'
        }}>
          ⚠️ {error}
        </div>
      )}

      {/* Result Card */}
      {result && (
        <div style={{ background: 'white', borderRadius: '20px', overflow: 'hidden', boxShadow: '0 4px 24px rgba(0,0,0,0.08)' }}>

          {/* Result Header */}
          <div style={{
            background: isHealthy ? 'linear-gradient(135deg, #16a34a, #15803d)' : 'linear-gradient(135deg, #dc2626, #b91c1c)',
            padding: '20px',
            color: 'white'
          }}>
            <div style={{ fontSize: '32px', marginBottom: '8px' }}>{isHealthy ? '✅' : '🦠'}</div>
            <h2 style={{ fontSize: '18px', fontWeight: 700, marginBottom: '4px' }}>{result.disease}</h2>
            <p style={{ fontSize: '14px', opacity: 0.9 }}>
              {isHealthy ? 'Your crop appears healthy!' : `Confidence: ${result.confidence}%`}
            </p>
          </div>

          <div style={{ padding: '20px' }}>

            {/* Uploaded image */}
            <img
              src={preview}
              alt="uploaded leaf"
              style={{ width: '100%', borderRadius: '12px', maxHeight: '200px', objectFit: 'cover', marginBottom: '20px' }}
            />

            {/* Top 3 confidence bars */}
            <h3 style={{ fontSize: '14px', fontWeight: 600, color: '#374151', marginBottom: '12px' }}>
              Top Predictions
            </h3>
            {result.top3.map((item, i) => (
              <ConfidenceBar
                key={i}
                label={item.class}
                confidence={item.confidence}
                isTop={i === 0}
              />
            ))}

            {/* Treatment */}
            {!isHealthy && (
              <div style={{
                background: '#fefce8', border: '1px solid #fde68a',
                borderRadius: '12px', padding: '16px', marginTop: '20px'
              }}>
                <h3 style={{ fontSize: '14px', fontWeight: 600, color: '#92400e', marginBottom: '8px' }}>
                  💊 Treatment Recommendation
                </h3>
                <p style={{ fontSize: '14px', color: '#78350f', lineHeight: '1.6' }}>
                  {result.treatment}
                </p>
              </div>
            )}

            {/* Analyse another */}
            <button
              onClick={reset}
              style={{
                width: '100%', marginTop: '20px', padding: '12px',
                background: '#f3f4f6', border: 'none', borderRadius: '12px',
                fontSize: '15px', fontWeight: 600, color: '#374151', cursor: 'pointer'
              }}
            >
              🔄 Analyse Another Leaf
            </button>
          </div>
        </div>
      )}
    </div>
  )
}