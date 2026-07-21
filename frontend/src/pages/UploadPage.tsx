import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { analyzeAudio } from '../api/client'

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    try {
      const results = await analyzeAudio(file)
      sessionStorage.setItem('results', JSON.stringify(results))
      sessionStorage.setItem('filename', file.name)
      navigate('/results')
    } catch (err) {
      console.error('Analysis failed:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col items-center gap-6">
      <div
        className="w-full border-2 border-dashed border-gray-300 rounded-lg p-12 text-center hover:border-blue-400 transition cursor-pointer"
        onClick={() => inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".wav,.mp3,.flac"
          className="hidden"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        {file ? (
          <p className="text-gray-700">{file.name}</p>
        ) : (
          <p className="text-gray-500">Click to upload audio (WAV, MP3, FLAC)</p>
        )}
      </div>
      <button
        onClick={handleUpload}
        disabled={!file || loading}
        className="px-6 py-2 bg-blue-600 text-white rounded-lg disabled:bg-gray-300 hover:bg-blue-700 transition"
      >
        {loading ? 'Analyzing...' : 'Analyze'}
      </button>
    </div>
  )
}
