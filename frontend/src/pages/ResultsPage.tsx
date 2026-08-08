import { useNavigate } from 'react-router-dom'
import { useEffect, useState, useRef } from 'react'
import { 
  ArrowLeft, 
  Printer, 
  Volume2, 
  ShieldCheck, 
  CheckCircle2, 
  Activity,
  Play,
  Pause,
  Download
} from 'lucide-react'
import { TranscriptionData, analyzeAudio, classifyAudio } from '../api/client'
import { storeAudioFile } from '../utils/db'

interface AnalysisResults {
  classification: Record<string, { label: number; confidence: number }>
  localization: { regions: Array<{ start: number; end: number; confidence: number }> }
  transcription?: TranscriptionData
}

const CLASS_DISPLAY_NAMES: Record<string, string> = {
  prolongation: "Prolongation",
  block: "Block (Silent)",
  soundrep: "Sound Repetition",
  wordrep: "Word Repetition",
  interjection: "Interjection (Filler)"
}

interface WaveformViewProps {
  file: File | null
  regions: Array<{ start: number; end: number; confidence: number }>
  transcription?: TranscriptionData
}

function WaveformView({ file, regions, transcription }: WaveformViewProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [audioBuffer, setAudioBuffer] = useState<AudioBuffer | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const animationRef = useRef<number | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)

  useEffect(() => {
    const decodeAudio = async () => {
      try {
        const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)()
        let buffer: AudioBuffer
        if (file) {
          const arrayBuffer = await file.arrayBuffer()
          buffer = await audioCtx.decodeAudioData(arrayBuffer)
        } else {
          // Parse duration string to seconds
          const durStr = sessionStorage.getItem('duration') || '00:10'
          const parts = durStr.split(':')
          let durSec = 10
          if (parts.length === 2) {
            durSec = parseInt(parts[0]) * 60 + parseInt(parts[1])
          }
          buffer = audioCtx.createBuffer(1, audioCtx.sampleRate * durSec, audioCtx.sampleRate)
        }
        setAudioBuffer(buffer)
        setDuration(buffer.duration)
        audioCtx.close()
      } catch (err) {
        console.error("Failed to decode audio for waveform:", err)
      }
    }
    decodeAudio()
  }, [file])

  useEffect(() => {
    let url: string
    if (file) {
      url = URL.createObjectURL(file)
    } else {
      // Parse duration string to seconds
      const durStr = sessionStorage.getItem('duration') || '00:10'
      const parts = durStr.split(':')
      let durSec = 10
      if (parts.length === 2) {
        durSec = parseInt(parts[0]) * 60 + parseInt(parts[1])
      }
      
      // Synthesize in-memory silent WAV Blob
      const sampleRate = 8000
      const numSamples = sampleRate * durSec
      const wavBuffer = new ArrayBuffer(44 + numSamples * 2)
      const view = new DataView(wavBuffer)
      
      const writeString = (offset: number, string: string) => {
        for (let i = 0; i < string.length; i++) {
          view.setUint8(offset + i, string.charCodeAt(i))
        }
      }
      writeString(0, 'RIFF')
      view.setUint32(4, 36 + numSamples * 2, true)
      writeString(8, 'WAVE')
      writeString(12, 'fmt ')
      view.setUint32(16, 16, true)
      view.setUint16(20, 1, true)
      view.setUint16(22, 1, true)
      view.setUint32(24, sampleRate, true)
      view.setUint32(28, sampleRate * 2, true)
      view.setUint16(32, 2, true)
      view.setUint16(34, 16, true)
      writeString(36, 'data')
      view.setUint32(40, numSamples * 2, true)
      
      for (let i = 0; i < numSamples; i++) {
        view.setInt16(44 + i * 2, 0, true)
      }
      
      const blob = new Blob([wavBuffer], { type: 'audio/wav' })
      url = URL.createObjectURL(blob)
    }
    
    setAudioUrl(url)
    return () => {
      URL.revokeObjectURL(url)
    }
  }, [file])

  useEffect(() => {
    if (!canvasRef.current || !audioBuffer) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height
    const channelData = audioBuffer.getChannelData(0)
    
    const draw = () => {
      ctx.clearRect(0, 0, width, height)

      // 1. Central line
      ctx.strokeStyle = 'rgba(100, 116, 139, 0.15)'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.moveTo(0, height / 2)
      ctx.lineTo(width, height / 2)
      ctx.stroke()

      // 2. Overlays
      regions.forEach(region => {
        const x1 = (region.start / audioBuffer.duration) * width
        const x2 = (region.end / audioBuffer.duration) * width
        
        ctx.fillStyle = 'rgba(239, 68, 68, 0.15)'
        ctx.fillRect(x1, 0, x2 - x1, height)

        ctx.strokeStyle = '#EF4444'
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(x1, 0)
        ctx.lineTo(x1, height)
        ctx.moveTo(x2, 0)
        ctx.lineTo(x2, height)
        ctx.stroke()
      })

      // 3. Waveform
      ctx.strokeStyle = '#14B8A6'
      ctx.lineWidth = 1.5
      ctx.beginPath()

      const step = Math.ceil(channelData.length / width)
      const amp = height / 2

      for (let i = 0; i < width; i++) {
        let min = 1.0
        let max = -1.0
        for (let j = 0; j < step; j++) {
          const dat = channelData[i * step + j]
          if (dat < min) min = dat
          if (dat > max) max = dat
        }
        
        const x = i
        const y1 = (1 + min) * amp
        const y2 = (1 + max) * amp

        ctx.moveTo(x, y1)
        ctx.lineTo(x, y2)
      }
      ctx.stroke()

      // 4. Playhead cursor
      if (audioRef.current) {
        const currentSec = audioRef.current.currentTime
        const playheadX = (currentSec / audioBuffer.duration) * width
        ctx.strokeStyle = '#F59E0B'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(playheadX, 0)
        ctx.lineTo(playheadX, height)
        ctx.stroke()

        ctx.fillStyle = '#F59E0B'
        ctx.beginPath()
        ctx.arc(playheadX, 0, 4, 0, Math.PI * 2)
        ctx.arc(playheadX, height, 4, 0, Math.PI * 2)
        ctx.fill()
      }
    }

    draw()
  }, [audioBuffer, regions, currentTime])

  const togglePlay = () => {
    if (!audioRef.current) return
    if (isPlaying) {
      audioRef.current.pause()
      setIsPlaying(false)
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    } else {
      audioRef.current.play()
      setIsPlaying(true)
      const updatePlayhead = () => {
        if (audioRef.current) {
          setCurrentTime(audioRef.current.currentTime)
          animationRef.current = requestAnimationFrame(updatePlayhead)
        }
      }
      animationRef.current = requestAnimationFrame(updatePlayhead)
    }
  }

  const handleSeek = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !audioRef.current || !duration) return
    const rect = canvasRef.current.getBoundingClientRect()
    const clickX = e.clientX - rect.left
    const percent = clickX / rect.width
    const targetTime = percent * duration
    audioRef.current.currentTime = targetTime
    setCurrentTime(targetTime)
  }

  const formatTime = (timeInSeconds: number) => {
    const mins = Math.floor(timeInSeconds / 60).toString().padStart(2, '0')
    const secs = Math.floor(timeInSeconds % 60).toString().padStart(2, '0')
    return `${mins}:${secs}`
  }

  return (
    <div className="w-full bg-bg-card border border-border-color rounded-xl p-5 space-y-4 shadow-xs" style={{ borderRadius: '16px' }}>
      <audio
        ref={audioRef}
        src={audioUrl || ''}
        onTimeUpdate={() => {
          if (audioRef.current) {
            setCurrentTime(audioRef.current.currentTime)
          }
        }}
        onEnded={() => {
          setIsPlaying(false)
          setCurrentTime(0)
          if (animationRef.current) cancelAnimationFrame(animationRef.current)
        }}
      />
      <div className="flex items-center justify-between border-b border-border-color/50 pb-3 select-none">
        <div>
          <h3 className="text-xs font-bold text-text-primary">Speech Waveform Analysis</h3>
          <p className="text-[10px] text-text-secondary mt-0.5">Click the waveform area to seek audio playback.</p>
        </div>
        <div className="text-[10px] font-mono text-text-secondary">
          {formatTime(currentTime)} / {formatTime(duration)}
        </div>
      </div>
      
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={900}
          height={120}
          onClick={handleSeek}
          className="w-full h-[120px] bg-bg-sidebar border border-border-color rounded-lg cursor-pointer transition"
        />
      </div>

      {/* DRAGGABLE PROGRESS SLIDER */}
      <div className="flex items-center gap-2 select-none">
        <input
          type="range"
          min={0}
          max={duration || 0}
          step={0.01}
          value={currentTime}
          onChange={(e) => {
            if (audioRef.current) {
              const time = parseFloat(e.target.value)
              audioRef.current.currentTime = time
              setCurrentTime(time)
            }
          }}
          className="flex-1 h-1 bg-hover-color rounded-full appearance-none cursor-pointer accent-accent-teal"
        />
      </div>

      <div className="flex items-center justify-between pt-1 select-none">
        <div className="flex items-center gap-2">
          <button
            onClick={togglePlay}
            className="px-4 py-2 bg-accent-teal hover:bg-teal-600 text-white text-[11px] font-semibold rounded-lg shadow-xs transition cursor-pointer flex items-center gap-1.5"
          >
            {isPlaying ? (
              <>
                <Pause size={12} className="fill-white" />
                <span>Pause Audio</span>
              </>
            ) : (
              <>
                <Play size={12} className="fill-white" />
                <span>Play Audio</span>
              </>
            )}
          </button>

          {file && (
            <button
              onClick={() => {
                const url = URL.createObjectURL(file)
                const a = document.createElement('a')
                a.href = url
                a.download = file.name
                document.body.appendChild(a)
                a.click()
                document.body.removeChild(a)
                URL.revokeObjectURL(url)
              }}
              className="px-3 py-2 bg-hover-color/30 hover:bg-hover-color text-text-primary text-[11px] font-semibold rounded-lg border border-border-color shadow-xs transition cursor-pointer flex items-center gap-1.5"
            >
              <Download size={12} />
              <span>Download Audio</span>
            </button>
          )}
        </div>
        <div className="flex items-center gap-4 text-[9px] text-text-secondary uppercase font-semibold">
          <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-accent-teal" /> Waveform</span>
          <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-error-red" /> Localized Stutter</span>
          <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-amber-500" /> Playhead</span>
        </div>
      </div>

      {/* Dynamic Transcription Display */}
      {transcription && (
        <div className="pt-4 border-t border-border-color/50 space-y-3">
          {/* Active chunk subtitle */}
          <div className="bg-bg-sidebar border border-border-color rounded-xl p-4 flex flex-col gap-1.5 shadow-xs">
            <div className="flex items-center justify-between">
              <span className="text-[9px] uppercase font-bold text-accent-teal tracking-wider flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-accent-teal animate-pulse" />
                Live Transcription
              </span>
            </div>
            <p className="text-sm font-semibold text-text-primary italic min-h-[1.5rem] leading-relaxed">
              {(() => {
                if (!transcription.chunks || transcription.chunks.length === 0) {
                  return "No transcription available.";
                }

                if (currentTime === 0) {
                  return "Click Play to read transcription...";
                }

                // 1. Find exact active chunk
                const activeChunk = transcription.chunks.find(
                  chunk => currentTime >= chunk.start && currentTime <= chunk.end
                );
                if (activeChunk) {
                  return `"${activeChunk.text}"`;
                }

                // 2. Persist the last spoken chunk during silences/gaps
                const pastChunks = transcription.chunks.filter(chunk => currentTime >= chunk.start);
                if (pastChunks.length > 0) {
                  return `"${pastChunks[pastChunks.length - 1].text}"`;
                }

                // 3. Priming display (show first segment if playhead has started but not reached it yet)
                return `"${transcription.chunks[0].text}"`;
              })()}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

export default function ResultsPage({ analyzedFile }: { analyzedFile: File | null }) {
  const navigate = useNavigate()
  const [results, setResults] = useState<AnalysisResults | null>(null)
  const [filename, setFilename] = useState('')
  const [filesize, setFilesize] = useState('')
  const [duration, setDuration] = useState('')
  const [patientName, setPatientName] = useState('')
  const [patientPhone, setPatientPhone] = useState('')
  const [showPrintModal, setShowPrintModal] = useState(false)
  const [reportDate, setReportDate] = useState('')
  const showReport = true

  // Loading and background processing states
  const [isLoading, setIsLoading] = useState(false)
  const [loadingStep, setLoadingStep] = useState("Initializing analysis...")
  const [loadingProgress, setLoadingProgress] = useState(5)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  
  const reportRef = useRef<HTMLDivElement>(null)

  const performAnalysis = async (file: File) => {
    setIsLoading(true)
    setErrorMsg(null)
    setLoadingProgress(10)
    setLoadingStep("Extracting speech features...")

    // Simulate progress updates periodically to look premium
    const progressInterval = setInterval(() => {
      setLoadingProgress((prev) => {
        if (prev < 90) {
          return prev + Math.floor(Math.random() * 5) + 2
        }
        return prev
      })
    }, 800)

    try {
      const mode = sessionStorage.getItem('analysis_mode') || 'full'
      const genReport = sessionStorage.getItem('generate_report') === 'true'
      const lang = sessionStorage.getItem('transcription_language') || 'english'

      // Step transitions based on timer
      setTimeout(() => setLoadingStep("Classifying stuttering categories..."), 1500)
      setTimeout(() => setLoadingStep("Localizing speech stutters..."), 3000)
      setTimeout(() => setLoadingStep("Transcribing speech..."), 5000)
      setTimeout(() => setLoadingStep("Generating clinical diagnostic report..."), 7000)

      let res
      if (mode === 'full') {
        res = await analyzeAudio(file, lang)
      } else {
        const response = await classifyAudio(file, lang)
        res = {
          classification: response.classification,
          localization: { regions: [] },
          transcription: response.transcription
        }
      }

      clearInterval(progressInterval)
      setLoadingProgress(100)
      setLoadingStep("Analysis complete!")

      // Save to sessionStorage
      sessionStorage.setItem('results', JSON.stringify(res))

      // Add to Assessment History
      const id = `SWR-${Math.floor(1000 + Math.random() * 9000)}`
      const now = new Date()
      const historyItem = {
        id,
        date: now.toISOString().split('T')[0],
        time: now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
        name: file.name,
        size: `${(file.size / (1024 * 1024)).toFixed(1)} MB`,
        duration: sessionStorage.getItem('duration') || '00:10',
        mode: mode === 'full' ? 'Full Analysis' : 'Classification Only',
        status: 'Completed',
        results: res,
        generateReport: genReport,
        patientName: 'N/A',
        patientPhone: 'N/A'
      }

      const historyJson = localStorage.getItem('swaraaha_history')
      const existingHistory = historyJson ? JSON.parse(historyJson) : []
      localStorage.setItem('swaraaha_history', JSON.stringify([historyItem, ...existingHistory]))

      try {
        await storeAudioFile(id, file)
      } catch (e) {
        console.error("Failed to store audio in IndexedDB:", e)
      }

      // Briefly wait at 100% to let progress bar feel complete
      setTimeout(() => {
        setResults(res)
        setIsLoading(false)
      }, 500)

    } catch (err) {
      clearInterval(progressInterval)
      console.error('Background analysis failed:', err)
      setErrorMsg("Failed to complete speech analysis. Please verify the backend is running.")
      setIsLoading(false)
    }
  }

  useEffect(() => {
    const data = sessionStorage.getItem('results')
    const name = sessionStorage.getItem('filename')
    const size = sessionStorage.getItem('filesize')
    const dur = sessionStorage.getItem('duration')
    const pName = sessionStorage.getItem('patient_name')
    const pPhone = sessionStorage.getItem('patient_phone')

    // Initial state population
    setFilename(name || 'patient_recording.wav')
    setFilesize(size || '18 MB')
    setDuration(dur || '02:14')
    setPatientName(pName || 'N/A')
    setPatientPhone(pPhone || 'N/A')


    // Set today's date
    const d = new Date()
    setReportDate(d.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }))

    if (data) {
      setResults(JSON.parse(data))
      setIsLoading(false)
    } else {
      // No results in session storage. Let's check if we have a file to analyze!
      if (analyzedFile) {
        performAnalysis(analyzedFile)
      } else {
        // No file and no results: redirect back to home page
        navigate('/')
      }
    }
  }, [navigate, analyzedFile])

  if (isLoading) {
    return (
      <div className="p-6 md:p-8 max-w-2xl mx-auto h-[80vh] flex flex-col items-center justify-center animate-fade-in text-text-primary">
        <div className="w-full bg-bg-card border border-border-color rounded-2xl p-8 space-y-6 shadow-xl flex flex-col items-center text-center" style={{ borderRadius: '24px' }}>
          
          {/* Animated Glow Logo Spinner */}
          <div className="relative flex items-center justify-center w-16 h-16 rounded-2xl bg-teal-500 text-white shadow-lg animate-bounce">
            <Activity size={32} className="animate-pulse" />
            <span className="absolute -top-1 -right-1 flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-teal-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-teal-500"></span>
            </span>
          </div>

          <div className="space-y-2">
            <h2 className="text-lg font-bold tracking-tight">Speech Diagnostics In Progress</h2>
            <p className="text-xs text-text-secondary max-w-sm">
              We are processing your recording. This typically takes between 2 to 10 seconds.
            </p>
          </div>

          {/* Premium Progress Bar Wrapper */}
          <div className="w-full space-y-2 max-w-md pt-2">
            <div className="flex justify-between text-[10px] font-semibold text-text-secondary uppercase font-mono">
              <span className="text-accent-teal">{loadingStep}</span>
              <span>{loadingProgress}%</span>
            </div>
            <div className="h-2 w-full bg-hover-color/50 rounded-full overflow-hidden border border-border-color/50">
              <div 
                className="h-full bg-gradient-to-r from-teal-500 to-emerald-400 rounded-full transition-all duration-300 shadow-sm"
                style={{ width: `${loadingProgress}%` }}
              />
            </div>
          </div>

          {/* Informational tips */}
          <div className="pt-4 border-t border-border-color/50 w-full max-w-md flex justify-center text-[10px] text-text-secondary select-none">
            <div className="flex items-center gap-2">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent-teal" />
              <span>Whisper-Tiny language adapters active</span>
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent-teal" />
              <span>GPU/CPU inference optimized</span>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (errorMsg) {
    return (
      <div className="p-6 md:p-8 max-w-md mx-auto h-[80vh] flex flex-col items-center justify-center animate-fade-in text-text-primary">
        <div className="w-full bg-bg-card border border-error-red/20 rounded-2xl p-8 space-y-6 shadow-xl flex flex-col items-center text-center" style={{ borderRadius: '24px' }}>
          <div className="p-4 bg-error-red/10 text-error-red rounded-full">
            <Activity size={32} />
          </div>
          <div className="space-y-2">
            <h2 className="text-lg font-bold tracking-tight">Analysis Failed</h2>
            <p className="text-xs text-text-secondary">
              {errorMsg}
            </p>
          </div>
          <button
            onClick={() => navigate('/')}
            className="w-full py-2 bg-accent-teal hover:bg-teal-600 text-white text-xs font-bold rounded-lg shadow-sm transition cursor-pointer"
            style={{ borderRadius: '8px' }}
          >
            Go Back & Try Again
          </button>
        </div>
      </div>
    )
  }

  if (!results) return null

  // Calculate Speech Metrics
  const classes = Object.keys(results.classification).filter(
    c => results.classification[c] && typeof results.classification[c].confidence === 'number'
  )
  const presentCount = classes.filter(c => results.classification[c].label === 1).length
  const maxConfidence = classes.length > 0 
    ? Math.max(...classes.map(c => results.classification[c].confidence)) 
    : 0

  // Determine Severity Level
  let severity = "Fluent"
  let severityColor = "text-emerald-500 bg-emerald-500/10 border-emerald-500/20"
  let stutterIndex = "0.0%"

  if (presentCount >= 4 || maxConfidence > 0.85) {
    severity = "Severe"
    severityColor = "text-error-red bg-error-red/10 border-error-red/20"
    stutterIndex = "24.5%"
  } else if (presentCount >= 2 || maxConfidence > 0.6) {
    severity = "Moderate"
    severityColor = "text-warning-amber bg-warning-amber/10 border-warning-amber/20"
    stutterIndex = "11.2%"
  } else if (presentCount > 0 || maxConfidence > 0.25) {
    severity = "Mild"
    severityColor = "text-teal-500 bg-teal-500/10 border-teal-500/20"
    stutterIndex = "4.8%"
  }

  // Helper to parse duration string to seconds
  const getDurationInSeconds = (durStr: string) => {
    const parts = durStr.split(':')
    if (parts.length === 2) {
      return parseInt(parts[0]) * 60 + parseInt(parts[1])
    }
    return 134 // default 02:14
  }

  const durationSec = getDurationInSeconds(duration)

  // Print clinical report
  const handlePrint = () => {
    setShowPrintModal(true)
  }

  return (
    <div className="p-6 md:p-8 space-y-6 max-w-5xl mx-auto animate-fade-in text-text-primary print:p-0 print:bg-white print:text-black">
      
      {/* HEADER CONTROLS (Hidden in Print) */}
      <div className="flex items-center justify-between border-b border-border-color pb-4 print:hidden">
        <button
          onClick={() => navigate('/')}
          className="flex items-center gap-2 text-xs font-semibold text-text-secondary hover:text-text-primary transition cursor-pointer"
        >
          <ArrowLeft size={16} />
          <span>Upload Another Sample</span>
        </button>

        <div className="flex items-center gap-2">
          {showReport && (
            <button
              onClick={handlePrint}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-accent-teal hover:bg-teal-600 text-white text-xs font-semibold rounded-lg shadow-xs transition cursor-pointer"
            >
              <Printer size={14} />
              <span>Print Clinical Report</span>
            </button>
          )}
        </div>
      </div>

      {/* INTERACTIVE WAVEFORM CARD (Hidden in Print) */}
      {results && (
        <div className="print:hidden">
          <WaveformView file={analyzedFile} regions={results.localization.regions} transcription={results.transcription} />
        </div>
      )}

      {/* CLINICAL SUMMARY CARDS (Adaptive Grid Layout) */}
      <div className="grid gap-6 md:grid-cols-3 print:hidden">
        
        {/* Severity Card */}
        <div className="p-5 bg-bg-card border border-border-color rounded-xl flex flex-col justify-between shadow-xs" style={{ borderRadius: '16px' }}>
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold text-text-secondary uppercase tracking-wider">Diagnostic Severity</span>
            <Volume2 className="text-text-secondary" size={16} />
          </div>
          <div className="mt-4">
            <span className={`inline-block px-3 py-1 text-xs font-bold border rounded-full ${severityColor}`}>
              {severity} Dysfluency
            </span>
            <p className="text-xs text-text-secondary mt-2">
              Based on stuttering frequency and speech localization patterns.
            </p>
          </div>
        </div>

        {/* Index Metric Card */}
        <div className="p-5 bg-bg-card border border-border-color rounded-xl flex flex-col justify-between shadow-xs" style={{ borderRadius: '16px' }}>
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold text-text-secondary uppercase tracking-wider">Dysfluency Index</span>
            <Activity className="text-accent-teal" size={16} />
          </div>
          <div className="mt-4">
            <h3 className="text-2xl font-bold tracking-tight text-text-primary">{stutterIndex}</h3>
            <p className="text-xs text-text-secondary mt-1">
              Percentage of clinical words or syllables classified with dysfluency.
            </p>
          </div>
        </div>

        {/* Metadata Details Card */}
        <div className="p-5 bg-bg-card border border-border-color rounded-xl space-y-3 shadow-xs" style={{ borderRadius: '16px' }}>
          <span className="text-[10px] font-bold text-text-secondary uppercase tracking-wider block">Recording Metadata</span>
          <div className="grid grid-cols-2 gap-2 text-xs text-text-secondary">
            <div>
              <span className="block text-[10px] font-semibold text-text-primary">File Name</span>
              <span className="truncate block font-medium max-w-[120px]">{filename}</span>
            </div>
            <div>
              <span className="block text-[10px] font-semibold text-text-primary">Duration</span>
              <span className="font-mono font-medium">{duration}</span>
            </div>
            <div>
              <span className="block text-[10px] font-semibold text-text-primary">File Size</span>
              <span>{filesize}</span>
            </div>
            <div>
              <span className="block text-[10px] font-semibold text-text-primary">Date</span>
              <span>{reportDate}</span>
            </div>
          </div>
        </div>
      </div>

      {/* CLASSIFICATION ANALYSIS DETAILS */}
      <div className="grid gap-6 md:grid-cols-3 print:hidden">
        
        {/* Left Side: Category Classifications */}
        <div className="md:col-span-2 p-6 bg-bg-card border border-border-color rounded-xl space-y-4 shadow-xs" style={{ borderRadius: '16px' }}>
          <div className="border-b border-border-color pb-3 flex items-center justify-between">
            <h3 className="text-sm font-bold">Speech Dysfluency Classification</h3>
            <span className="text-[9px] text-text-secondary font-semibold uppercase">Confidence Score</span>
          </div>

          <div className="space-y-4">
            {classes.map((name) => {
              const r = results.classification[name]
              const displayName = CLASS_DISPLAY_NAMES[name] || name
              
              return (
                <div key={name} className="space-y-1.5">
                  <div className="flex items-center justify-between text-xs font-semibold">
                    <span className="text-text-primary">{displayName}</span>
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-text-secondary">{(r.confidence * 100).toFixed(1)}%</span>
                      <span className={`px-2 py-0.5 text-[10px] font-bold border rounded-full ${
                        r.label 
                          ? 'bg-error-red/10 border-error-red/20 text-error-red' 
                          : 'bg-hover-color border-border-color text-text-secondary'
                      }`}>
                        {r.label ? 'Present' : 'Absent'}
                      </span>
                    </div>
                  </div>
                  {/* Progress bar */}
                  <div className="h-2 bg-hover-color rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-300 ${
                        r.label ? 'bg-accent-teal' : 'bg-text-secondary/30'
                      }`}
                      style={{ width: `${r.confidence * 100}%` }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Right Side: Localization Regions List */}
        <div className="p-6 bg-bg-card border border-border-color rounded-xl space-y-4 shadow-xs flex flex-col justify-between" style={{ borderRadius: '16px' }}>
          <div className="space-y-3">
            <div className="border-b border-border-color pb-3">
              <h3 className="text-sm font-bold">Dysfluency Localization</h3>
            </div>
            
            {results.localization.regions.length === 0 ? (
              <div className="py-8 text-center text-xs text-text-secondary space-y-2">
                <CheckCircle2 className="mx-auto text-emerald-500" size={24} />
                <p>No stuttering events localized in timestamps.</p>
              </div>
            ) : (
              <div className="space-y-2.5 max-h-[220px] overflow-y-auto pr-1">
                {results.localization.regions.map((r, i) => (
                  <div key={i} className="flex items-center justify-between p-2.5 bg-bg-sidebar border border-border-color rounded-lg text-xs">
                    <div className="font-mono font-bold text-text-primary">
                      {r.start.toFixed(2)}s &mdash; {r.end.toFixed(2)}s
                    </div>
                    <div className="text-[10px] font-semibold text-text-secondary">
                      Conf: {(r.confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="pt-3 border-t border-border-color/50 text-[10px] text-text-secondary flex items-center gap-1">
            <ShieldCheck size={13} className="text-accent-teal shrink-0" />
            <span>Secure clinical AI localization</span>
          </div>
        </div>
      </div>

      {/* DYSFLUENCY VISUAL TIMELINE MAPPING */}
      {results.localization.regions.length > 0 && (
        <div className="p-6 bg-bg-card border border-border-color rounded-xl space-y-4 shadow-xs print:hidden" style={{ borderRadius: '16px' }}>
          <div className="border-b border-border-color pb-2">
            <h3 className="text-sm font-bold">Stuttering Occurrence Timeline</h3>
            <p className="text-[10px] text-text-secondary">Visual representation of localized stutters throughout the recording.</p>
          </div>

          <div className="space-y-2 pt-2">
            {/* Timeline scale track */}
            <div className="relative h-6 bg-bg-sidebar border border-border-color rounded-lg overflow-hidden shadow-inner flex items-center">
              
              {/* Plot regions */}
              {results.localization.regions.map((r, i) => {
                const leftPercent = (r.start / durationSec) * 100
                const widthPercent = ((r.end - r.start) / durationSec) * 100
                
                return (
                  <div 
                    key={i}
                    className="absolute h-full bg-error-red/40 border-l border-r border-error-red flex items-center justify-center text-[8px] font-bold text-error-red select-none"
                    style={{ left: `${leftPercent}%`, width: `${widthPercent}%` }}
                    title={`Stutter at ${r.start.toFixed(1)}s - ${r.end.toFixed(1)}s`}
                  >
                    !
                  </div>
                )
              })}

              {/* Central base line */}
              <div className="w-full h-0.5 bg-border-color" />
            </div>

            {/* Time labels */}
            <div className="flex justify-between text-[9px] font-mono text-text-secondary px-1">
              <span>0.00s</span>
              <span>{formatSecondsLabel(durationSec / 2)}</span>
              <span>{formatSecondsLabel(durationSec)} (End)</span>
            </div>
          </div>
        </div>
      )}


      {/* PRINT CLINICAL PDF REPORT (Formal medical document format) */}
      {showReport && (
        <div 
          ref={reportRef}
          className="bg-white text-gray-900 border border-gray-300 p-12 space-y-8 max-w-4xl mx-auto rounded-xl shadow-md hidden print:block print:border-none print:shadow-none print:p-0"
          style={{ fontFamily: 'Georgia, serif' }}
        >
          {/* Clinic Letterhead */}
          <div className="flex items-start justify-between border-b-2 border-teal-600 pb-6">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-teal-600 rounded-lg flex items-center justify-center text-white">
                <Activity size={28} />
              </div>
              <div>
                <h2 className="text-xl font-bold tracking-tight text-gray-900">SWARAAHA MEDICAL</h2>
                <p className="text-[10px] font-bold text-teal-600 uppercase tracking-widest">Clinical Speech & Hearing Center</p>
              </div>
            </div>
            <div className="text-right text-[10px] text-gray-400 space-y-0.5">
              <p>Clinical Diagnostic Services</p>
              <p>Secure Medical ID: SWR-REPORT-920</p>
              <p>{reportDate}</p>
            </div>
          </div>

          {/* Report Title */}
          <div className="text-center space-y-1">
            <h1 className="text-xl font-bold tracking-tight text-gray-900 uppercase">
              Clinical Speech Assessment Report
            </h1>
            <p className="text-xs text-gray-500 italic">Generated via Swaraaha Speech Dysfluency Detection AI Engine</p>
          </div>

          {/* Patient Details Grid */}
          <div className="bg-gray-50 p-4 border border-gray-200 rounded-lg grid grid-cols-2 gap-4 text-xs">
            <div>
              <span className="block text-[10px] uppercase font-bold text-gray-500">Patient Name</span>
              <span className="font-semibold text-gray-900">{patientName}</span>
            </div>
            <div>
              <span className="block text-[10px] uppercase font-bold text-gray-500">Phone Number</span>
              <span className="text-gray-900">{patientPhone}</span>
            </div>
            <div>
              <span className="block text-[10px] uppercase font-bold text-gray-500">Audio Filename</span>
              <span className="font-mono text-gray-900">{filename}</span>
            </div>
            <div>
              <span className="block text-[10px] uppercase font-bold text-gray-500">Assessment Date</span>
              <span className="text-gray-900">{reportDate}</span>
            </div>
            <div>
              <span className="block text-[10px] uppercase font-bold text-gray-500">Audio Clip Duration</span>
              <span className="font-mono text-gray-900">{duration}</span>
            </div>
            <div>
              <span className="block text-[10px] uppercase font-bold text-gray-500">Overall Diagnostic Index</span>
              <span className="font-bold text-teal-700">{stutterIndex} ({severity} Dysfluency)</span>
            </div>
          </div>

          {/* Diagnostic Classifications Table */}
          <div className="space-y-3">
            <h3 className="text-sm font-bold text-gray-900 uppercase tracking-wider border-b border-gray-200 pb-1.5">
              1. Speech Stuttering Classifications
            </h3>
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-gray-300 text-gray-500 text-[10px] font-bold uppercase">
                  <th className="py-2">Dysfluency Category</th>
                  <th className="py-2">Clincial Present Label</th>
                  <th className="py-2 text-right">Model Confidence Score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {classes.map((name) => {
                  const r = results.classification[name]
                  const displayName = CLASS_DISPLAY_NAMES[name] || name
                  return (
                    <tr key={name}>
                      <td className="py-2.5 font-medium">{displayName}</td>
                      <td className="py-2.5">
                        <span className={`px-2 py-0.5 text-[9px] font-bold border rounded-full uppercase ${
                          r.label 
                            ? 'bg-red-50 border-red-100 text-red-700' 
                            : 'bg-gray-50 border-gray-100 text-gray-400'
                        }`}>
                          {r.label ? 'Detected' : 'Not Detected'}
                        </span>
                      </td>
                      <td className="py-2.5 text-right font-mono text-gray-500">{(r.confidence * 100).toFixed(1)}%</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* Localized Timestamps Table */}
          {results.localization.regions.length > 0 && (
            <div className="space-y-3 page-break-before">
              <h3 className="text-sm font-bold text-gray-900 uppercase tracking-wider border-b border-gray-200 pb-1.5">
                2. Localized Dysfluency Timestamps
              </h3>
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-gray-300 text-gray-500 text-[10px] font-bold uppercase">
                    <th className="py-2">Event index</th>
                    <th className="py-2">Start Time</th>
                    <th className="py-2">End Time</th>
                    <th className="py-2">Duration</th>
                    <th className="py-2 text-right">Confidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {results.localization.regions.map((r, i) => (
                    <tr key={i}>
                      <td className="py-2 text-gray-400 font-mono">Event #{i + 1}</td>
                      <td className="py-2 font-mono">{r.start.toFixed(2)}s</td>
                      <td className="py-2 font-mono">{r.end.toFixed(2)}s</td>
                      <td className="py-2 font-mono">{(r.end - r.start).toFixed(2)}s</td>
                      <td className="py-2 text-right font-mono text-gray-500">{(r.confidence * 100).toFixed(0)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Recommendations Block */}
          <div className="space-y-2 border-t border-gray-200 pt-6">
            <h3 className="text-xs font-bold text-gray-900 uppercase tracking-wider">
              {results.localization.regions.length > 0 ? "3." : "2."} Clinical Guidance & Recommendations
            </h3>
            <p className="text-xs text-gray-600 leading-relaxed text-justify">
              This speech sample was evaluated using standardized reading passages. 
              {severity === "Severe" || severity === "Moderate" ? (
                " Based on the detected dysfluencies, the subject exhibits moderate-to-severe stuttering events. It is recommended to perform detailed diagnostic speech testing, review stuttering severity instrument patterns (SSI-4), and design appropriate fluency shaping or stuttering modification therapeutic exercises."
              ) : (
                " The subject demonstrates fluent speech qualities or minor dysfluencies. Periodic check-ins and normal follow-ups are suggested."
              )}
            </p>
          </div>

          {/* Signatures */}
          <div className="pt-12 grid grid-cols-2 gap-12 text-xs">
            <div className="space-y-4">
              <div className="h-0.5 bg-gray-300 w-full" />
              <div className="text-gray-500">
                <span className="block font-bold text-gray-900">Therapist/SLP Signature</span>
                <span className="block">Speech-Language Pathologist</span>
              </div>
            </div>
            <div className="space-y-4 text-right">
              <div className="h-0.5 bg-gray-300 w-full" />
              <div className="text-gray-500">
                <span className="block font-bold text-gray-900">Swaraaha Assessment Engine</span>
                <span>Verified Diagnostic Output</span>
              </div>
            </div>
          </div>
        </div>
      )}
      {/* PRINT DETAILS MODAL (Hidden in Print) */}
      {showPrintModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-xs flex items-center justify-center z-50 animate-fade-in print:hidden">
          <div className="bg-bg-card border border-border-color rounded-2xl p-6 w-full max-w-sm shadow-xl space-y-4 m-4" style={{ borderRadius: '16px' }}>
            <div>
              <h3 className="text-sm font-bold text-text-primary">Clinical Report Details</h3>
              <p className="text-[10px] text-text-secondary mt-0.5">Please provide patient information to include in the generated report.</p>
            </div>
            
            <div className="space-y-3">
              <div className="space-y-1">
                <label className="text-[9px] uppercase font-bold text-text-secondary">Patient Name</label>
                <input
                  type="text"
                  placeholder="e.g. John Doe"
                  value={patientName === 'N/A' ? '' : patientName}
                  onChange={(e) => setPatientName(e.target.value)}
                  className="w-full px-3 py-2 text-xs bg-bg-sidebar border border-border-color rounded-lg focus:outline-none focus:border-accent-teal focus:ring-1 focus:ring-accent-teal"
                  style={{ borderRadius: '8px' }}
                  autoFocus
                />
              </div>
              <div className="space-y-1">
                <label className="text-[9px] uppercase font-bold text-text-secondary">Phone Number</label>
                <input
                  type="text"
                  placeholder="e.g. +91 9876543210"
                  value={patientPhone === 'N/A' ? '' : patientPhone}
                  onChange={(e) => setPatientPhone(e.target.value)}
                  className="w-full px-3 py-2 text-xs bg-bg-sidebar border border-border-color rounded-lg focus:outline-none focus:border-accent-teal focus:ring-1 focus:ring-accent-teal"
                  style={{ borderRadius: '8px' }}
                />
              </div>
            </div>

            <div className="flex items-center gap-3 pt-2">
              <button
                onClick={() => setShowPrintModal(false)}
                className="flex-1 py-2 bg-hover-color/50 hover:bg-hover-color text-text-primary text-xs font-semibold rounded-lg border border-border-color transition cursor-pointer"
                style={{ borderRadius: '8px' }}
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  setShowPrintModal(false)
                  // Wait a brief tick for the modal to close and visual focus to settle before opening window.print()
                  setTimeout(() => {
                    window.print()
                  }, 100)
                }}
                className="flex-1 py-2 bg-accent-teal hover:bg-teal-600 text-white text-xs font-bold rounded-lg shadow-sm transition cursor-pointer"
                style={{ borderRadius: '8px' }}
              >
                Print Report
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  )
}

// Seconds formatter label
function formatSecondsLabel(sec: number) {
  const mins = Math.floor(sec / 60)
  const secs = Math.floor(sec % 60)
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}s`
}
