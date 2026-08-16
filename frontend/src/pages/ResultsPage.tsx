import { useNavigate } from 'react-router-dom'
import { useEffect, useState, useRef } from 'react'
import { 
  ArrowLeft, 
  Volume2, 
  ShieldCheck, 
  CheckCircle2, 
  Activity,
  Play,
  Pause,
  Download
} from 'lucide-react'
import { SeverityResult, TranscriptionData, RegionType, LocalizationRegion, CombinedResults, analyzeAudio, downloadReport } from '../api/client'
import { storeAudioFile } from '../utils/db'

interface AnalysisResults {
  classification: Record<string, { label: number; confidence: number }>
  localization: { regions: LocalizationRegion[] }
  transcription?: TranscriptionData
  severity?: SeverityResult
  combined?: CombinedResults
}

const CLASS_DISPLAY_NAMES: Record<string, string> = {
  prolongation: "Prolongation",
  block: "Block (Silent)",
  soundrep: "Sound Repetition",
  wordrep: "Word Repetition",
  interjection: "Interjection (Filler)"
}

// Per-type styling: fills/strokes for the waveform canvas, plus a solid color
// used for badges, the timeline and the legend.
const REGION_STYLES: Record<string, { label: string; fill: string; stroke: string; color: string }> = {
  prolongation: { label: "Prolongation", fill: "rgba(196, 181, 253, 0.45)", stroke: "#8B5CF6", color: "#8B5CF6" },
  block:        { label: "Block",        fill: "rgba(148, 163, 184, 0.45)", stroke: "#64748B", color: "#64748B" },
  soundrep:     { label: "Sound Repetition", fill: "rgba(147, 197, 253, 0.45)", stroke: "#3B82F6", color: "#3B82F6" },
  wordrep:      { label: "Word Repetition",  fill: "rgba(253, 186, 116, 0.45)", stroke: "#F97316", color: "#F97316" },
  interjection: { label: "Interjection", fill: "rgba(254, 240, 138, 0.55)", stroke: "#EAB308", color: "#EAB308" },
}

const DEFAULT_REGION_STYLE = { label: "Stutter", fill: "rgba(239, 68, 68, 0.15)", stroke: "#EF4444", color: "#EF4444" }

// Regions may come from the localizer (has `type`) or the fusion combiner
// (has `primary_type`); both shapes are displayed uniformly.
type DisplayRegion = LocalizationRegion & { primary_type?: string | null }

function regionStyle(type?: RegionType | string | null) {
  return type && REGION_STYLES[type] ? REGION_STYLES[type] : DEFAULT_REGION_STYLE
}

interface WaveformViewProps {
  file: File | null
  regions: DisplayRegion[]
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

      // 2. Overlays (coarse whole-clip estimates hatched, precise drawn on top)
      const sortedRegions = [...regions].sort((a, b) => a.confidence - b.confidence)
      sortedRegions.forEach(region => {
        const x1 = (region.start / audioBuffer.duration) * width
        const x2 = (region.end / audioBuffer.duration) * width
        const style = regionStyle(region.type ?? region.primary_type)

        if (region.coarse) {
          ctx.save()
          ctx.beginPath()
          ctx.rect(x1, 0, x2 - x1, height)
          ctx.clip()
          ctx.fillStyle = style.fill
          ctx.fillRect(x1, 0, x2 - x1, height)
          ctx.strokeStyle = style.stroke
          ctx.globalAlpha = 0.55
          ctx.lineWidth = 1
          ctx.beginPath()
          for (let x = x1 - height; x < x2 + height; x += 8) {
            ctx.moveTo(x, height)
            ctx.lineTo(x + height, 0)
          }
          ctx.stroke()
          ctx.restore()
          return
        }

        ctx.fillStyle = style.fill
        ctx.fillRect(x1, 0, x2 - x1, height)

        ctx.strokeStyle = style.stroke
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
        <div className="flex items-center gap-4 text-[9px] text-text-secondary uppercase font-semibold flex-wrap">
          <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full bg-accent-teal" /> Waveform</span>
          {(() => {
            const types = Array.from(new Set(regions.map(r => r.type ?? r.primary_type ?? 'stutter')))
            if (types.length === 0) {
              return <span className="flex items-center gap-1"><span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: DEFAULT_REGION_STYLE.color }} /> Stutter</span>
            }
            return types.map(t => {
              const s = regionStyle(t === 'stutter' ? undefined : t)
              return (
                <span key={t} className="flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: s.color }} /> {s.label}
                </span>
              )
            })
          })()}
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
  const [isDownloading, setIsDownloading] = useState(false)
  const [downloadError, setDownloadError] = useState<string | null>(null)
  const [reportDate, setReportDate] = useState('')
  const showReport = true

  // Loading and background processing states
  const [isLoading, setIsLoading] = useState(false)
  const [loadingStep, setLoadingStep] = useState("Initializing analysis...")
  const [loadingProgress, setLoadingProgress] = useState(5)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

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
      const genReport = sessionStorage.getItem('generate_report') === 'true'
      const lang = sessionStorage.getItem('transcription_language') || 'english'

      // Step transitions based on timer
      setTimeout(() => setLoadingStep("Classifying stuttering categories..."), 1500)
      setTimeout(() => setLoadingStep("Localizing speech stutters..."), 3000)
      setTimeout(() => setLoadingStep("Transcribing speech..."), 5000)
      setTimeout(() => setLoadingStep("Generating clinical diagnostic report..."), 7000)

      const res = await analyzeAudio(file, lang)

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
        mode: 'Full Analysis',
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
    setFilename(name || '')
    setFilesize(size || '')
    setDuration(dur || '00:00')
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

  const combined = results.combined && !('error' in results.combined)
    ? results.combined
    : undefined
  const regions: DisplayRegion[] =
    combined?.regions ?? results.localization.regions

  // Calculate Speech Metrics
  const classes = Object.keys(results.classification).filter(
    c => results.classification[c] && typeof results.classification[c].confidence === 'number'
  )
  const localizationCoverage = results.localization.regions.reduce(
    (sum, r) => sum + Math.max(0, r.end - r.start),
    0
  )

  // Helper to parse duration string (MM:SS or HH:MM:SS) to seconds
  const getDurationInSeconds = (durStr: string) => {
    const parts = durStr.split(':').map(Number)
    if (parts.every(n => Number.isFinite(n))) {
      return parts.reduce((total, p) => total * 60 + p, 0)
    }
    return 0
  }

  const durationSec = getDurationInSeconds(duration)

  // Stutter index: backend-computed when available, else derived from regions
  const stutterIndexValue = results.severity?.index_pct ??
    (durationSec > 0 ? (localizationCoverage / durationSec) * 100 : 0)

  const SEVERITY_COLORS: Record<string, string> = {
    Fluent: "text-emerald-500 bg-emerald-500/10 border-emerald-500/20",
    Mild: "text-teal-500 bg-teal-500/10 border-teal-500/20",
    Moderate: "text-warning-amber bg-warning-amber/10 border-warning-amber/20",
    Severe: "text-error-red bg-error-red/10 border-error-red/20",
  }

  const computedSeverity = (() => {
    if (results.severity?.label) {
      return {
        label: results.severity.label,
        color: SEVERITY_COLORS[results.severity.label] ?? SEVERITY_COLORS.Fluent,
      }
    }
    let label = 'Fluent'
    if (stutterIndexValue >= 15) label = 'Severe'
    else if (stutterIndexValue >= 5) label = 'Moderate'
    else if (stutterIndexValue >= 2) label = 'Mild'
    return { label, color: SEVERITY_COLORS[label] }
  })()

  // Print clinical report
  const handlePrint = () => {
    setShowPrintModal(true)
  }

  const handleDownload = async () => {
    if (!results) return
    setIsDownloading(true)
    setDownloadError(null)
    try {
      const blob = await downloadReport({
        patient: { name: patientName, phone: patientPhone },
        audio: { filename, size: filesize, duration },
        date: reportDate,
        classification: results.classification,
        severity: results.severity ?? {
          index_pct: stutterIndexValue,
          severity: computedSeverity.label.toLowerCase() as SeverityResult['severity'],
          label: computedSeverity.label,
        },
        combined: combined ?? undefined,
        transcription: results.transcription,
        localization: results.localization,
      })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `swaraaha-report-${reportDate.replace(/[^a-zA-Z0-9]+/g, '-')}.pdf`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      setShowPrintModal(false)
    } catch {
      setDownloadError('Failed to generate the PDF report. Please try again.')
    } finally {
      setIsDownloading(false)
    }
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
              <Download size={14} />
              <span>Download PDF Report</span>
            </button>
          )}
        </div>
      </div>

      {/* INTERACTIVE WAVEFORM CARD (Hidden in Print) */}
      {results && (
        <div className="print:hidden">
          <WaveformView file={analyzedFile} regions={regions} transcription={results.transcription} />
        </div>
      )}

      {/* CLINICAL SUMMARY CARDS (Adaptive Grid Layout) */}
      <div className="grid gap-6 md:grid-cols-2 print:hidden">
        
        {/* Severity Card */}
        <div className="p-5 bg-bg-card border border-border-color rounded-xl flex flex-col justify-between shadow-xs" style={{ borderRadius: '16px' }}>
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold text-text-secondary uppercase tracking-wider">Diagnostic Severity</span>
            <Volume2 className="text-text-secondary" size={16} />
          </div>
          <div className="mt-4">
            <span className={`inline-block px-3 py-1 text-xs font-bold border rounded-full ${computedSeverity.color}`}>
              {computedSeverity.label} Dysfluency
            </span>
            <p className="text-xs text-text-secondary mt-2">
              Based on speech localization regions.
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
            <div className="border-b border-border-color pb-3 flex items-center justify-between">
              <h3 className="text-sm font-bold">Dysfluency Localization</h3>
              <span className="text-[10px] font-semibold text-text-secondary">
                {(combined ? combined.total_stutters : regions.length)} Events
              </span>
            </div>
            
            {regions.length === 0 ? (
              <div className="py-8 text-center text-xs text-text-secondary space-y-2">
                <CheckCircle2 className="mx-auto text-emerald-500" size={24} />
                <p>No stuttering events localized in timestamps.</p>
              </div>
            ) : (
              <div className="space-y-2.5 max-h-[220px] overflow-y-auto pr-1">
                {regions.map((r, i) => (
                  <div key={i} className="flex items-center justify-between p-2.5 bg-bg-sidebar border border-border-color rounded-lg text-xs">
                    <div className="flex flex-col gap-1 min-w-0">
                      <div className="font-mono font-bold text-text-primary">
                        {r.start.toFixed(2)}s &mdash; {r.end.toFixed(2)}s
                      </div>
                      <div className="flex items-center gap-1">
                        <span
                          className="inline-block w-1.5 h-1.5 rounded-full shrink-0"
                          style={{ backgroundColor: regionStyle(r.type ?? r.primary_type).color }}
                        />
                        <span className="text-[9px] font-bold uppercase tracking-wide text-text-secondary truncate">
                          {regionStyle(r.type ?? r.primary_type).label}
                          {r.coarse ? ' (est.)' : ''}
                        </span>
                      </div>
                      {r.primary_type && (
                        <span className="inline-block px-1.5 py-0.5 rounded text-[9px] font-bold text-text-primary">
                          {CLASS_DISPLAY_NAMES[r.primary_type] ?? r.primary_type}
                        </span>
                      )}
                    </div>
                    <div className="text-[10px] font-semibold text-text-secondary shrink-0 ml-2">
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
      {regions.length > 0 && (
        <div className="p-6 bg-bg-card border border-border-color rounded-xl space-y-4 shadow-xs print:hidden" style={{ borderRadius: '16px' }}>
          <div className="border-b border-border-color pb-2">
            <h3 className="text-sm font-bold">Stuttering Occurrence Timeline</h3>
            <p className="text-[10px] text-text-secondary">Visual representation of localized stutters throughout the recording.</p>
          </div>

          <div className="space-y-2 pt-2">
            {/* Timeline scale track */}
            <div className="relative h-6 bg-bg-sidebar border border-border-color rounded-lg overflow-hidden shadow-inner flex items-center">
              
              {/* Plot regions (coarse whole-clip regions first, precise on top) */}
              {[...regions]
                .sort((a, b) => a.confidence - b.confidence)
                .map((r, i) => {
                  const leftPercent = (r.start / durationSec) * 100
                  const widthPercent = ((r.end - r.start) / durationSec) * 100
                  const s = regionStyle(r.type ?? r.primary_type)
                  
                  return (
                    <div 
                      key={i}
                      className="absolute h-full flex items-center justify-center text-[8px] font-bold select-none"
                      style={{
                        left: `${leftPercent}%`,
                        width: `${widthPercent}%`,
                        backgroundColor: r.coarse ? undefined : `${s.color}66`,
                        backgroundImage: r.coarse
                          ? `repeating-linear-gradient(45deg, ${s.color}55, ${s.color}55 4px, transparent 4px, transparent 8px)`
                          : undefined,
                        borderLeft: r.coarse ? `1px dashed ${s.color}` : `1px solid ${s.color}`,
                        borderRight: r.coarse ? `1px dashed ${s.color}` : `1px solid ${s.color}`,
                        color: s.color,
                      }}
                      title={`${s.label}${r.coarse ? ' (estimated)' : ''} at ${r.start.toFixed(1)}s - ${r.end.toFixed(1)}s`}
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

          {downloadError && (
            <p className="text-[10px] text-red-500 font-semibold">{downloadError}</p>
          )}

          <div className="flex items-center gap-3 pt-2">
            <button
              onClick={() => setShowPrintModal(false)}
              className="flex-1 py-2 bg-hover-color/50 hover:bg-hover-color text-text-primary text-xs font-semibold rounded-lg border border-border-color transition cursor-pointer"
              style={{ borderRadius: '8px' }}
            >
              Cancel
            </button>
            <button
              onClick={handleDownload}
              disabled={isDownloading}
              className="flex-1 py-2 bg-accent-teal hover:bg-teal-600 disabled:opacity-50 disabled:cursor-not-allowed text-white text-xs font-bold rounded-lg shadow-sm transition cursor-pointer"
              style={{ borderRadius: '8px' }}
            >
              {isDownloading ? 'Generating...' : 'Download PDF'}
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
