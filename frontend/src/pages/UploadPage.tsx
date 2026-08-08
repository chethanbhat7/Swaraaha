import { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  UploadCloud,
  Mic,
  Trash2,
  Info,
  ShieldCheck,
  Play,
  Pause,
  Square,
  Music,
  CheckCircle2,
  AlertCircle,
  Volume2
} from 'lucide-react'
import PdfViewer from '../components/PdfViewer'

function AudioPlayer({ file }: { file: File }) {
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [isMuted, setIsMuted] = useState(false)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  useEffect(() => {
    if (file) {
      const url = URL.createObjectURL(file)
      setAudioUrl(url)
      setIsPlaying(false)
      setCurrentTime(0)

      // Decode audio using AudioContext to guarantee a valid, finite duration!
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)()
      const reader = new FileReader()
      reader.onload = (e) => {
        const arrayBuffer = e.target?.result as ArrayBuffer
        audioCtx.decodeAudioData(arrayBuffer, (buffer) => {
          setDuration(buffer.duration)
          audioCtx.close()
        }, () => {
          audioCtx.close()
        })
      }
      reader.readAsArrayBuffer(file)

      return () => {
        URL.revokeObjectURL(url)
      }
    }
  }, [file])

  const togglePlay = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (!audioRef.current) return
    if (isPlaying) {
      audioRef.current.pause()
      setIsPlaying(false)
    } else {
      audioRef.current.play()
      setIsPlaying(true)
    }
  }

  const toggleMute = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (!audioRef.current) return
    audioRef.current.muted = !isMuted
    setIsMuted(!isMuted)
  }

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime)
    }
  }

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration)
    }
  }

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (audioRef.current) {
      const time = parseFloat(e.target.value)
      audioRef.current.currentTime = time
      setCurrentTime(time)
    }
  }

  const formatTime = (timeInSeconds: number) => {
    const mins = Math.floor(timeInSeconds / 60).toString().padStart(2, '0')
    const secs = Math.floor(timeInSeconds % 60).toString().padStart(2, '0')
    return `${mins}:${secs}`
  }

  return (
    <div className="w-full bg-hover-color/30 border border-border-color rounded-xl p-3 flex flex-col gap-2 mt-3 animate-fade-in" style={{ borderRadius: '12px' }}>
      <audio
        ref={audioRef}
        src={audioUrl || ''}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={() => setIsPlaying(false)}
      />
      <div className="flex items-center gap-3">
        <button
          onClick={togglePlay}
          className="w-8 h-8 rounded-full bg-accent-teal hover:bg-teal-600 text-white flex items-center justify-center transition shrink-0 cursor-pointer shadow-xs"
        >
          {isPlaying ? <Pause size={14} /> : <Play size={14} className="ml-0.5" />}
        </button>
        <div className="flex-1 min-w-0">
          <div className="text-[10px] font-bold text-text-primary truncate">Listen Preview</div>
          <div className="text-[9px] font-medium text-text-secondary">Click play to review recording</div>
        </div>
        <button
          onClick={toggleMute}
          className={`p-1.5 rounded-lg text-text-secondary hover:bg-hover-color transition cursor-pointer shrink-0 ${isMuted ? 'text-error-red bg-error-red/10' : ''}`}
          title={isMuted ? "Unmute" : "Mute"}
        >
          <Volume2 size={14} />
        </button>
        <div className="text-[10px] font-mono text-text-secondary shrink-0 select-none">
          {formatTime(currentTime)} / {formatTime(duration || 0)}
        </div>
      </div>
      <div className="flex items-center gap-2">
        <input
          type="range"
          min={0}
          max={duration || 0}
          value={currentTime}
          onChange={handleSeek}
          className="flex-1 h-1 bg-hover-color rounded-full appearance-none cursor-pointer accent-accent-teal"
        />
      </div>
    </div>
  )
}


export default function UploadPage({
  customPdfs,
  setCustomPdfs,
  setAnalyzedFile
}: {
  customPdfs: Array<{ name: string; url: string }>;
  setCustomPdfs: React.Dispatch<React.SetStateAction<Array<{ name: string; url: string }>>>;
  setAnalyzedFile: React.Dispatch<React.SetStateAction<File | null>>;
}) {
  const navigate = useNavigate()

  // Input selection: 'upload' | 'record'
  const [inputMode, setInputMode] = useState<'upload' | 'record'>('upload')

  // File state
  const [file, setFile] = useState<File | null>(null)
  const [fileDuration, setFileDuration] = useState<string>('00:00')
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Recording state
  const [isRecording, setIsRecording] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)


  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const recordingTimerRef = useRef<number | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const animationFrameRef = useRef<number | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  // Analysis options
  const [analysisMode, setAnalysisMode] = useState<'full' | 'classify'>('full')
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  const [selectedLanguage, setSelectedLanguage] = useState<'english' | 'kannada' | 'hindi'>('english')

  // Clean up recording on unmount
  useEffect(() => {
    return () => {
      stopAudioStreams()
    };
  }, [])

  const stopAudioStreams = () => {
    if (recordingTimerRef.current) {
      window.clearInterval(recordingTimerRef.current)
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close()
    }
  }

  // --- UPLOAD HANDLERS ---
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processSelectedFile(e.dataTransfer.files[0])
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processSelectedFile(e.target.files[0])
    }
  }

  const processSelectedFile = (selectedFile: File) => {
    // Validate format
    const validExtensions = ['.wav', '.mp3', '.flac', '.m4a']
    const hasValidExt = validExtensions.some(ext => selectedFile.name.toLowerCase().endsWith(ext))
    if (!hasValidExt) {
      setErrorMsg("Unsupported file format. Please upload WAV, MP3, FLAC, or M4A.")
      return
    }
    setErrorMsg(null)
    setFile(selectedFile)
    setAnalyzedFile(selectedFile)

    // Determine audio duration dynamically using AudioContext
    const reader = new FileReader()
    reader.onload = function (e) {
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)()
      audioCtx.decodeAudioData(e.target?.result as ArrayBuffer, (buffer) => {
        const durationSec = Math.floor(buffer.duration)
        const mins = Math.floor(durationSec / 60).toString().padStart(2, '0')
        const secs = (durationSec % 60).toString().padStart(2, '0')
        setFileDuration(`${mins}:${secs}`)
        audioCtx.close()
      }, () => {
        // Fallback if decoding fails
        setFileDuration('00:00')
        audioCtx.close()
      })
    }
    reader.readAsArrayBuffer(selectedFile)
  }

  const handleDeleteFile = () => {
    setFile(null)
    setAnalyzedFile(null)
    setFileDuration('00:00')
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  // --- RECORDING HANDLERS ---
  const startRecording = async () => {
    setErrorMsg(null)
    audioChunksRef.current = []

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream

      // Set up Audio Context and Analyser for live wave and sound level
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext
      const audioCtx = new AudioContextClass()
      audioContextRef.current = audioCtx

      const source = audioCtx.createMediaStreamSource(stream)
      const analyser = audioCtx.createAnalyser()
      analyser.fftSize = 256
      analyserRef.current = analyser
      source.connect(analyser)

      // Start Media Recorder
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder

      mediaRecorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          audioChunksRef.current.push(e.data)
        }
      }

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' })
        const recordedFile = new File([audioBlob], 'recorded_speech_sample.wav', { type: 'audio/wav' })
        setFile(recordedFile)
        setAnalyzedFile(recordedFile)

        const mins = Math.floor(recordingTime / 60).toString().padStart(2, '0')
        const secs = (recordingTime % 60).toString().padStart(2, '0')
        setFileDuration(`${mins}:${secs}`)
      }

      mediaRecorder.start(250) // slice data chunks every 250ms
      setIsRecording(true)
      setIsPaused(false)
      setRecordingTime(0)

      // Recording timer update
      recordingTimerRef.current = window.setInterval(() => {
        setRecordingTime(t => t + 1)
      }, 1000)

      // Start live visualizer draw loop
      drawWaveform()

    } catch (err) {
      console.error('Error accessing microphone:', err)
      setErrorMsg("Unable to access microphone. Please check permissions.")
    }
  }

  const pauseRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      if (!isPaused) {
        mediaRecorderRef.current.pause()
        setIsPaused(true)
        if (recordingTimerRef.current) {
          window.clearInterval(recordingTimerRef.current)
        }
      } else {
        mediaRecorderRef.current.resume()
        setIsPaused(false)
        recordingTimerRef.current = window.setInterval(() => {
          setRecordingTime(t => t + 1)
        }, 1000)
        drawWaveform()
      }
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      stopAudioStreams()
      setIsRecording(false)
      setIsPaused(false)
    }
  }

  // Draw Live Waveform loop
  const drawWaveform = () => {
    if (!analyserRef.current || !canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const analyser = analyserRef.current
    const bufferLength = analyser.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)

    const width = canvas.width
    const height = canvas.height

    const draw = () => {
      if (!isRecording || isPaused) {
        // Draw standard flat line when idle or paused
        ctx.clearRect(0, 0, width, height)
        ctx.lineWidth = 2
        ctx.strokeStyle = '#64748B' // slate-500
        ctx.beginPath()
        ctx.moveTo(0, height / 2)
        ctx.lineTo(width, height / 2)
        ctx.stroke()
        return
      }

      animationFrameRef.current = requestAnimationFrame(draw)
      analyser.getByteTimeDomainData(dataArray)

      ctx.clearRect(0, 0, width, height)
      ctx.lineWidth = 2.5
      ctx.strokeStyle = '#14B8A6' // teal-500

      ctx.beginPath()

      const sliceWidth = width / bufferLength
      let x = 0

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0
        const y = (v * height) / 2

        if (i === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }

        x += sliceWidth
      }

      ctx.lineTo(width, height / 2)
      ctx.stroke()
    }

    draw()
  }

  // --- START ANALYSIS TRIGGER ---
  const handleStartAnalysis = async () => {
    if (!file) {
      setErrorMsg("Please select or record a speech sample first.")
      return
    }

    setErrorMsg(null)

    // Clear previous results & save analysis configs to sessionStorage
    sessionStorage.removeItem('results')
    sessionStorage.setItem('analysis_mode', analysisMode)
    sessionStorage.setItem('generate_report', 'true')

    sessionStorage.setItem('transcription_language', selectedLanguage)
    sessionStorage.setItem('filename', file.name)
    sessionStorage.setItem('filesize', `${(file.size / (1024 * 1024)).toFixed(1)} MB`)
    sessionStorage.setItem('duration', fileDuration)
    sessionStorage.setItem('patient_name', 'N/A')
    sessionStorage.setItem('patient_phone', 'N/A')

    // Navigate immediately to the results page
    navigate('/results')
  }

  // Format bytes helper
  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i]
  }

  // Format duration digits
  const formatTime = (timeInSeconds: number) => {
    const mins = Math.floor(timeInSeconds / 60).toString().padStart(2, '0')
    const secs = (timeInSeconds % 60).toString().padStart(2, '0')
    return `${mins}:${secs}`
  }

  return (
    <div className="flex h-full overflow-hidden">

      {/* CENTER PANE: AUDIO ANALYSIS */}
      <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">

        {/* Large Header */}
        <div className="space-y-1">
          <h2 className="text-2xl font-bold tracking-tight">Analyze Speech Sample</h2>
          <p className="text-sm text-text-secondary">Upload or record a patient's speech for dysfluency analysis.</p>
        </div>

        {errorMsg && (
          <div className="flex items-center gap-2 p-3 bg-error-red/10 border border-error-red/30 rounded-lg text-xs text-error-red">
            <AlertCircle size={16} />
            <span>{errorMsg}</span>
          </div>
        )}

        {/* STEP 1: CHOOSE AUDIO INPUT */}
        <section className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="flex items-center justify-center w-5 h-5 rounded-full bg-teal-500/10 text-accent-teal text-xs font-bold">1</span>
            <h3 className="text-sm font-bold">Choose Audio Input</h3>
          </div>

          {/* Segmented Controller Tab */}
          <div className="inline-flex p-1 bg-hover-color/60 rounded-xl border border-border-color">
            <button
              onClick={() => { setInputMode('upload'); handleDeleteFile(); }}
              className={`flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg transition duration-200 cursor-pointer ${inputMode === 'upload'
                ? 'bg-bg-card text-text-primary shadow-xs'
                : 'text-text-secondary hover:text-text-primary'
                }`}
              style={{ borderRadius: '8px' }}
            >
              <UploadCloud size={14} />
              <span>Upload Audio</span>
            </button>
            <button
              onClick={() => { setInputMode('record'); handleDeleteFile(); }}
              className={`flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-lg transition duration-200 cursor-pointer ${inputMode === 'record'
                ? 'bg-bg-card text-text-primary shadow-xs'
                : 'text-text-secondary hover:text-text-primary'
                }`}
              style={{ borderRadius: '8px' }}
            >
              <Mic size={14} />
              <span>Record Audio</span>
            </button>
          </div>

          {/* Mode Panels */}
          {inputMode === 'upload' ? (
            /* UPLOAD CONTAINER */
            <div className="space-y-4">
              {!file ? (
                <div
                  onDragEnter={handleDrag}
                  onDragOver={handleDrag}
                  onDragLeave={handleDrag}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-200 flex flex-col items-center justify-center gap-3 ${dragActive
                    ? 'border-accent-teal bg-teal-500/5'
                    : 'border-border-color hover:border-accent-teal bg-bg-card'
                    }`}
                  style={{ borderRadius: '16px' }}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".wav,.mp3,.flac,.m4a"
                    className="hidden"
                    onChange={handleFileChange}
                  />
                  <div className="p-3 bg-teal-500/10 text-accent-teal rounded-full">
                    <UploadCloud size={24} />
                  </div>
                  <div className="space-y-1">
                    <p className="text-xs font-bold text-text-primary">Drag and drop speech recording here</p>
                    <p className="text-[10px] text-text-secondary">or click to browse local files</p>
                  </div>
                  <button
                    type="button"
                    className="px-3 py-1.5 text-xs font-semibold bg-accent-teal hover:bg-teal-600 text-white rounded-lg transition shadow-xs"
                  >
                    Choose Audio File
                  </button>
                  <div className="flex gap-4 text-[9px] text-text-secondary pt-2 border-t border-border-color/50 w-full max-w-[280px] justify-center">
                    <span>Supports: WAV, MP3, FLAC, M4A</span>
                    <span>Max Size: 200 MB</span>
                  </div>
                </div>
              ) : (
                /* FILE COMPLETED CARD */
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-4 bg-bg-card border border-border-color rounded-xl shadow-xs" style={{ borderRadius: '16px' }}>
                    <div className="flex items-center gap-3">
                      <div className="p-2.5 bg-teal-500/10 text-accent-teal rounded-xl">
                        <Music size={18} />
                      </div>
                      <div>
                        <h4 className="text-xs font-bold truncate max-w-[200px] sm:max-w-[320px]">{file.name}</h4>
                        <p className="text-[10px] text-text-secondary mt-0.5">
                          Duration: {fileDuration} &bull; Size: {formatBytes(file.size)}
                        </p>
                      </div>
                    </div>
                    <button
                      onClick={handleDeleteFile}
                      className="p-2 text-text-secondary hover:text-error-red hover:bg-error-red/10 rounded-lg transition cursor-pointer"
                      title="Delete File"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                  <AudioPlayer file={file} />
                </div>
              )}
            </div>
          ) : (
            /* RECORD AUDIO SECTION */
            <div className="bg-bg-card border border-border-color rounded-xl p-5 space-y-4 shadow-xs" style={{ borderRadius: '16px' }}>
              <div className="flex flex-col items-center justify-center p-6 border border-border-color bg-bg-sidebar rounded-xl gap-4">

                {/* Simulated/Live Oscilloscope Canvas */}
                <canvas
                  ref={canvasRef}
                  width={360}
                  height={80}
                  className="w-full max-w-[360px] bg-bg-app border border-border-color rounded-lg waveform-canvas"
                />


                {/* Timer Clock */}
                <div className="text-2xl font-bold font-mono tracking-wider select-none">
                  {formatTime(recordingTime)}
                </div>

                {/* Micro Actions Controls */}
                <div className="flex items-center gap-3">
                  {!isRecording ? (
                    <button
                      onClick={startRecording}
                      className="flex items-center gap-2 px-5 py-2.5 bg-accent-teal hover:bg-teal-600 text-white text-xs font-bold rounded-full transition shadow-md cursor-pointer animate-pulse"
                    >
                      <Mic size={14} />
                      <span>Start Recording</span>
                    </button>
                  ) : (
                    <>
                      <button
                        onClick={pauseRecording}
                        className="p-3 bg-hover-color hover:bg-border-color text-text-primary rounded-full transition cursor-pointer"
                        title={isPaused ? "Resume" : "Pause"}
                      >
                        {isPaused ? <Play size={16} /> : <Pause size={16} />}
                      </button>
                      <button
                        onClick={stopRecording}
                        className="p-3 bg-error-red/10 hover:bg-error-red text-error-red hover:text-white rounded-full transition cursor-pointer"
                        title="Stop"
                      >
                        <Square size={16} />
                      </button>
                    </>
                  )}
                </div>

                {isRecording && (
                  <span className="flex items-center gap-1.5 text-[9px] font-bold text-error-red uppercase tracking-wider">
                    <span className="w-2 h-2 rounded-full bg-error-red animate-ping" />
                    <span>Live recording speech sample...</span>
                  </span>
                )}
              </div>

              {/* Show file card if recording was finished and stored */}
              {file && !isRecording && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-bg-sidebar border border-border-color rounded-lg">
                    <div className="flex items-center gap-2.5">
                      <CheckCircle2 size={16} className="text-emerald-500" />
                      <div>
                        <h4 className="text-xs font-bold text-text-primary">Recorded Speech Sample Ready</h4>
                        <p className="text-[9px] text-text-secondary">Duration: {fileDuration} &bull; Size: {formatBytes(file.size)}</p>
                      </div>
                    </div>
                    <button
                      onClick={handleDeleteFile}
                      className="p-1.5 text-text-secondary hover:text-error-red transition cursor-pointer"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                  <AudioPlayer file={file} />
                </div>
              )}
            </div>
          )}
        </section>

        {/* STEP 2: READING MATERIAL CARD */}
        <section className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="flex items-center justify-center w-5 h-5 rounded-full bg-teal-500/10 text-accent-teal text-xs font-bold">2</span>
            <h3 className="text-sm font-bold">Reading Material</h3>
          </div>

          <div className="flex items-start gap-3 p-4 bg-bg-card border border-border-color rounded-xl shadow-xs" style={{ borderRadius: '16px' }}>
            <Info size={16} className="text-accent-teal mt-0.5 shrink-0" />
            <div className="space-y-1">
              <p className="text-xs text-text-primary font-bold">Standardized Clinical Reading</p>
              <p className="text-xs text-text-secondary leading-relaxed">
                The document displayed on the right contains the passage the patient should read aloud while recording. Reading from a standardized passage improves consistency across speech assessments.
              </p>
            </div>
          </div>
        </section>

        {/* STEP 3: ANALYSIS OPTIONS */}
        <section className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="flex items-center justify-center w-5 h-5 rounded-full bg-teal-500/10 text-accent-teal text-xs font-bold">3</span>
            <h3 className="text-sm font-bold">Analysis Options</h3>
          </div>

          <div className="bg-bg-card border border-border-color rounded-xl p-5 space-y-4 shadow-xs" style={{ borderRadius: '16px' }}>

            {/* Analysis Mode Radio Buttons */}
            <div className="space-y-3">
              <label className="text-[10px] font-bold text-text-secondary uppercase tracking-wider block">Select Analysis Mode</label>

              <div className="grid gap-3 sm:grid-cols-2">

                {/* Full Analysis Option */}
                <div
                  onClick={() => setAnalysisMode('full')}
                  className={`p-3.5 border rounded-xl cursor-pointer transition-all duration-200 select-none ${analysisMode === 'full'
                    ? 'border-accent-teal bg-teal-500/5'
                    : 'border-border-color bg-bg-sidebar hover:border-text-secondary'
                    }`}
                  style={{ borderRadius: '12px' }}
                >
                  <div className="flex items-center gap-2">
                    <input
                      type="radio"
                      name="analysisMode"
                      checked={analysisMode === 'full'}
                      onChange={() => setAnalysisMode('full')}
                      className="accent-accent-teal"
                    />
                    <span className="text-xs font-bold">Full Analysis</span>
                    <span className="px-1.5 py-0.5 text-[8px] font-bold bg-teal-500/10 text-accent-teal rounded-full shrink-0">Recommended</span>
                  </div>
                  <p className="text-[10px] text-text-secondary mt-1.5 leading-relaxed">
                    Runs Speech Dysfluency Classification and localization regions mapping.
                  </p>
                </div>

                {/* Classification Only Option */}
                <div
                  onClick={() => setAnalysisMode('classify')}
                  className={`p-3.5 border rounded-xl cursor-pointer transition-all duration-200 select-none ${analysisMode === 'classify'
                    ? 'border-accent-teal bg-teal-500/5'
                    : 'border-border-color bg-bg-sidebar hover:border-text-secondary'
                    }`}
                  style={{ borderRadius: '12px' }}
                >
                  <div className="flex items-center gap-2">
                    <input
                      type="radio"
                      name="analysisMode"
                      checked={analysisMode === 'classify'}
                      onChange={() => setAnalysisMode('classify')}
                      className="accent-accent-teal"
                    />
                    <span className="text-xs font-bold">Classification Only</span>
                  </div>
                  <p className="text-[10px] text-text-secondary mt-1.5 leading-relaxed">
                    Predicts global stuttering categories without localizing timestamps.
                  </p>
                </div>

              </div>
            </div>

            {/* Language Selection */}
            <div className="pt-3 border-t border-border-color/50 space-y-2 select-none">
              <label className="text-[10px] font-bold text-text-secondary uppercase tracking-wider block">Transcription Language</label>
              <div className="flex gap-4 flex-wrap">
                <label className="flex items-center gap-1.5 text-xs font-semibold cursor-pointer">
                  <input
                    type="radio"
                    name="transcriptionLanguage"
                    checked={selectedLanguage === 'english'}
                    onChange={() => setSelectedLanguage('english')}
                    className="accent-accent-teal cursor-pointer"
                  />
                  <span>English</span>
                </label>
                <label className="flex items-center gap-1.5 text-xs font-semibold cursor-pointer">
                  <input
                    type="radio"
                    name="transcriptionLanguage"
                    checked={selectedLanguage === 'kannada'}
                    onChange={() => setSelectedLanguage('kannada')}
                    className="accent-accent-teal cursor-pointer"
                  />
                  <span>ಕನ್ನಡ (Kannada)</span>
                </label>
                <label className="flex items-center gap-1.5 text-xs font-semibold cursor-pointer">
                  <input
                    type="radio"
                    name="transcriptionLanguage"
                    checked={selectedLanguage === 'hindi'}
                    onChange={() => setSelectedLanguage('hindi')}
                    className="accent-accent-teal cursor-pointer"
                  />
                  <span>हिंदी (Hindi)</span>
                </label>
              </div>
            </div>

          </div>
        </section>

        {/* Start Analysis CTA Button */}
        <div className="space-y-3">
          <button
            onClick={handleStartAnalysis}
            disabled={!file}
            className="w-full py-3.5 bg-gradient-to-r from-teal-500 to-teal-600 hover:from-teal-600 hover:to-teal-700 disabled:from-border-color disabled:to-border-color text-white text-xs font-bold rounded-xl transition duration-200 shadow-sm disabled:shadow-none flex items-center justify-center gap-2 cursor-pointer disabled:cursor-not-allowed transform hover:scale-[1.01] active:scale-[0.99]"
            style={{ borderRadius: '12px' }}
          >
            <span>Start Speech Analysis</span>
          </button>

          {/* Secure lock metadata indicator */}
          <div className="flex items-center justify-center gap-1.5 text-[9px] text-text-secondary select-none">
            <ShieldCheck size={13} className="text-accent-teal" />
            <span>Secure End-to-End Processing &bull; Audio samples are not stored</span>
          </div>
        </div>

      </div>

      {/* RIGHT PANE: STANDARD PDF VIEWER */}
      <div className="w-[380px] md:w-[460px] lg:w-[500px] shrink-0 h-full border-l border-border-color bg-bg-sidebar">
        <PdfViewer customPdfs={customPdfs} setCustomPdfs={setCustomPdfs} />
      </div>

    </div>
  )
}
