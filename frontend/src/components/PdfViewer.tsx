import { useState, useRef, useEffect } from 'react'
import { 
  ChevronLeft, 
  ChevronRight, 
  ZoomIn, 
  ZoomOut, 
  Download, 
  Maximize2, 
  Minimize2, 
  Plus,
  X
} from 'lucide-react'
import readingDocuments from '../data/readingDocuments.json'

interface CustomPdf {
  name: string
  url: string
}

export default function PdfViewer({
  customPdfs,
  setCustomPdfs
}: {
  customPdfs: CustomPdf[];
  setCustomPdfs: React.Dispatch<React.SetStateAction<CustomPdf[]>>;
}) {
  const [currentDocIndex, setCurrentDocIndex] = useState(0)
  const [zoom, setZoom] = useState(100)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [currentCustomPdfIndex, setCurrentCustomPdfIndex] = useState(0)
  const [activeDocType, setActiveDocType] = useState<'standard' | 'custom'>('standard')
  const containerRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const standardDoc = readingDocuments[Math.min(currentDocIndex, readingDocuments.length - 1)]

  const activeStandard = activeDocType === 'standard' && readingDocuments.length > 0
  const activeCustom = activeDocType === 'custom' && customPdfs.length > 0

  const activeSrc = activeStandard
    ? `/documents/${standardDoc.id}.pdf`
    : activeCustom
      ? customPdfs[currentCustomPdfIndex]?.url
      : null
  const activeName = activeStandard
    ? standardDoc.title
    : activeCustom
      ? customPdfs[currentCustomPdfIndex]?.name
      : null

  const handlePrev = () => {
    if (activeDocType === 'standard') {
      if (currentDocIndex > 0) setCurrentDocIndex(currentDocIndex - 1)
    } else if (currentCustomPdfIndex > 0) {
      setCurrentCustomPdfIndex(currentCustomPdfIndex - 1)
    }
  }

  const handleNext = () => {
    if (activeDocType === 'standard') {
      if (currentDocIndex < readingDocuments.length - 1) setCurrentDocIndex(currentDocIndex + 1)
    } else if (currentCustomPdfIndex < customPdfs.length - 1) {
      setCurrentCustomPdfIndex(currentCustomPdfIndex + 1)
    }
  }

  const handleZoomIn = () => {
    if (zoom < 150) setZoom(zoom + 10)
  }

  const handleZoomOut = () => {
    if (zoom > 70) setZoom(zoom - 10)
  }

  const toggleFullscreen = () => {
    if (!containerRef.current) return
    if (!isFullscreen) {
      containerRef.current.requestFullscreen?.()
      setIsFullscreen(true)
    } else {
      document.exitFullscreen?.()
      setIsFullscreen(false)
    }
  }

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement)
    }
    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange)
    }
  }, [])

  const handlePdfUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const newPdfs = Array.from(e.target.files).map(file => ({
        name: file.name,
        url: URL.createObjectURL(file)
      }))
      setCustomPdfs(prev => {
        const updated = [...prev, ...newPdfs]
        setActiveDocType('custom')
        setCurrentCustomPdfIndex(prev.length)
        return updated
      })
    }
  }

  const handleRemovePdf = (idxToRemove: number) => {
    setCustomPdfs(prev => {
      if (prev[idxToRemove]) {
        URL.revokeObjectURL(prev[idxToRemove].url)
      }
      const updated = prev.filter((_, idx) => idx !== idxToRemove)

      if (activeDocType === 'custom' && currentCustomPdfIndex === idxToRemove) {
        if (updated.length > 0) {
          setCurrentCustomPdfIndex(Math.max(0, idxToRemove - 1))
        } else {
          setActiveDocType('standard')
          setCurrentDocIndex(0)
        }
      } else if (activeDocType === 'custom' && currentCustomPdfIndex > idxToRemove) {
        setCurrentCustomPdfIndex(prevIndex => prevIndex - 1)
      }
      return updated
    })
  }

  const handleDownload = () => {
    if (activeCustom && customPdfs[currentCustomPdfIndex]) {
      const activePdf = customPdfs[currentCustomPdfIndex]
      const a = document.createElement('a')
      a.href = activePdf.url
      a.download = activePdf.name
      a.click()
      return
    }
    if (activeSrc) {
      const a = document.createElement('a')
      a.href = activeSrc
      a.download = `${standardDoc.id}.pdf`
      a.click()
    }
  }

  return (
    <div 
      ref={containerRef}
      className={`flex flex-col h-full bg-bg-sidebar border-l border-border-color ${isFullscreen ? 'p-6 bg-bg-app' : ''}`}
      style={{ minWidth: isFullscreen ? '100vw' : 'auto' }}
    >
      <div className="flex flex-wrap items-center justify-between gap-2 p-3 border-b border-border-color bg-bg-card animate-fade-in">
        <div className="flex items-center gap-1">
          <button
            onClick={handlePrev}
            disabled={activeDocType === 'standard' ? currentDocIndex === 0 : currentCustomPdfIndex === 0}
            className="p-1.5 rounded-lg text-text-secondary hover:bg-hover-color disabled:opacity-40 transition cursor-pointer"
            title="Previous Document"
          >
            <ChevronLeft size={18} />
          </button>
          <span className="text-xs font-medium text-text-primary px-1 select-none">
            {activeDocType === 'standard'
              ? `Document ${currentDocIndex + 1} of ${readingDocuments.length}`
              : `PDF Document`
            }
          </span>
          <button
            onClick={handleNext}
            disabled={activeDocType === 'standard'
              ? currentDocIndex >= readingDocuments.length - 1
              : currentCustomPdfIndex >= customPdfs.length - 1}
            className="p-1.5 rounded-lg text-text-secondary hover:bg-hover-color disabled:opacity-40 transition cursor-pointer"
            title="Next Document"
          >
            <ChevronRight size={18} />
          </button>
        </div>

        <div className="flex items-center gap-1 border-l border-r border-border-color px-2">
          <button
            onClick={handleZoomOut}
            disabled={zoom <= 70}
            className="p-1.5 rounded-lg text-text-secondary hover:bg-hover-color disabled:opacity-40 transition cursor-pointer"
            title="Zoom Out"
          >
            <ZoomOut size={16} />
          </button>
          <span className="text-xs font-mono w-10 text-center select-none text-text-secondary">
            {zoom}%
          </span>
          <button
            onClick={handleZoomIn}
            disabled={zoom >= 150}
            className="p-1.5 rounded-lg text-text-secondary hover:bg-hover-color disabled:opacity-40 transition cursor-pointer"
            title="Zoom In"
          >
            <ZoomIn size={16} />
          </button>
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={handleDownload}
            className="p-1.5 rounded-lg text-text-secondary hover:bg-hover-color transition cursor-pointer"
            title="Download Document"
          >
            <Download size={18} />
          </button>
          <button
            onClick={toggleFullscreen}
            className="p-1.5 rounded-lg text-text-secondary hover:bg-hover-color transition cursor-pointer"
            title={isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
          >
            {isFullscreen ? <Minimize2 size={18} /> : <Maximize2 size={18} />}
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-auto p-6 flex justify-center bg-bg-app relative select-text">
        {activeSrc ? (
          <div className="w-full h-full flex items-center justify-center overflow-auto relative">
            <div
              style={{
                transform: `scale(${zoom / 100})`,
                transformOrigin: 'top center',
                width: `${100 / (zoom / 100)}%`,
                height: `${100 / (zoom / 100)}%`,
                transition: 'transform 0.2s ease-out'
              }}
              className="w-full h-full"
            >
              <iframe
                src={activeSrc}
                className="w-full h-full border border-border-color rounded-xl bg-white shadow-xs"
                title={activeName ?? 'Document'}
              />
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center gap-2 text-text-secondary text-xs h-full bg-bg-card rounded-xl border border-border-color">
            No document selected or loaded.
          </div>
        )}
      </div>

      <div className="p-3 border-t border-border-color bg-bg-card flex flex-col gap-2 select-none">
        <span className="text-[10px] font-semibold text-text-secondary uppercase tracking-wider">
          Document Outline / Pages
        </span>
        <div className="flex gap-3 overflow-x-auto pb-1 items-center">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="flex-none flex flex-col items-center justify-center w-28 h-[52px] p-2 rounded-lg border border-dashed border-accent-teal text-accent-teal bg-teal-500/5 hover:bg-teal-500/10 transition duration-200 cursor-pointer text-[10px] font-bold"
            style={{ borderRadius: '8px' }}
            title="Upload multiple PDFs"
          >
            <Plus size={14} className="mb-0.5" />
            <span>Add PDF(s)</span>
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf"
            multiple
            className="hidden"
            onChange={handlePdfUpload}
          />

          {readingDocuments.map((doc) => {
            const originalIndex = readingDocuments.indexOf(doc)
            return (
              <button
                key={`standard-${doc.id}`}
                onClick={() => {
                  setActiveDocType('standard')
                  setCurrentDocIndex(originalIndex)
                }}
                className={`flex-none text-left w-36 p-2 rounded-lg border text-xs transition duration-200 cursor-pointer ${
                  activeStandard && currentDocIndex === originalIndex
                    ? 'border-accent-teal bg-teal-500/5 text-text-primary ring-1 ring-accent-teal font-semibold'
                    : 'border-border-color bg-bg-sidebar text-text-secondary hover:border-text-secondary'
                }`}
                style={{ borderRadius: '8px' }}
              >
                <div className="font-semibold truncate">Document {originalIndex + 1}</div>
                <div className="text-[10px] truncate opacity-80">{doc.title}</div>
              </button>
            )
          })}

          {customPdfs.map((pdf) => {
            const originalIndex = customPdfs.indexOf(pdf)
            return (
              <div key={`custom-container-${originalIndex}`} className="relative flex-none group">
                <button
                  onClick={() => {
                    setActiveDocType('custom')
                    setCurrentCustomPdfIndex(originalIndex)
                  }}
                  className={`text-left w-36 p-2 rounded-lg border text-xs transition duration-200 cursor-pointer ${
                    activeCustom && currentCustomPdfIndex === originalIndex
                      ? 'border-accent-teal bg-teal-500/5 text-text-primary ring-1 ring-accent-teal font-semibold'
                      : 'border-border-color bg-bg-sidebar text-text-secondary hover:border-text-secondary'
                  }`}
                  style={{ borderRadius: '8px' }}
                >
                  <div className="font-semibold truncate pr-4">PDF {originalIndex + 1}</div>
                  <div className="text-[10px] truncate opacity-80">{pdf.name}</div>
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    handleRemovePdf(originalIndex)
                  }}
                  className="absolute top-1 right-1 p-0.5 rounded-full bg-hover-color/80 hover:bg-error-red/20 text-text-secondary hover:text-error-red opacity-0 group-hover:opacity-100 transition cursor-pointer flex items-center justify-center w-4 h-4"
                  title="Remove PDF"
                >
                  <X size={10} />
                </button>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
