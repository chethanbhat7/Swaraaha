import { useState, useRef, useEffect } from 'react'
import { 
  ChevronLeft, 
  ChevronRight, 
  ZoomIn, 
  ZoomOut, 
  Search, 
  Download, 
  Maximize2, 
  Minimize2, 
  FileText,
  CheckCircle2,
  Plus,
  X
} from 'lucide-react'

interface PageData {
  title: string
  subtitle: string
  sections: Array<{
    heading: string
    content: string | string[]
  }>
}

const PAGES: PageData[] = [
  {
    title: "Speech Assessment Reading Passage",
    subtitle: "Section I: Standardized Continuous Speech Passages",
    sections: [
      {
        heading: "Instructions for Patient",
        content: "Read the following passages aloud in your normal speaking voice. Maintain a natural, comfortable pace. Try to read without stopping, even if you stumble or repeat words."
      },
      {
        heading: "1. The Grandfather Passage",
        content: "You wish to know all about my grandfather. Well, he is nearly ninety-three years old. He limits his spoon to three bites of food, yet he still thinks as swiftly as ever. A long, flowing beard clings to his chin, giving those who observe him a pronounced feeling of the utmost respect. When he speaks, his voice is just a bit cracked and quivers a trifle. In a double wooden box, he keeps an old, rusty knife which he uses to shave his grandfather's plug tobacco. Every day he plays a short round of golf on the links near the house. We have often urged him to walk more and play less, but he only nods his head and smiles. He is a grandfather we all love dearly."
      },
      {
        heading: "2. The Rainbow Passage",
        content: "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end. People look, but no one ever finds it. When a man looks for something beyond his reach, his friends say he is looking for the pot of gold at the end of the rainbow."
      }
    ]
  },
  {
    title: "Phonetically Balanced Sentences",
    subtitle: "Section II: Standard Speech Assessment Sentences",
    sections: [
      {
        heading: "Clinical Objective",
        content: "These sentences are phonetically balanced and contain key phoneme combinations frequently associated with speech dysfluencies (such as prolongations on sibilants, blocks on plosives, and word repetitions)."
      },
      {
        heading: "Reading Sentences",
        content: [
          "1. The birch canoe slid on the smooth planks.",
          "2. Glue the sheet to the dark blue background.",
          "3. It's easy to tell the depth of a well.",
          "4. These days a chicken leg is a rare dish.",
          "5. Rice is often served in round bowls.",
          "6. The juice of lemons makes fine punch.",
          "7. The box was thrown beside the park gate.",
          "8. The boy was silent when the grandfather spoke.",
          "9. A king ruled the state with a fair hand.",
          "10. Four hours of steady work faced us."
        ]
      }
    ]
  },
  {
    title: "Diagnostic Reading Exercises",
    subtitle: "Section III: Complex Speech Patterns",
    sections: [
      {
        heading: "Exercise A: Prolongation Detection (Sibilants & Fricatives)",
        content: "Sally saw seven silly sheep sleeping silently in the summer sunshine. The sheep seemed somewhat sleepy, so Sally walked slowly past them, trying not to startle their quiet slumber."
      },
      {
        heading: "Exercise B: Plosive & Block Detection (Stop Consonants)",
        content: "Peter Piper picked a peck of pickled peppers. A peck of pickled peppers Peter Piper picked. If Peter Piper picked a peck of pickled peppers, where is the peck of pickled peppers Peter Piper picked?"
      },
      {
        heading: "Exercise C: Conversational Flow & Transition Phrases",
        content: "Yesterday morning, I woke up early, made a cup of warm tea, and went out for a brisk walk. The crisp air was refreshing, and the birds were singing in the trees. It was a perfect start to a busy day."
      }
    ]
  }
]

export default function PdfViewer({
  customPdfs,
  setCustomPdfs
}: {
  customPdfs: Array<{ name: string; url: string }>;
  setCustomPdfs: React.Dispatch<React.SetStateAction<Array<{ name: string; url: string }>>>;
}) {
  const [currentPage, setCurrentPage] = useState(0)
  const [zoom, setZoom] = useState(100)
  const [searchTerm, setSearchTerm] = useState('')
  const [searchFocused, setSearchFocused] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [currentCustomPdfIndex, setCurrentCustomPdfIndex] = useState(0)
  const [activeDocType, setActiveDocType] = useState<'standard' | 'custom'>('standard')
  const containerRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handlePrevPage = () => {
    if (currentPage > 0) setCurrentPage(currentPage - 1)
  }

  const handleNextPage = () => {
    if (currentPage < PAGES.length - 1) setCurrentPage(currentPage + 1)
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
      if (containerRef.current.requestFullscreen) {
        containerRef.current.requestFullscreen()
      }
      setIsFullscreen(true)
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen()
      }
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
        // Automatically switch to the first newly uploaded PDF
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
          setCurrentPage(0)
        }
      } else if (activeDocType === 'custom' && currentCustomPdfIndex > idxToRemove) {
        setCurrentCustomPdfIndex(prevIndex => prevIndex - 1)
      }
      return updated
    })
  }

  const handleDownload = () => {
    if (activeDocType === 'custom' && customPdfs[currentCustomPdfIndex]) {
      const activePdf = customPdfs[currentCustomPdfIndex]
      const a = document.createElement('a')
      a.href = activePdf.url
      a.download = activePdf.name
      a.click()
      return
    }

    const textContent = PAGES.map((p, idx) => {
      return `PAGE ${idx + 1}: ${p.title}\n${p.subtitle}\n\n` + 
        p.sections.map(s => {
          const contentStr = Array.isArray(s.content) ? s.content.join('\n') : s.content
          return `--- ${s.heading} ---\n${contentStr}\n`
        }).join('\n')
    }).join('\n=============================\n\n')

    const blob = new Blob([textContent], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'Speech_Assessment_Passage.txt'
    a.click()
    URL.revokeObjectURL(url)
  }

  const highlightText = (text: string, search: string) => {
    if (!search.trim()) return <span>{text}</span>
    const regex = new RegExp(`(${search.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&')})`, 'gi')
    const parts = text.split(regex)
    return (
      <>
        {parts.map((part, i) => 
          regex.test(part) ? (
            <mark key={i} className="bg-amber-300 dark:bg-amber-600/80 text-black dark:text-white rounded-xs px-0.5 font-semibold">
              {part}
            </mark>
          ) : (
            part
          )
        )}
      </>
    )
  }

  const activePage = PAGES[currentPage] || PAGES[0]

  return (
    <div 
      ref={containerRef}
      className={`flex flex-col h-full bg-bg-sidebar border-l border-border-color ${isFullscreen ? 'p-6 bg-bg-app' : ''}`}
      style={{ minWidth: isFullscreen ? '100vw' : 'auto' }}
    >
      <div className="flex flex-wrap items-center justify-between gap-2 p-3 border-b border-border-color bg-bg-card animate-fade-in">
        <div className="flex items-center gap-1">
          <button
            onClick={handlePrevPage}
            disabled={activeDocType !== 'standard' || currentPage === 0}
            className="p-1.5 rounded-lg text-text-secondary hover:bg-hover-color disabled:opacity-40 transition cursor-pointer"
            title="Previous Page"
          >
            <ChevronLeft size={18} />
          </button>
          <span className="text-xs font-medium text-text-primary px-1 select-none">
            {activeDocType === 'standard' 
              ? `Page ${currentPage + 1} of ${PAGES.length}`
              : `PDF Document`
            }
          </span>
          <button
            onClick={handleNextPage}
            disabled={activeDocType !== 'standard' || currentPage === PAGES.length - 1}
            className="p-1.5 rounded-lg text-text-secondary hover:bg-hover-color disabled:opacity-40 transition cursor-pointer"
            title="Next Page"
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

        <div className={`relative flex items-center border border-border-color rounded-lg bg-bg-sidebar transition px-2 py-1 max-w-[140px] md:max-w-[180px] ${searchFocused ? 'ring-1 ring-accent-teal border-accent-teal' : ''}`}>
          <Search size={14} className="text-text-secondary mr-1.5 shrink-0" />
          <input
            type="text"
            placeholder="Search passage..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            onFocus={() => setSearchFocused(true)}
            onBlur={() => setSearchFocused(false)}
            className="bg-transparent text-xs w-full text-text-primary focus:outline-none"
          />
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
        {activeDocType === 'standard' ? (
          <div 
            className="w-full bg-white text-gray-900 border border-gray-200 shadow-md rounded-xl transition-all duration-200 ease-out origin-top flex flex-col h-fit"
            style={{ 
              width: '100%',
              maxWidth: `${(zoom / 100) * 560}px`,
              fontSize: `${(zoom / 100) * 0.875}rem`,
              lineHeight: '1.6',
              borderRadius: `${(zoom / 100) * 16}px`,
              padding: `${(zoom / 100) * 2}rem`
            }}
          >
            <div className="flex items-center gap-3 border-b border-gray-100 pb-5 mb-5 shrink-0">
              <div className="p-2 bg-teal-50 text-teal-600 rounded-lg">
                <FileText size={24} />
              </div>
              <div>
                <h1 className="font-bold tracking-tight text-gray-900 uppercase" style={{ fontSize: '1.25em' }}>
                  {activePage.title}
                </h1>
                <p className="font-semibold text-teal-600 uppercase tracking-wider" style={{ fontSize: '0.8em' }}>
                  {activePage.subtitle}
                </p>
              </div>
            </div>

            <div className="space-y-6 flex-1 text-gray-800">
              {activePage.sections.map((sec, idx) => (
                <section key={idx} className="space-y-2">
                  <h3 className="font-bold text-gray-900 border-l-2 border-teal-500 pl-2" style={{ fontSize: '1.05em' }}>
                    {highlightText(sec.heading, searchTerm)}
                  </h3>
                  {Array.isArray(sec.content) ? (
                    <ul className="space-y-1.5 text-gray-700 font-mono" style={{ fontSize: '0.85em' }}>
                      {sec.content.map((line, lIdx) => (
                        <li key={lIdx} className="hover:bg-teal-50/50 p-1 rounded-md transition pl-1">
                          {highlightText(line, searchTerm)}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-gray-700 leading-relaxed text-justify" style={{ fontSize: '0.85em' }}>
                      {highlightText(sec.content, searchTerm)}
                    </p>
                  )}
                </section>
              ))}
            </div>

            <div className="border-t border-gray-100 pt-5 mt-8 flex items-center justify-between text-gray-400 select-none shrink-0" style={{ fontSize: '0.7em' }}>
              <span>Swaraaha Medical Assessment System</span>
              <div className="flex items-center gap-1 text-emerald-600 font-medium">
                <CheckCircle2 size={12} />
                <span>Standardized Passage v1.2</span>
              </div>
            </div>
          </div>
        ) : (
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
              {customPdfs[currentCustomPdfIndex] ? (
                <iframe
                  src={customPdfs[currentCustomPdfIndex].url}
                  className="w-full h-full border border-border-color rounded-xl bg-white shadow-xs"
                  title={customPdfs[currentCustomPdfIndex].name}
                />
              ) : (
                <div className="flex flex-col items-center justify-center gap-2 text-text-secondary text-xs h-full bg-bg-card rounded-xl border border-border-color">
                  No PDF selected or loaded.
                </div>
              )}
            </div>
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

          {PAGES.map((page, idx) => (
            <button
              key={`standard-${idx}`}
              onClick={() => {
                setActiveDocType('standard');
                setCurrentPage(idx);
              }}
              className={`flex-none text-left w-36 p-2 rounded-lg border text-xs transition duration-200 cursor-pointer ${
                activeDocType === 'standard' && currentPage === idx 
                  ? 'border-accent-teal bg-teal-500/5 text-text-primary ring-1 ring-accent-teal font-semibold' 
                  : 'border-border-color bg-bg-sidebar text-text-secondary hover:border-text-secondary'
              }`}
              style={{ borderRadius: '8px' }}
            >
              <div className="font-semibold truncate">Page {idx + 1}</div>
              <div className="text-[10px] truncate opacity-80">{page.title}</div>
            </button>
          ))}

          {customPdfs.map((pdf, idx) => (
            <div key={`custom-container-${idx}`} className="relative flex-none group">
              <button
                onClick={() => {
                  setActiveDocType('custom');
                  setCurrentCustomPdfIndex(idx);
                }}
                className={`text-left w-36 p-2 rounded-lg border text-xs transition duration-200 cursor-pointer ${
                  activeDocType === 'custom' && currentCustomPdfIndex === idx 
                    ? 'border-accent-teal bg-teal-500/5 text-text-primary ring-1 ring-accent-teal font-semibold' 
                    : 'border-border-color bg-bg-sidebar text-text-secondary hover:border-text-secondary'
                }`}
                style={{ borderRadius: '8px' }}
              >
                <div className="font-semibold truncate pr-4">PDF {idx + 1}</div>
                <div className="text-[10px] truncate opacity-80">{pdf.name}</div>
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleRemovePdf(idx);
                }}
                className="absolute top-1 right-1 p-0.5 rounded-full bg-hover-color/80 hover:bg-error-red/20 text-text-secondary hover:text-error-red opacity-0 group-hover:opacity-100 transition cursor-pointer flex items-center justify-center w-4 h-4"
                title="Remove PDF"
              >
                <X size={10} />
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
