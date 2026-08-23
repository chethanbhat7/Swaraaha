import { useState, useEffect, useRef } from 'react'
import { BrowserRouter, Routes, Route, Link, useLocation, useNavigate } from 'react-router-dom'
import { retrieveAudioFile, clearAudioFiles } from './utils/db'
import { 
  Activity, 
  Mic, 
  BarChart3, 
  BookOpen, 
  History, 
  Settings, 
  Info, 
  HelpCircle, 
  BookOpenCheck,
  Sun, 
  Moon,
  Sparkles,
  X,
  Volume2,
  Plus
} from 'lucide-react'
import UploadPage from './pages/UploadPage'
import ResultsPage from './pages/ResultsPage'

// Simple mock components for other pages to make the interface complete and rich
function DocumentsPage({
  customPdfs,
  setCustomPdfs
}: {
  customPdfs: Array<{ name: string; url: string }>;
  setCustomPdfs: React.Dispatch<React.SetStateAction<Array<{ name: string; url: string }>>>;
}) {
  const docFileInputRef = useRef<HTMLInputElement>(null)

  const handlePdfUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const newPdfs = Array.from(e.target.files).map(file => ({
        name: file.name,
        url: URL.createObjectURL(file)
      }))
      setCustomPdfs(prev => [...prev, ...newPdfs])
    }
  }

  const handleRemovePdf = (idxToRemove: number) => {
    setCustomPdfs(prev => {
      if (prev[idxToRemove]) {
        URL.revokeObjectURL(prev[idxToRemove].url)
      }
      return prev.filter((_, idx) => idx !== idxToRemove)
    })
  }

  return (
    <div className="p-8 space-y-6 max-w-4xl animate-fade-in text-text-primary">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Reading Documents</h2>
        <p className="text-sm text-text-secondary mt-1">Manage standardized passages used during patient speech assessments.</p>
      </div>
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <div className="p-5 bg-bg-card border border-border-color rounded-xl hover:border-accent-teal transition cursor-pointer">
          <div className="flex items-center justify-between mb-3">
            <span className="px-2.5 py-1 text-[10px] font-semibold text-accent-teal bg-teal-500/10 rounded-full">Standard</span>
            <span className="text-xs text-text-secondary">3 Pages</span>
          </div>
          <h3 className="font-bold text-sm">Speech Assessment Reading Passage</h3>
          <p className="text-xs text-text-secondary mt-1.5 leading-relaxed">Contains the Grandfather Passage, Rainbow Passage, and phonetically balanced sentences.</p>
        </div>
        {customPdfs.map((pdf, idx) => (
          <div key={`doc-pdf-${idx}`} className="p-5 bg-bg-card border border-border-color rounded-xl hover:border-accent-teal transition relative group flex flex-col justify-between min-h-[140px]">
            <div>
              <div className="flex items-center justify-between mb-3">
                <span className="px-2.5 py-1 text-[10px] font-semibold text-emerald-500 bg-emerald-500/10 rounded-full">Custom PDF</span>
                <span className="text-xs text-text-secondary">Document</span>
              </div>
              <h3 className="font-bold text-sm truncate pr-6" title={pdf.name}>{pdf.name}</h3>
              <p className="text-xs text-text-secondary mt-1.5 leading-relaxed line-clamp-2">User-uploaded reading material. Selectable from the speech analysis side pane.</p>
            </div>
            <div className="mt-3">
              <a 
                href={pdf.url} 
                target="_blank" 
                rel="noopener noreferrer" 
                className="text-[11px] font-bold text-accent-teal hover:underline cursor-pointer"
              >
                Open PDF Document
              </a>
            </div>
            <button
              onClick={() => handleRemovePdf(idx)}
              className="absolute top-4 right-4 p-1 rounded-lg bg-hover-color/80 hover:bg-error-red/20 text-text-secondary hover:text-error-red opacity-0 group-hover:opacity-100 transition cursor-pointer"
              title="Remove PDF"
            >
              <X size={14} />
            </button>
          </div>
        ))}

        <div 
          onClick={() => docFileInputRef.current?.click()}
          className="p-5 border-2 border-dashed border-border-color hover:border-accent-teal bg-bg-card/50 hover:bg-teal-500/5 rounded-xl transition cursor-pointer flex flex-col items-center justify-center min-h-[140px] text-center gap-2"
        >
          <input
            ref={docFileInputRef}
            type="file"
            accept=".pdf"
            multiple
            className="hidden"
            onChange={handlePdfUpload}
          />
          <div className="p-2 bg-teal-500/10 text-accent-teal rounded-full">
            <Plus size={18} />
          </div>
          <div>
            <h4 className="text-xs font-bold text-text-primary">Add Custom PDF</h4>
            <p className="text-[10px] text-text-secondary mt-0.5">Upload PDF materials for patient speech reading</p>
          </div>
        </div>
      </div>
    </div>
  )
}

interface HistoryItem {
  id: string
  date: string
  time: string
  name: string
  size: string
  duration: string
  mode: string
  status: string
  results: any
  generateReport: boolean
  patientName?: string
  patientPhone?: string
}

function HistoryPage({ setAnalyzedFile }: { setAnalyzedFile: React.Dispatch<React.SetStateAction<File | null>> }) {
  const navigate = useNavigate()
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([])

  useEffect(() => {
    const historyJson = localStorage.getItem('swaraaha_history')
    if (historyJson) {
      setHistoryItems(JSON.parse(historyJson))
    }
  }, [])

  const handleRowClick = async (item: HistoryItem) => {
    // 1. Retrieve the file from IndexedDB if it exists
    const file = await retrieveAudioFile(item.id)
    setAnalyzedFile(file)

    // 2. Populate sessionStorage with the historical data
    sessionStorage.setItem('results', JSON.stringify(item.results))
    sessionStorage.setItem('filename', item.name)
    sessionStorage.setItem('filesize', item.size)
    sessionStorage.setItem('duration', item.duration)
    sessionStorage.setItem('patient_name', item.patientName || 'N/A')
    sessionStorage.setItem('patient_phone', item.patientPhone || 'N/A')
    sessionStorage.setItem('generate_report', item.generateReport ? 'true' : 'false')

    // 3. Redirect to the results/analysis page
    navigate('/results')
  }

  const handleClearHistory = async () => {
    const confirmed = window.confirm(
      "Are you sure you want to clear your entire assessment history? This will delete all saved audio files and analysis reports permanently."
    )
    if (!confirmed) return

    try {
      localStorage.removeItem('swaraaha_history')
      await clearAudioFiles()
      setHistoryItems([])
    } catch (e) {
      console.error("Failed to clear history:", e)
    }
  }

  return (
    <div className="p-8 space-y-6 max-w-4xl animate-fade-in text-text-primary">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Assessment History</h2>
          <p className="text-sm text-text-secondary mt-1">Review past patient recordings, classifications, and generated reports.</p>
        </div>
        {historyItems.length > 0 && (
          <button
            onClick={handleClearHistory}
            className="px-4 py-2 border border-red-500/20 hover:border-red-500 bg-red-500/5 hover:bg-red-500/10 text-red-500 text-xs font-semibold rounded-lg transition duration-200 cursor-pointer"
          >
            Clear History
          </button>
        )}
      </div>
      <div className="bg-bg-card border border-border-color rounded-xl overflow-hidden shadow-xs">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-border-color bg-bg-sidebar text-text-secondary text-[11px] font-bold uppercase tracking-wider">
              <th className="p-4">Assessment ID</th>
              <th className="p-4">Date & Time</th>
              <th className="p-4">Filename</th>
              <th className="p-4">Analysis Type</th>
              <th className="p-4">Status</th>
            </tr>
          </thead>
          <tbody className="text-xs divide-y divide-border-color">
            {historyItems.length === 0 ? (
              <tr>
                <td colSpan={5} className="p-8 text-center text-text-secondary italic">
                  No assessments recorded yet.
                </td>
              </tr>
            ) : (
              historyItems.map((item) => (
                <tr 
                  key={item.id} 
                  onClick={() => handleRowClick(item)}
                  className="hover:bg-hover-color/50 transition cursor-pointer"
                >
                  <td className="p-4 font-mono font-bold text-accent-teal">{item.id}</td>
                  <td className="p-4 text-text-secondary">{item.date} at {item.time}</td>
                  <td className="p-4 font-medium">{item.name}</td>
                  <td className="p-4 text-text-secondary">{item.mode}</td>
                  <td className="p-4">
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/10 text-emerald-500">
                      {item.status}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function SettingsPage() {
  return (
    <div className="p-8 space-y-6 max-w-2xl animate-fade-in text-text-primary">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">Settings</h2>
        <p className="text-sm text-text-secondary mt-1">Configure speech detection sensitivities, microphone feeds, and defaults.</p>
      </div>
      <div className="bg-bg-card border border-border-color rounded-xl p-6 space-y-6">
        <div className="space-y-2">
          <label className="block text-sm font-bold">Default Input Device</label>
          <select className="w-full text-xs p-2.5 border border-border-color rounded-lg bg-bg-sidebar text-text-primary focus:ring-1 focus:ring-accent-teal focus:outline-none">
            <option>Default System Microphone</option>
            <option>External USB Speech Mic (High Definition)</option>
          </select>
        </div>
        <div className="pt-4 border-t border-border-color flex justify-end gap-3">
          <button className="px-4 py-2 text-xs border border-border-color rounded-lg hover:bg-hover-color cursor-pointer transition">Reset</button>
          <button className="px-4 py-2 text-xs bg-accent-teal text-white font-semibold rounded-lg hover:bg-teal-600 cursor-pointer transition shadow-xs">Save Changes</button>
        </div>
      </div>
    </div>
  )
}

function AboutPage() {
  return (
    <div className="p-8 space-y-6 max-w-2xl animate-fade-in text-text-primary">
      <div>
        <h2 className="text-2xl font-bold tracking-tight">About Swaraaha</h2>
        <p className="text-sm text-text-secondary mt-1">Speech Dysfluency Detection System.</p>
      </div>
      <div className="bg-bg-card border border-border-color rounded-xl p-6 space-y-4 text-xs leading-relaxed text-text-secondary">
        <p>
          <strong className="text-text-primary">Swaraaha</strong> is a modern healthcare AI system designed to assist speech-language pathologists, clinicians, and researchers in identifying, classifiying, and localizing stuttering events.
        </p>
        <p>
          The system leverages deep neural networks trained on clinical speech libraries to detect prolongations, blocks, sound repetitions, word repetitions, and interjections.
        </p>
        <div className="pt-4 border-t border-border-color grid grid-cols-2 gap-4 text-[10px] font-mono">
          <div>
            <span className="block text-text-primary font-bold">System Version</span>
            <span>0.1.0 (Beta)</span>
          </div>
          <div>
            <span className="block text-text-primary font-bold">Model Engine</span>
            <span>Wav2Vec 2.0 Classifiers</span>
          </div>
        </div>
      </div>
    </div>
  )
}

// Sidebar Navigation Element Wrapper
function SidebarLink({ to, icon: Icon, children }: { to: string; icon: any; children: string }) {
  const location = useLocation()
  const isActive = location.pathname === to

  return (
    <Link 
      to={to} 
      className={`relative flex items-center gap-3 px-4 py-2.5 text-xs font-semibold rounded-lg transition duration-200 group cursor-pointer ${
        isActive 
          ? 'bg-teal-500/10 text-accent-teal font-bold' 
          : 'text-text-secondary hover:bg-hover-color hover:text-text-primary'
      }`}
      style={{ borderRadius: '8px' }}
    >
      {/* Left indicator bar */}
      {isActive && (
        <div className="absolute left-0 top-2 bottom-2 w-1 bg-accent-teal rounded-r-md" />
      )}
      <Icon size={16} className={isActive ? 'text-accent-teal' : 'text-text-secondary group-hover:text-text-primary'} />
      <span>{children}</span>
    </Link>
  )
}

function MainAppShell() {
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    return (localStorage.getItem('theme') as 'light' | 'dark') || 'light'
  })
  
  const [showHelpModal, setShowHelpModal] = useState(false)
  const [showHowItWorksModal, setShowHowItWorksModal] = useState(false)
  const [showTipsModal, setShowTipsModal] = useState(false)
  const [customPdfs, setCustomPdfs] = useState<Array<{ name: string; url: string }>>([])
  const [analyzedFile, setAnalyzedFile] = useState<File | null>(null)

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
    localStorage.setItem('theme', theme)
  }, [theme])

  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light')
  }

  return (
    <div className="flex flex-col h-screen bg-bg-app text-text-primary transition-colors overflow-hidden">
      
      {/* TOP NAVIGATION BAR */}
      <header className="flex items-center justify-between h-14 px-6 border-b border-border-color bg-bg-card shrink-0 select-none">
        
        {/* Brand Logo and Subtitle */}
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-teal-500 text-white shadow-sm">
            <Activity size={18} />
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tight text-text-primary flex items-center gap-1.5">
              Swaraaha
            </h1>
            <p className="text-[10px] text-text-secondary font-medium tracking-wide">
              Speech Dysfluency Detection
            </p>
          </div>
        </div>

        {/* Global Toolbar */}
        <div className="flex items-center gap-2">
          
          {/* How it Works Button */}
          <button 
            onClick={() => setShowHowItWorksModal(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-semibold text-text-secondary hover:text-text-primary hover:bg-hover-color border border-border-color rounded-lg cursor-pointer transition"
          >
            <BookOpenCheck size={14} />
            <span>How it Works</span>
          </button>

          {/* Help Button */}
          <button 
            onClick={() => setShowHelpModal(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-semibold text-text-secondary hover:text-text-primary hover:bg-hover-color border border-border-color rounded-lg cursor-pointer transition"
          >
            <HelpCircle size={14} />
            <span>Help</span>
          </button>

          {/* Theme Toggle Button */}
          <button 
            onClick={toggleTheme}
            className="p-2 text-text-secondary hover:text-text-primary hover:bg-hover-color border border-border-color rounded-lg cursor-pointer transition"
            title={theme === 'light' ? "Switch to Dark Mode" : "Switch to Light Mode"}
          >
            {theme === 'light' ? <Moon size={15} /> : <Sun size={15} />}
          </button>
        </div>
      </header>

      {/* BODY SHELL WRAPPER */}
      <div className="flex flex-1 overflow-hidden">
        
        {/* SIDEBAR ON THE LEFT */}
        <aside className="w-[240px] border-r border-border-color bg-bg-sidebar p-4 flex flex-col justify-between shrink-0 select-none">
          
          {/* Sidebar Menu Navigation Links */}
          <div className="space-y-1">
            <div className="px-3 pb-2 text-[10px] font-bold text-text-secondary uppercase tracking-wider">
              Dashboard
            </div>
            <SidebarLink to="/" icon={Mic}>Analyze Speech</SidebarLink>
            <SidebarLink to="/results" icon={BarChart3}>Results</SidebarLink>
            <SidebarLink to="/documents" icon={BookOpen}>Reading Documents</SidebarLink>
            <SidebarLink to="/history" icon={History}>History</SidebarLink>
            <SidebarLink to="/settings" icon={Settings}>Settings</SidebarLink>
            <SidebarLink to="/about" icon={Info}>About</SidebarLink>
          </div>

          {/* Sidebar Bottom Promo Card */}
          <div className="p-4 bg-bg-card border border-border-color rounded-xl space-y-3 shadow-xs" style={{ borderRadius: '16px' }}>
            <div className="flex items-center gap-2 text-accent-teal">
              <Sparkles size={16} />
              <span className="text-xs font-bold">Need Help?</span>
            </div>
            <p className="text-[10px] text-text-secondary leading-relaxed">
              Learn how to perform a proper speech recording for clean analysis.
            </p>
            <button 
              onClick={() => setShowTipsModal(true)}
              className="w-full py-1.5 text-[10px] font-bold text-accent-teal hover:text-white bg-teal-500/10 hover:bg-accent-teal rounded-lg transition cursor-pointer"
            >
              Recording Tips
            </button>
          </div>
        </aside>

        {/* MAIN DISPLAY CONTENT WRAPPER */}
        <main className="flex-1 overflow-y-auto bg-bg-app">
          <Routes>
            <Route path="/" element={<UploadPage customPdfs={customPdfs} setCustomPdfs={setCustomPdfs} setAnalyzedFile={setAnalyzedFile} />} />
            <Route path="/results" element={<ResultsPage analyzedFile={analyzedFile} />} />
            <Route path="/documents" element={<DocumentsPage customPdfs={customPdfs} setCustomPdfs={setCustomPdfs} />} />
            <Route path="/history" element={<HistoryPage setAnalyzedFile={setAnalyzedFile} />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </main>
      </div>

      {/* HELP MODAL POPUP */}
      {showHelpModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 animate-fade-in p-4">
          <div className="bg-bg-card border border-border-color rounded-2xl max-w-md w-full p-6 shadow-2xl space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="text-sm font-bold flex items-center gap-2">
                <HelpCircle className="text-accent-teal" size={18} />
                <span>Frequently Asked Questions</span>
              </h3>
              <button onClick={() => setShowHelpModal(false)} className="p-1 rounded-lg hover:bg-hover-color cursor-pointer text-text-secondary">
                <X size={16} />
              </button>
            </div>
            <div className="text-xs text-text-secondary space-y-4 max-h-[300px] overflow-y-auto pr-1">
              <div>
                <h4 className="font-bold text-text-primary mb-1">What formats are supported?</h4>
                <p>We support WAV, MP3, FLAC, and M4A audio files up to 200 MB in size.</p>
              </div>
              <div>
                <h4 className="font-bold text-text-primary mb-1">Is my audio kept permanently?</h4>
                <p>No. Speech sample recordings are processed securely in temporary memory and are not permanently stored on our servers.</p>
              </div>
              <div>
                <h4 className="font-bold text-text-primary mb-1">How long does analysis take?</h4>
                <p>Typically a few seconds depending on the recording duration and analysis options selected.</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* HOW IT WORKS MODAL */}
      {showHowItWorksModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 animate-fade-in p-4">
          <div className="bg-bg-card border border-border-color rounded-2xl max-w-md w-full p-6 shadow-2xl space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="text-sm font-bold flex items-center gap-2">
                <BookOpenCheck className="text-accent-teal" size={18} />
                <span>How Swaraaha Works</span>
              </h3>
              <button onClick={() => setShowHowItWorksModal(false)} className="p-1 rounded-lg hover:bg-hover-color cursor-pointer text-text-secondary">
                <X size={16} />
              </button>
            </div>
            <div className="text-xs text-text-secondary space-y-3">
              <p>
                Swaraaha automates speech dysfluency diagnostic testing through three simple steps:
              </p>
              <ol className="list-decimal pl-4 space-y-2 font-medium">
                <li>
                  <span className="text-text-primary font-bold">Standardized Passage:</span> Patient reads the text passage displayed on the right panel.
                </li>
                <li>
                  <span className="text-text-primary font-bold">Audio Recording:</span> Audio is recorded live or uploaded via a clean WAV/MP3 file.
                </li>
                <li>
                  <span className="text-text-primary font-bold">AI Diagnostics:</span> The system runs AI classifiers to detect prolongations, blocks, interjections, sound repetitions, and word repetitions.
                </li>
              </ol>
            </div>
          </div>
        </div>
      )}

      {/* RECORDING TIPS MODAL */}
      {showTipsModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 animate-fade-in p-4">
          <div className="bg-bg-card border border-border-color rounded-2xl max-w-md w-full p-6 shadow-2xl space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="text-sm font-bold flex items-center gap-2">
                <Volume2 className="text-accent-teal" size={18} />
                <span>Recording Best Practices</span>
              </h3>
              <button onClick={() => setShowTipsModal(false)} className="p-1 rounded-lg hover:bg-hover-color cursor-pointer text-text-secondary">
                <X size={16} />
              </button>
            </div>
            <ul className="text-xs text-text-secondary space-y-3 list-disc pl-4 leading-relaxed">
              <li>
                <strong className="text-text-primary">Minimize Noise:</strong> Turn off fans, AC units, and perform assessments in a closed, quiet room.
              </li>
              <li>
                <strong className="text-text-primary">Microphone Placement:</strong> Place the mic approximately 6-12 inches away from the mouth to avoid breath clipping.
              </li>
              <li>
                <strong className="text-text-primary">Standardized Materials:</strong> Ensure the patient reads directly from the right-hand PDF passage to preserve phonetic diagnostic properties.
              </li>
              <li>
                <strong className="text-text-primary">Speak Normally:</strong> Instruct the patient to speak with their natural volume and pace without over-articulating.
              </li>
            </ul>
          </div>
        </div>
      )}

    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <MainAppShell />
    </BrowserRouter>
  )
}
