import { useNavigate } from 'react-router-dom'
import { useEffect, useState } from 'react'

interface AnalysisResults {
  classification: Record<string, { label: number; confidence: number }>
  localization: { regions: Array<{ start: number; end: number; confidence: number }> }
}

const CLASS_NAMES = ['prolongation', 'block', 'soundrep', 'wordrep', 'interjection']

export default function ResultsPage() {
  const navigate = useNavigate()
  const [results, setResults] = useState<AnalysisResults | null>(null)
  const [filename, setFilename] = useState('')

  useEffect(() => {
    const data = sessionStorage.getItem('results')
    const name = sessionStorage.getItem('filename')
    if (!data) {
      navigate('/')
      return
    }
    setResults(JSON.parse(data))
    setFilename(name || '')
  }, [navigate])

  if (!results) return null

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Results for {filename}</h2>
        <button
          onClick={() => navigate('/')}
          className="text-blue-600 hover:underline"
        >
          Upload another
        </button>
      </div>

      <section>
        <h3 className="text-lg font-medium mb-3">Classification</h3>
        <div className="space-y-2">
          {CLASS_NAMES.map((name) => {
            const r = results.classification[name]
            return (
              <div key={name} className="flex items-center gap-3">
                <span className="w-28 text-sm font-mono">{name}</span>
                <div className="flex-1 bg-gray-200 rounded-full h-4">
                  <div
                    className="bg-blue-600 h-4 rounded-full transition-all"
                    style={{ width: `${r.confidence * 100}%` }}
                  />
                </div>
                <span className="w-12 text-sm text-right">{(r.confidence * 100).toFixed(1)}%</span>
                <span className={`text-xs ${r.label ? 'text-red-600' : 'text-gray-400'}`}>
                  {r.label ? 'Present' : 'Absent'}
                </span>
              </div>
            )
          })}
        </div>
      </section>

      <section>
        <h3 className="text-lg font-medium mb-3">Localization</h3>
        {results.localization.regions.length === 0 ? (
          <p className="text-gray-500">No dysfluency regions detected.</p>
        ) : (
          <div className="space-y-2">
            {results.localization.regions.map((r, i) => (
              <div key={i} className="flex items-center gap-4 p-3 bg-red-50 rounded-lg">
                <span className="font-mono text-sm">
                  {r.start.toFixed(2)}s — {r.end.toFixed(2)}s
                </span>
                <span className="text-sm text-gray-600">
                  ({(r.confidence * 100).toFixed(1)}% confidence)
                </span>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
