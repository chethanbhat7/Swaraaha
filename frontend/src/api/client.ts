const API_BASE = '/api'

export async function classifyAudio(file: File): Promise<Record<string, { label: number; confidence: number }>> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/classify`, { method: 'POST', body: form })
  return res.json()
}

export async function localizeAudio(file: File): Promise<{ regions: Array<{ start: number; end: number; confidence: number }> }> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/localize`, { method: 'POST', body: form })
  return res.json()
}

export async function analyzeAudio(file: File) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: form })
  return res.json()
}
