const API_BASE = '/api'

export interface TranscriptionChunk {
  text: string;
  start: number;
  end: number;
  language: string;
}

export interface TranscriptionData {
  text: string;
  language: string;
  chunks: TranscriptionChunk[];
}

export interface ClassificationResults {
  [className: string]: {
    label: number;
    confidence: number;
  };
}

export interface AnalyzeResults {
  classification: ClassificationResults;
  localization: {
    regions: Array<{ start: number; end: number; confidence: number }>;
  };
  transcription: TranscriptionData;
}

export async function classifyAudio(file: File, language: string = 'english'): Promise<{ classification: ClassificationResults; transcription: TranscriptionData }> {
  const form = new FormData()
  form.append('file', file)
  form.append('language', language)
  const res = await fetch(`${API_BASE}/classify`, { method: 'POST', body: form })
  return res.json()
}

export async function localizeAudio(file: File): Promise<{ regions: Array<{ start: number; end: number; confidence: number }> }> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/localize`, { method: 'POST', body: form })
  return res.json()
}

export async function analyzeAudio(file: File, language: string = 'english'): Promise<AnalyzeResults> {
  const form = new FormData()
  form.append('file', file)
  form.append('language', language)
  const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: form })
  return res.json()
}
