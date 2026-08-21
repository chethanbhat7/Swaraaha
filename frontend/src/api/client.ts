const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api'

async function parseOrThrow(res: Response) {
  if (!res.ok) {
    throw new Error(`Request failed with status ${res.status}`)
  }
  return res.json()
}

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
    prob_present?: number;
    prob_not_present?: number;
  };
}

export interface SeverityResult {
  index_pct: number;
  severity: 'fluent' | 'mild' | 'moderate' | 'severe';
  label: string;
}

export type RegionType =
  | 'prolongation'
  | 'block'
  | 'soundrep'
  | 'wordrep'
  | 'interjection'

export interface LocalizationRegion {
  start: number;
  end: number;
  confidence: number;
  type?: RegionType;
  coarse?: boolean;
}

export interface CombinedClass {
  label: number;
  confidence: number;
  prob_present: number;
  prob_not_present: number;
}

export interface CombinedRegion {
  start: number;
  end: number;
  confidence: number;
  primary_type: string | null;
  classes: Record<string, CombinedClass>;
  syllables: never[];
}

export interface CombinedResults {
  regions: CombinedRegion[];
  audio_duration: number;
  total_stutters: number;
}

export interface AnalyzeResults {
  classification: ClassificationResults;
  localization: {
    regions: LocalizationRegion[];
    duration_sec?: number;
  };
  transcription: TranscriptionData;
  severity?: SeverityResult;
  combined?: CombinedResults;
}

export async function classifyAudio(file: File, language: string = 'english'): Promise<{ classification: ClassificationResults; transcription: TranscriptionData }> {
  const form = new FormData()
  form.append('file', file)
  form.append('language', language)
  const res = await fetch(`${API_BASE}/classify`, { method: 'POST', body: form })
  return parseOrThrow(res)
}

export async function localizeAudio(file: File): Promise<{ regions: LocalizationRegion[] }> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/localize`, { method: 'POST', body: form })
  return parseOrThrow(res)
}

export async function analyzeAudio(file: File, language: string = 'english'): Promise<AnalyzeResults> {
  const form = new FormData()
  form.append('file', file)
  form.append('language', language)
  const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: form })
  return parseOrThrow(res)
}

export interface ReportData {
  patient: {
    name: string;
    phone: string;
  };
  audio: {
    filename: string;
    size: string;
    duration: string;
  };
  date: string;
  classification: ClassificationResults;
  severity?: SeverityResult;
  combined?: CombinedResults;
  transcription?: TranscriptionData;
  localization?: {
    regions: Array<{ start: number; end: number; confidence: number }>;
  };
}

export async function downloadReport(data: ReportData): Promise<Blob> {
  const res = await fetch(`${API_BASE}/report`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!res.ok) {
    throw new Error(`Request failed with status ${res.status}`)
  }
  return res.blob()
}
