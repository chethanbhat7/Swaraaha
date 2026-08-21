export function resolveApiBase(raw?: string): string {
  const value = raw?.trim()
  if (!value) return '/api'
  const base = value.replace(/\/+$/, '')
  if (!base) return '/api'
  return base.endsWith('/api') ? base : `${base}/api`
}
