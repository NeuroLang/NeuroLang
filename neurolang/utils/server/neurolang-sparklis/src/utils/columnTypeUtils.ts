/**
 * columnTypeUtils.ts
 *
 * Utilities for parsing and classifying result column types returned by the
 * NeuroLang backend (row_type strings like "<class 'float'>").
 */

/**
 * Parse a row_type string like "<class 'float'>" or "<class 'int'>" into
 * a short label: "float", "int", "str", "VBROverlay", "VBR", "figure", or "?"
 *
 * Note: VBROverlay must be checked before VBR (it is a more-specific type).
 */
export function parseColumnType(rowTypeStr: string): string {
  if (!rowTypeStr) return '?'
  const lower = rowTypeStr.toLowerCase()
  if (lower.includes('explicitvbroverlay')) return 'VBROverlay'
  if (lower.includes('explicitvbr')) return 'VBR'
  if (lower.includes('figure')) return 'figure'
  if (lower.includes('float')) return 'float'
  if (lower.includes('int')) return 'int'
  if (lower.includes('str')) return 'str'
  if (lower.includes('bool')) return 'bool'
  // Fallback: take the last part after the last dot or single-quote
  const match = rowTypeStr.match(/'([^']+)'/)
  if (match) {
    const parts = match[1].split('.')
    return parts[parts.length - 1]
  }
  return '?'
}

/**
 * Returns true if the column type label is a VBR or VBROverlay type.
 */
export function isVbrColumnType(typeLabel: string): boolean {
  return typeLabel === 'VBR' || typeLabel === 'VBROverlay'
}
