import { useContext } from 'react'
import BrainOverlayContext, {
  type BrainOverlayContextValue,
} from './BrainOverlayContext'

export function useBrainOverlay(): BrainOverlayContextValue {
  const ctx = useContext(BrainOverlayContext)
  if (!ctx) {
    throw new Error('useBrainOverlay must be used within a BrainOverlayProvider')
  }
  return ctx
}
