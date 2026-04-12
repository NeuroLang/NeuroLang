/**
 * overlayUtils.ts
 *
 * Utility functions for brain overlay management.
 */
import { type BrainOverlay, type OverlayColormap } from '../context/BrainOverlayContext'

const COLORMAP_CYCLE: OverlayColormap[] = [
  'hot',
  'blue',
  'green',
  'red',
  'yellow',
  'cyan',
  'pink',
  'violet',
]

/**
 * Given the current list of overlays, return the next colormap in the cycle
 * that is not already used.  Falls back to the first in the cycle.
 */
export function nextColormap(overlays: BrainOverlay[]): OverlayColormap {
  const used = new Set(overlays.map((o) => o.colormap))
  return COLORMAP_CYCLE.find((c) => !used.has(c)) ?? COLORMAP_CYCLE[0]
}
