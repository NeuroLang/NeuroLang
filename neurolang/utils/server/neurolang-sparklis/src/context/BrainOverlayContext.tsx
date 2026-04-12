/**
 * BrainOverlayContext.tsx
 *
 * Manages the list of brain overlays added to the Niivue viewer.
 * Supports up to MAX_OVERLAYS simultaneous overlays.
 *
 * Each overlay has:
 *   - id: unique identifier (e.g. "symbolName:rowIndex")
 *   - name: display name (e.g. "region_left [0]")
 *   - base64: base64-encoded NIfTI data
 *   - colormap: "hot" for VBROverlay (probability), "blue" for VBR
 *   - isOverlay: true for VBROverlay (probability), false for plain VBR
 *
 * Consumers:
 *   - DataTable: calls addOverlay() when "View in brain" is clicked
 *   - BrainViewer: subscribes to overlays to add/remove from Niivue
 *   - OverlayManager: lists overlays, calls removeOverlay()
 */
import React, { createContext, useState, useCallback } from 'react'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export const MAX_OVERLAYS = 8

export type OverlayColormap = 'hot' | 'blue' | 'green' | 'red' | 'yellow' | 'cyan' | 'pink' | 'violet'

/** A single brain overlay entry. */
export interface BrainOverlay {
  /** Unique ID for this overlay (e.g., "symbolName:rowIndex:colIndex"). */
  id: string
  /** Human-readable display name shown in the overlay manager. */
  name: string
  /** Base64-encoded NIfTI image data. */
  base64: string
  /** Color map used to render the overlay. */
  colormap: OverlayColormap
  /** True if this is a VBROverlay (probability data). */
  isProbabilistic: boolean
  /** Visibility state of the overlay. */
  visible: boolean
  /** Lower threshold for overlay display (0-1 range). */
  threshold: number
  /** Minimum value for colorbar scale. */
  colorbarMin: number
  /** Maximum value for colorbar scale. */
  colorbarMax: number
}

export interface BrainOverlayContextValue {
  /** Currently active overlays (up to MAX_OVERLAYS). */
  overlays: BrainOverlay[]
  /**
   * Add a new overlay. If MAX_OVERLAYS is already reached, the oldest overlay
   * is automatically removed to make room.
   */
  addOverlay: (overlay: BrainOverlay) => void
  /** Remove an overlay by ID. */
  removeOverlay: (id: string) => void
  /** Remove all overlays. */
  clearOverlays: () => void
  /** Toggle overlay visibility. */
  toggleOverlayVisibility: (id: string) => void
  /** Update overlay threshold. */
  updateOverlayThreshold: (id: string, threshold: number) => void
  /** Update overlay colormap. */
  updateOverlayColormap: (id: string, colormap: OverlayColormap) => void
  /** Update overlay colorbar limits. */
  updateOverlayColorbarLimits: (id: string, min: number, max: number) => void
  /** Download overlay as NIfTI file. */
  downloadOverlay: (id: string) => void
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

const BrainOverlayContext = createContext<BrainOverlayContextValue | null>(null)

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

export function BrainOverlayProvider({
  children,
}: {
  children: React.ReactNode
}): React.ReactElement {
  const [overlays, setOverlays] = useState<BrainOverlay[]>([])

  const addOverlay = useCallback((overlay: BrainOverlay) => {
    setOverlays((prev) => {
      // Do not add duplicates (same id already present).
      if (prev.some((o) => o.id === overlay.id)) return prev
      // If at capacity, drop the oldest.
      const next = prev.length >= MAX_OVERLAYS ? prev.slice(1) : prev
      return [...next, overlay]
    })
  }, [])

  const removeOverlay = useCallback((id: string) => {
    setOverlays((prev) => prev.filter((o) => o.id !== id))
  }, [])

  const clearOverlays = useCallback(() => {
    setOverlays([])
  }, [])

  const toggleOverlayVisibility = useCallback((id: string) => {
    setOverlays((prev) =>
      prev.map((o) => (o.id === id ? { ...o, visible: !o.visible } : o)),
    )
  }, [])

  const updateOverlayThreshold = useCallback((id: string, threshold: number) => {
    setOverlays((prev) =>
      prev.map((o) => (o.id === id ? { ...o, threshold } : o)),
    )
  }, [])

  const updateOverlayColormap = useCallback(
    (id: string, colormap: OverlayColormap) => {
      setOverlays((prev) =>
        prev.map((o) => (o.id === id ? { ...o, colormap } : o)),
      )
    },
    [],
  )

  const updateOverlayColorbarLimits = useCallback(
    (id: string, min: number, max: number) => {
      setOverlays((prev) =>
        prev.map((o) => (o.id === id ? { ...o, colorbarMin: min, colorbarMax: max } : o)),
      )
    },
    [],
  )

  const downloadOverlay = useCallback((id: string) => {
    const overlay = overlays.find((o) => o.id === id)
    if (!overlay) return

    // Convert base64 to blob and trigger download
    const linkSource = `data:application/octet-stream;base64,${overlay.base64}`
    const downloadLink = document.createElement('a')
    downloadLink.href = linkSource
    downloadLink.download = `${overlay.name.replace(/[^a-z0-9]/gi, '_')}.nii`
    downloadLink.click()
  }, [overlays])

  return (
    <BrainOverlayContext.Provider
      value={{
        overlays,
        addOverlay,
        removeOverlay,
        clearOverlays,
        toggleOverlayVisibility,
        updateOverlayThreshold,
        updateOverlayColormap,
        updateOverlayColorbarLimits,
        downloadOverlay,
      }}
    >
      {children}
    </BrainOverlayContext.Provider>
  )
}

export default BrainOverlayContext
