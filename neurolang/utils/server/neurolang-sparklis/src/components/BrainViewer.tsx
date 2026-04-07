/**
 * BrainViewer.tsx
 *
 * Niivue-based brain atlas viewer that shows three orthogonal slice views
 * (axial, sagittal, coronal) and displays the current MNI coordinates as the
 * user moves the crosshair.
 *
 * On engine selection the component fetches GET /v2/atlas/:engine, decodes
 * the base64 NIfTI, and loads it as the background volume. When no VBR
 * results exist the atlas is shown without overlays and without errors.
 *
 * Design notes:
 * - Niivue is imported lazily (dynamic import) to avoid breaking jsdom tests
 *   that do not support WebGL.
 * - Atlas loading and Niivue initialization are coordinated via a shared ref
 *   (niivueRef). If Niivue cannot initialize (e.g., no WebGL), the atlas
 *   data is still fetched and the loading state is cleared gracefully.
 * - The coordinate display state is managed by React, updated via the
 *   Niivue onLocationChange callback.
 */
import React, { useEffect, useRef, useState, useCallback } from 'react'
import { useEngine } from '../context/useEngine'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** MNI coordinate triple. */
interface MNICoords {
  x: number
  y: number
  z: number
}

/** Shape of the location-change event from Niivue. */
export interface NiivueLocationData {
  mm: [number, number, number]
  vox?: [number, number, number]
  frac?: [number, number, number]
  string?: string
}

// ---------------------------------------------------------------------------
// BrainViewer component
// ---------------------------------------------------------------------------

export interface BrainViewerProps {
  /** Optional test ID for easier querying in tests. */
  'data-testid'?: string
  /**
   * Callback exposed for testing: called when the internal location-change
   * handler is registered, allowing tests to trigger coordinate updates.
   * @internal
   */
  onLocationHandlerReady?: (
    handler: (data: NiivueLocationData) => void,
  ) => void
}

function BrainViewer(props: BrainViewerProps): React.ReactElement {
  const { selectedEngine } = useEngine()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  // Niivue instance stored here after successful initialization.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const niivueRef = useRef<any>(null)
  const [coords, setCoords] = useState<MNICoords>({ x: 0, y: 0, z: 0 })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Keep onLocationHandlerReady ref stable so it doesn't trigger re-runs.
  const onLocationHandlerReadyRef = useRef(props.onLocationHandlerReady)
  useEffect(() => {
    onLocationHandlerReadyRef.current = props.onLocationHandlerReady
  })

  /** Round to one decimal place for display. */
  const fmt = (n: number): string => n.toFixed(1)

  // Handle location change from Niivue crosshair movement.
  const handleLocationChange = useCallback((data: NiivueLocationData) => {
    if (data?.mm && data.mm.length >= 3) {
      setCoords({ x: data.mm[0], y: data.mm[1], z: data.mm[2] })
    }
  }, [])

  // Initialize Niivue once the canvas is mounted. We import Niivue lazily to
  // avoid breaking jsdom environments that don't support WebGL.
  useEffect(() => {
    if (!canvasRef.current) return
    if (niivueRef.current) return // already initialized

    let cancelled = false

    async function initNiivue(): Promise<void> {
      try {
        // Lazy import avoids loading WebGL code in test environments.
        const { Niivue } = await import('@niivue/niivue')
        if (cancelled || !canvasRef.current) return

        const nv = new Niivue({
          // Start in multiplanar mode to show all three orthogonal slices.
          sliceType: 3, // MULTIPLANAR
          backColor: [0.1, 0.1, 0.1, 1],
          crosshairColor: [1, 0, 0, 1],
          show3Dcrosshair: false,
          isOrientCube: false,
        })

        // Register location-change handler BEFORE attaching to canvas so
        // events are not missed during initialization.
        // Cast to any because Niivue's onLocationChange signature is not
        // strongly typed and accepts an unknown data object.
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        nv.onLocationChange = (data: any) =>
          handleLocationChange(data as NiivueLocationData)

        // Notify any test hook that the handler is ready.
        onLocationHandlerReadyRef.current?.(handleLocationChange)

        await nv.attachToCanvas(canvasRef.current)
        if (!cancelled) {
          niivueRef.current = nv
        }
      } catch (err) {
        if (!cancelled) {
          // Niivue init failure (e.g., no WebGL) is non-fatal: the component
          // still shows the coordinate display and can load atlas data if
          // Niivue becomes available later.
          console.error('[BrainViewer] Failed to initialize Niivue:', err)
        }
      }
    }

    initNiivue()

    return () => {
      cancelled = true
    }
    // handleLocationChange is stable (useCallback with no deps).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Wire up the location-change handler whenever it changes.
  useEffect(() => {
    if (!niivueRef.current) return
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    niivueRef.current.onLocationChange = (data: any) =>
      handleLocationChange(data as NiivueLocationData)
  }, [handleLocationChange])

  // Fetch and load the atlas when the engine changes.
  useEffect(() => {
    if (!selectedEngine) return

    const controller = new AbortController()
    setLoading(true)
    setError(null)

    fetch(`/v2/atlas/${selectedEngine}`, { signal: controller.signal })
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(`Atlas request failed: ${res.status}`)
        }
        return res.json() as Promise<{
          status: string
          data?: { image: string }
          image?: string
        }>
      })
      .then(async (json) => {
        // Support both envelope format {status, data: {image}} and flat
        // {image} shapes for robustness.
        const base64 =
          json?.data?.image ?? (json as unknown as { image: string }).image
        if (!base64) {
          throw new Error('Atlas response missing image data')
        }

        // Lazily import NVImage. This is safe even when Niivue couldn't
        // initialize fully (e.g., no WebGL) because NVImage is a pure-JS
        // class that does not require a GL context.
        try {
          const { NVImage } = await import('@niivue/niivue')
          // Use a .nii extension in the name so Niivue correctly identifies
          // the format as NIfTI rather than DICOM.
          const volume = NVImage.loadFromBase64({
            base64,
            name: `${selectedEngine}.nii`,
          })

          // Only add the volume if Niivue is initialized (has a GL context).
          if (niivueRef.current) {
            await niivueRef.current.loadVolumes([])
            niivueRef.current.addVolume(volume)
          }
        } catch (nvErr) {
          // NVImage decode failure should not surface as a user-visible error
          // as long as the canvas can still be shown.
          console.warn('[BrainViewer] NVImage.loadFromBase64 failed:', nvErr)
        }
      })
      .catch((err: unknown) => {
        if ((err as { name?: string }).name !== 'AbortError') {
          console.error('[BrainViewer] Failed to load atlas:', err)
          setError('Failed to load atlas image')
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLoading(false)
        }
      })

    return () => controller.abort()
  }, [selectedEngine])

  const testId = props['data-testid'] ?? 'brain-viewer'

  return (
    <div className="brain-viewer" data-testid={testId}>
      {/* Panel header */}
      <div className="brain-viewer-header">
        <span className="brain-viewer-title">Brain Viewer</span>
        {loading && (
          <span
            className="brain-viewer-loading"
            data-testid="brain-viewer-loading"
          >
            Loading atlas…
          </span>
        )}
        {error && (
          <span
            className="brain-viewer-error"
            data-testid="brain-viewer-error"
            role="alert"
          >
            {error}
          </span>
        )}
      </div>

      {/* Niivue canvas */}
      <div className="brain-viewer-canvas-container">
        <canvas
          ref={canvasRef}
          className="brain-viewer-canvas"
          data-testid="brain-viewer-canvas"
          aria-label="Brain atlas viewer"
        />
      </div>

      {/* MNI coordinate display */}
      <div
        className="brain-viewer-coords"
        data-testid="brain-viewer-coords"
        aria-label="MNI coordinates"
      >
        <span className="brain-viewer-coords-label">MNI:</span>
        <span className="brain-viewer-coord" data-testid="brain-viewer-coord-x">
          x&nbsp;=&nbsp;{fmt(coords.x)}
        </span>
        <span className="brain-viewer-coord" data-testid="brain-viewer-coord-y">
          y&nbsp;=&nbsp;{fmt(coords.y)}
        </span>
        <span className="brain-viewer-coord" data-testid="brain-viewer-coord-z">
          z&nbsp;=&nbsp;{fmt(coords.z)}
        </span>
      </div>
    </div>
  )
}

export default BrainViewer
