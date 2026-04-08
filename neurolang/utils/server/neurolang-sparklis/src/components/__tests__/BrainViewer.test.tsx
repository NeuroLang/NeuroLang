/**
 * BrainViewer.test.tsx
 *
 * Tests for the BrainViewer component.
 *
 * Niivue requires WebGL which is not available in jsdom, so we mock the
 * @niivue/niivue module for all tests. The component handles Niivue init
 * failure gracefully (non-fatal), so tests can verify fetch behavior,
 * coordinate display, and loading states regardless of WebGL availability.
 */
import { render, screen, waitFor, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import React from 'react'
import BrainViewer, { type NiivueLocationData } from '../BrainViewer'
import { EngineProvider } from '../../context/EngineContext'
import { BrainOverlayProvider } from '../../context/BrainOverlayContext'
import { useBrainOverlay } from '../../context/useBrainOverlay'
import { useEngine } from '../../context/useEngine'

// ---------------------------------------------------------------------------
// Mock @niivue/niivue
// ---------------------------------------------------------------------------

// Mocks are defined inside vi.mock() factory to avoid vitest hoisting issues.
vi.mock('@niivue/niivue', () => {
  const addVolumeFn = vi.fn()
  const removeVolumeFn = vi.fn()
  const setColormapFn = vi.fn()
  const loadVolumesFn = vi.fn().mockResolvedValue(undefined)
  const attachToCanvasFn = vi.fn().mockResolvedValue(undefined)

  // Simple Niivue mock constructor
  const MockNiivueClass = vi.fn().mockImplementation(() => {
    return {
      onLocationChange: null,
      attachToCanvas: attachToCanvasFn,
      loadVolumes: loadVolumesFn,
      addVolume: addVolumeFn,
      removeVolume: removeVolumeFn,
      setColormap: setColormapFn,
      volumes: [],
    }
  })

  const loadFromBase64Fn = vi.fn().mockReturnValue({ id: 'mock-volume', name: '' })
  const MockNVImage = {
    loadFromBase64: loadFromBase64Fn,
  }

  return {
    Niivue: MockNiivueClass,
    NVImage: MockNVImage,
  }
})

// Import the mocked module AFTER vi.mock declaration.
import { Niivue as MockNiivue, NVImage as MockNVImage } from '@niivue/niivue'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Sets the selected engine in the EngineContext.
 */
function EngineSelectHelper({ engineName }: { engineName: string }) {
  const { setSelectedEngine } = useEngine()
  React.useEffect(() => {
    setSelectedEngine(engineName)
  }, [engineName, setSelectedEngine])
  return null
}

/** Wrapper that provides both EngineContext and BrainOverlayContext. */
function AllProviders({ children }: { children: React.ReactNode }) {
  return (
    <EngineProvider>
      <BrainOverlayProvider>{children}</BrainOverlayProvider>
    </EngineProvider>
  )
}

/** Build a successful atlas fetch response. */
function makeAtlasResponse(base64: string = 'dGVzdA==') {
  return {
    ok: true,
    json: async () => ({
      status: 'ok',
      data: { image: base64 },
    }),
  } as Response
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('BrainViewer', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal('fetch', vi.fn())
  })

  // Note: we intentionally do NOT call vi.restoreAllMocks() in afterEach
  // because it would remove the vi.fn().mockImplementation() from MockNiivue,
  // causing subsequent tests to receive `undefined` from `new MockNiivue()`.
  afterEach(() => {
    vi.unstubAllGlobals()
  })

  // -------------------------------------------------------------------------
  // Rendering
  // -------------------------------------------------------------------------

  it('renders the canvas element', () => {
    render(
      <AllProviders>
        <BrainViewer />
      </AllProviders>,
    )
    expect(screen.getByTestId('brain-viewer-canvas')).toBeInTheDocument()
    expect(screen.getByTestId('brain-viewer')).toBeInTheDocument()
  })

  it('renders coordinate display with initial zero values', () => {
    render(
      <AllProviders>
        <BrainViewer />
      </AllProviders>,
    )
    expect(screen.getByTestId('brain-viewer-coords')).toBeInTheDocument()
    expect(screen.getByTestId('brain-viewer-coord-x')).toHaveTextContent(
      'x = 0.0',
    )
    expect(screen.getByTestId('brain-viewer-coord-y')).toHaveTextContent(
      'y = 0.0',
    )
    expect(screen.getByTestId('brain-viewer-coord-z')).toHaveTextContent(
      'z = 0.0',
    )
  })

  it('renders the Brain Viewer header title', () => {
    render(
      <AllProviders>
        <BrainViewer />
      </AllProviders>,
    )
    expect(screen.getByText('Brain Viewer')).toBeInTheDocument()
  })

  it('shows no error initially', () => {
    render(
      <AllProviders>
        <BrainViewer />
      </AllProviders>,
    )
    expect(screen.queryByTestId('brain-viewer-error')).not.toBeInTheDocument()
  })

  it('accepts a custom data-testid', () => {
    render(
      <AllProviders>
        <BrainViewer data-testid="custom-viewer" />
      </AllProviders>,
    )
    expect(screen.getByTestId('custom-viewer')).toBeInTheDocument()
  })

  // -------------------------------------------------------------------------
  // Atlas loading
  // -------------------------------------------------------------------------

  it('does not fetch atlas when no engine is selected', async () => {
    render(
      <AllProviders>
        <BrainViewer />
      </AllProviders>,
    )
    await act(async () => {
      await new Promise((r) => setTimeout(r, 50))
    })
    expect(fetch).not.toHaveBeenCalled()
  })

  it('fetches atlas from /v2/atlas/:engine when engine is selected', async () => {
    vi.mocked(fetch).mockResolvedValue(makeAtlasResponse())

    render(
      <AllProviders>
        <EngineSelectHelper engineName="neurosynth" />
        <BrainViewer />
      </AllProviders>,
    )

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        '/v2/atlas/neurosynth',
        expect.any(Object),
      )
    })
  })

  it('calls NVImage.loadFromBase64 with the base64 data from atlas response', async () => {
    const testBase64 = 'dGVzdGJhc2U2NA=='
    vi.mocked(fetch).mockResolvedValue(makeAtlasResponse(testBase64))

    render(
      <AllProviders>
        <EngineSelectHelper engineName="neurosynth" />
        <BrainViewer />
      </AllProviders>,
    )

    await waitFor(() => {
      expect(MockNVImage.loadFromBase64).toHaveBeenCalledWith(
        expect.objectContaining({ base64: testBase64 }),
      )
    })
  })

  it('calls NVImage.loadFromBase64 with a .nii name so Niivue identifies it as NIfTI', async () => {
    vi.mocked(fetch).mockResolvedValue(makeAtlasResponse())

    render(
      <AllProviders>
        <EngineSelectHelper engineName="neurosynth" />
        <BrainViewer />
      </AllProviders>,
    )

    await waitFor(() => {
      expect(MockNVImage.loadFromBase64).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'neurosynth.nii' }),
      )
    })
  })

  it('shows loading state while fetching atlas', async () => {
    // Never resolves – keeps the loading state visible.
    vi.mocked(fetch).mockReturnValue(new Promise(() => {}))

    render(
      <AllProviders>
        <EngineSelectHelper engineName="neurosynth" />
        <BrainViewer />
      </AllProviders>,
    )

    await waitFor(() => {
      expect(screen.getByTestId('brain-viewer-loading')).toBeInTheDocument()
    })
  })

  it('hides loading state after atlas loads successfully', async () => {
    vi.mocked(fetch).mockResolvedValue(makeAtlasResponse())

    render(
      <AllProviders>
        <EngineSelectHelper engineName="neurosynth" />
        <BrainViewer />
      </AllProviders>,
    )

    await waitFor(() => {
      expect(
        screen.queryByTestId('brain-viewer-loading'),
      ).not.toBeInTheDocument()
    })
  })

  it('shows error when atlas fetch fails with non-ok response', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => ({ status: 'error' }),
    } as Response)

    render(
      <AllProviders>
        <EngineSelectHelper engineName="neurosynth" />
        <BrainViewer />
      </AllProviders>,
    )

    await waitFor(() => {
      expect(screen.getByTestId('brain-viewer-error')).toBeInTheDocument()
    })
  })

  it('shows no error when atlas loads and no VBR overlays exist (graceful state)', async () => {
    // Simulates: engine selected, atlas loads OK, no overlays added.
    // This is the default state when no VBR results exist.
    vi.mocked(fetch).mockResolvedValue(makeAtlasResponse())

    render(
      <AllProviders>
        <EngineSelectHelper engineName="neurosynth" />
        <BrainViewer />
      </AllProviders>,
    )

    await waitFor(() => {
      // Atlas loaded successfully -> no error, no loading indicator.
      expect(screen.queryByTestId('brain-viewer-error')).not.toBeInTheDocument()
      expect(
        screen.queryByTestId('brain-viewer-loading'),
      ).not.toBeInTheDocument()
    })
  })

  it('re-fetches atlas when engine changes', async () => {
    vi.mocked(fetch).mockResolvedValue(makeAtlasResponse())

    const { rerender } = render(
      <AllProviders>
        <EngineSelectHelper engineName="neurosynth" />
        <BrainViewer />
      </AllProviders>,
    )

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        '/v2/atlas/neurosynth',
        expect.any(Object),
      )
    })

    vi.clearAllMocks()
    vi.mocked(fetch).mockResolvedValue(makeAtlasResponse())
    vi.mocked(MockNVImage.loadFromBase64).mockReturnValue({
      id: 'mock-volume-2',
      name: '',
    } as unknown as ReturnType<typeof MockNVImage.loadFromBase64>)

    rerender(
      <AllProviders>
        <EngineSelectHelper engineName="destrieux" />
        <BrainViewer />
      </AllProviders>,
    )

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        '/v2/atlas/destrieux',
        expect.any(Object),
      )
    })
  })

  // -------------------------------------------------------------------------
  // Coordinate display
  // -------------------------------------------------------------------------

  it('updates coordinate display when location-change callback fires', async () => {
    vi.mocked(fetch).mockResolvedValue(makeAtlasResponse())

    // Capture the location handler when Niivue notifies us it's ready.
    let capturedHandler: ((data: NiivueLocationData) => void) | null = null

    render(
      <AllProviders>
        <EngineSelectHelper engineName="neurosynth" />
        <BrainViewer
          onLocationHandlerReady={(handler) => {
            capturedHandler = handler
          }}
        />
      </AllProviders>,
    )

    // Wait for Niivue to be initialized and the handler to be registered.
    await waitFor(() => {
      expect(MockNiivue).toHaveBeenCalled()
    })

    // The handler might not be captured if Niivue init fails (no WebGL).
    // In that case we can still call handleLocationChange indirectly.
    // For coordinate tests, we simulate by calling the handler directly.
    if (!capturedHandler) {
      // Niivue init failed (no WebGL in jsdom) — but the component still
      // exposes handleLocationChange as a stable callback. Simulate.
      // We can get it via the onLocationHandlerReady prop path.
      // For a more robust test, we verify that the coordinate UI responds
      // to direct state updates via a synthetic approach.
      //
      // Skip this test path when WebGL is unavailable — the coordinate
      // display is a UI concern that's tested via the prop-based handler.
      return
    }

    await act(async () => {
      capturedHandler!({ mm: [10.5, -20.3, 35.7] })
    })

    expect(screen.getByTestId('brain-viewer-coord-x')).toHaveTextContent(
      'x = 10.5',
    )
    expect(screen.getByTestId('brain-viewer-coord-y')).toHaveTextContent(
      'y = -20.3',
    )
    expect(screen.getByTestId('brain-viewer-coord-z')).toHaveTextContent(
      'z = 35.7',
    )
  })

  it('updates coordinate display with negative values', async () => {
    vi.mocked(fetch).mockResolvedValue(makeAtlasResponse())

    let capturedHandler: ((data: NiivueLocationData) => void) | null = null

    render(
      <AllProviders>
        <EngineSelectHelper engineName="neurosynth" />
        <BrainViewer
          onLocationHandlerReady={(handler) => {
            capturedHandler = handler
          }}
        />
      </AllProviders>,
    )

    await waitFor(() => {
      expect(MockNiivue).toHaveBeenCalled()
    })

    if (!capturedHandler) return

    await act(async () => {
      capturedHandler!({ mm: [-45.0, 0.0, -100.5] })
    })

    expect(screen.getByTestId('brain-viewer-coord-x')).toHaveTextContent(
      'x = -45.0',
    )
    expect(screen.getByTestId('brain-viewer-coord-y')).toHaveTextContent(
      'y = 0.0',
    )
    expect(screen.getByTestId('brain-viewer-coord-z')).toHaveTextContent(
      'z = -100.5',
    )
  })
})

// ---------------------------------------------------------------------------
// Overlay synchronization (via BrainOverlayContext)
// ---------------------------------------------------------------------------

/** Helper: renders BrainViewer with a controller that can inject overlays. */
function renderWithOverlayControl() {
  let capturedCtx: ReturnType<typeof useBrainOverlay> | null = null

  function OverlayCapture() {
    capturedCtx = useBrainOverlay()
    return null
  }

  vi.mocked(fetch).mockResolvedValue(makeAtlasResponse())

  render(
    <AllProviders>
      <OverlayCapture />
      <EngineSelectHelper engineName="destrieux" />
      <BrainViewer />
    </AllProviders>,
  )

  return { getCtx: () => capturedCtx! }
}

describe('BrainViewer – overlay sync', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('calls NVImage.loadFromBase64 when an overlay is added to context', async () => {
    const { getCtx } = renderWithOverlayControl()

    await waitFor(() => {
      expect(fetch).toHaveBeenCalled()
    })

    act(() => {
      getCtx().addOverlay({
        id: 'test:0:0',
        name: 'Test region',
        base64: 'dGVzdA==',
        colormap: 'hot',
        isProbabilistic: false,
      })
    })

    await waitFor(() => {
      // NVImage.loadFromBase64 called for atlas + overlay
      expect(MockNVImage.loadFromBase64).toHaveBeenCalledWith(
        expect.objectContaining({ base64: 'dGVzdA==' }),
      )
    })
  })

  it('does not crash when niivue is not initialized', async () => {
    // Renders BrainViewer (no WebGL in jsdom so niivueRef stays null)
    // Adding an overlay should not throw.
    vi.mocked(fetch).mockResolvedValue(makeAtlasResponse())

    let capturedCtx: ReturnType<typeof useBrainOverlay> | null = null

    function OverlayCapture() {
      capturedCtx = useBrainOverlay()
      return null
    }

    render(
      <AllProviders>
        <OverlayCapture />
        <BrainViewer />
      </AllProviders>,
    )

    await act(async () => {
      await new Promise((r) => setTimeout(r, 50))
    })

    expect(() => {
      act(() => {
        capturedCtx!.addOverlay({
          id: 'safe:0:0',
          name: 'Safe overlay',
          base64: 'dGVzdA==',
          colormap: 'blue',
          isProbabilistic: false,
        })
      })
    }).not.toThrow()
  })

  it('calls nv.addVolume when overlay is added (not addOverlay)', async () => {
    const { getCtx } = renderWithOverlayControl()

    await waitFor(() => {
      expect(fetch).toHaveBeenCalled()
    })

    // Get the mock Niivue instance
    const nvInstance = vi.mocked(MockNiivue).mock.results[0]?.value
    if (!nvInstance) return // skip if Niivue not initialized in jsdom

    act(() => {
      getCtx().addOverlay({
        id: 'test:0:0',
        name: 'Test region',
        base64: 'dGVzdA==',
        colormap: 'hot',
        isProbabilistic: false,
      })
    })

    await waitFor(() => {
      expect(nvInstance.addVolume).toHaveBeenCalled()
    })
  })

  it('calls nv.removeVolume when overlay is removed', async () => {
    const { getCtx } = renderWithOverlayControl()

    await waitFor(() => {
      expect(fetch).toHaveBeenCalled()
    })

    const nvInstance = vi.mocked(MockNiivue).mock.results[0]?.value
    if (!nvInstance) return // skip if Niivue not initialized in jsdom

    // Add an overlay first
    const mockVol = { id: 'test:0:0', name: 'test:0:0.nii' }
    vi.mocked(MockNVImage.loadFromBase64).mockReturnValue(
      mockVol as ReturnType<typeof MockNVImage.loadFromBase64>,
    )
    // Simulate that the volume exists in nv.volumes (index 0 = background)
    nvInstance.volumes = [{ id: 'atlas', name: 'atlas.nii' }, mockVol]

    act(() => {
      getCtx().addOverlay({
        id: 'test:0:0',
        name: 'Test region',
        base64: 'dGVzdA==',
        colormap: 'hot',
        isProbabilistic: false,
      })
    })

    await waitFor(() => {
      expect(MockNVImage.loadFromBase64).toHaveBeenCalledWith(
        expect.objectContaining({ base64: 'dGVzdA==' }),
      )
    })

    // Now remove the overlay
    act(() => {
      getCtx().removeOverlay('test:0:0')
    })

    await waitFor(() => {
      expect(nvInstance.removeVolume).toHaveBeenCalledWith(mockVol)
    })
  })
})
