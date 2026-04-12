/**
 * OverlayManager.controls.test.tsx
 * Tests for overlay control features: visibility toggle, threshold control,
 * colormap selection, colorbar limits, and NIfTI download.
 */
import { render, screen, fireEvent, act } from '@testing-library/react'
import { describe, it, expect, vitest } from 'vitest'
import OverlayManager, { ColorBar } from '../OverlayManager'
import {
  BrainOverlayProvider,
  MAX_OVERLAYS,
  type BrainOverlay,
} from '../../context/BrainOverlayContext'
import { useBrainOverlay } from '../../context/useBrainOverlay'
import { nextColormap } from '../../utils/overlayUtils'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeOverlay(partial: Partial<BrainOverlay> = {}): BrainOverlay {
  return {
    id: partial.id ?? 'test:0:0',
    name: partial.name ?? 'Test Region',
    base64: partial.base64 ?? 'dGVzdA==',
    colormap: partial.colormap ?? 'hot',
    isProbabilistic: partial.isProbabilistic ?? false,
    visible: partial.visible ?? true,
    threshold: partial.threshold ?? 0,
    colorbarMin: partial.colorbarMin ?? 0,
    colorbarMax: partial.colorbarMax ?? 1,
  }
}

function renderOverlayManager() {
  let capturedCtx: ReturnType<typeof useBrainOverlay> | null = null

  function CtxCapture() {
    capturedCtx = useBrainOverlay()
    return null
  }

  render(
    <BrainOverlayProvider>
      <CtxCapture />
      <OverlayManager />
    </BrainOverlayProvider>,
  )

  return { getCtx: () => capturedCtx! }
}

// ---------------------------------------------------------------------------
// BrainOverlayContext – visibility and controls
// ---------------------------------------------------------------------------

describe('BrainOverlayContext – visibility and controls', () => {
  it('starts with visible overlays and default threshold/colorbar values', () => {
    let capturedCtx: ReturnType<typeof useBrainOverlay> | null = null

    function CtxCapture() {
      capturedCtx = useBrainOverlay()
      return null
    }

    render(
      <BrainOverlayProvider>
        <CtxCapture />
      </BrainOverlayProvider>,
    )

    act(() => {
      capturedCtx!.addOverlay(makeOverlay({ id: 'test1' }))
    })

    expect(capturedCtx!.overlays).toHaveLength(1)
    expect(capturedCtx!.overlays[0].visible).toBe(true)
    expect(capturedCtx!.overlays[0].threshold).toBe(0)
    expect(capturedCtx!.overlays[0].colorbarMin).toBe(0)
    expect(capturedCtx!.overlays[0].colorbarMax).toBe(1)
  })

  it('toggleOverlayVisibility toggles overlay visibility', () => {
    let capturedCtx: ReturnType<typeof useBrainOverlay> | null = null

    function CtxCapture() {
      capturedCtx = useBrainOverlay()
      return null
    }

    render(
      <BrainOverlayProvider>
        <CtxCapture />
      </BrainOverlayProvider>,
    )

    act(() => {
      capturedCtx!.addOverlay(makeOverlay({ id: 'ov-hide' }))
    })

    expect(capturedCtx!.overlays[0].visible).toBe(true)
    act(() => {
      capturedCtx!.toggleOverlayVisibility('ov-hide')
    })
    expect(capturedCtx!.overlays[0].visible).toBe(false)
    act(() => {
      capturedCtx!.toggleOverlayVisibility('ov-hide')
    })
    expect(capturedCtx!.overlays[0].visible).toBe(true)
  })

  it('updateOverlayThreshold updates threshold for an overlay', () => {
    let capturedCtx: ReturnType<typeof useBrainOverlay> | null = null

    function CtxCapture() {
      capturedCtx = useBrainOverlay()
      return null
    }

    render(
      <BrainOverlayProvider>
        <CtxCapture />
      </BrainOverlayProvider>,
    )

    act(() => {
      capturedCtx!.addOverlay(makeOverlay({ id: 'ov-thresh', threshold: 0 }))
    })

    expect(capturedCtx!.overlays[0].threshold).toBe(0)
    act(() => {
      capturedCtx!.updateOverlayThreshold('ov-thresh', 0.5)
    })
    expect(capturedCtx!.overlays[0].threshold).toBe(0.5)
  })

  it('updateOverlayColormap updates colormap for an overlay', () => {
    let capturedCtx: ReturnType<typeof useBrainOverlay> | null = null

    function CtxCapture() {
      capturedCtx = useBrainOverlay()
      return null
    }

    render(
      <BrainOverlayProvider>
        <CtxCapture />
      </BrainOverlayProvider>,
    )

    act(() => {
      capturedCtx!.addOverlay(makeOverlay({ id: 'ov-colmap', colormap: 'hot' }))
    })

    expect(capturedCtx!.overlays[0].colormap).toBe('hot')
    act(() => {
      capturedCtx!.updateOverlayColormap('ov-colmap', 'blue')
    })
    expect(capturedCtx!.overlays[0].colormap).toBe('blue')
  })

  it('updateOverlayColorbarLimits updates colorbar min/max for an overlay', () => {
    let capturedCtx: ReturnType<typeof useBrainOverlay> | null = null

    function CtxCapture() {
      capturedCtx = useBrainOverlay()
      return null
    }

    render(
      <BrainOverlayProvider>
        <CtxCapture />
      </BrainOverlayProvider>,
    )

    act(() => {
      capturedCtx!.addOverlay(makeOverlay({ id: 'ov-cb', colorbarMin: 0.1, colorbarMax: 0.9 }))
    })

    expect(capturedCtx!.overlays[0].colorbarMin).toBe(0.1)
    expect(capturedCtx!.overlays[0].colorbarMax).toBe(0.9)
    act(() => {
      capturedCtx!.updateOverlayColorbarLimits('ov-cb', 0.2, 0.8)
    })
    expect(capturedCtx!.overlays[0].colorbarMin).toBe(0.2)
    expect(capturedCtx!.overlays[0].colorbarMax).toBe(0.8)
  })

  it('downloadOverlay triggers a file download', () => {
    let capturedCtx: ReturnType<typeof useBrainOverlay> | null = null
    const dlSpy = vitest.spyOn(document, 'createElement')

    function CtxCapture() {
      capturedCtx = useBrainOverlay()
      return null
    }

    render(
      <BrainOverlayProvider>
        <CtxCapture />
      </BrainOverlayProvider>,
    )

    act(() => {
      capturedCtx!.addOverlay(makeOverlay({ id: 'ov-dl', name: 'test-overlay' }))
    })

    act(() => {
      capturedCtx!.downloadOverlay('ov-dl')
    })

    expect(dlSpy).toHaveBeenCalled()
    dlSpy.mockRestore()
  })
})

// ---------------------------------------------------------------------------
// OverlayManager – visibility toggle
// ---------------------------------------------------------------------------

describe('OverlayManager – visibility toggle', () => {
  it('renders visibility toggle button for each overlay', () => {
    const { getCtx } = renderOverlayManager()

    act(() => {
      getCtx().addOverlay(makeOverlay({ id: 'vis-ov', name: 'Visible Overlay' }))
    })

    expect(screen.getByTestId('overlay-visibility-toggle')).toBeInTheDocument()
  })

  it('clicking visibility toggle hides the overlay view', () => {
    const { getCtx } = renderOverlayManager()

    act(() => {
      getCtx().addOverlay(makeOverlay({ id: 'vis-test', name: 'Test Overlay' }))
    })

    expect(getCtx().overlays[0].visible).toBe(true)
    act(() => {
      const toggle = screen.getByTestId('overlay-visibility-toggle')
      fireEvent.click(toggle)
    })
    expect(getCtx().overlays[0].visible).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// OverlayManager – threshold control
// ---------------------------------------------------------------------------

describe('OverlayManager – threshold control', () => {
  it('renders threshold slider with correct initial value when settings shown', () => {
    const { getCtx } = renderOverlayManager()

    act(() => {
      getCtx().addOverlay(
        makeOverlay({
          id: 'thresh-ov',
          colormap: 'hot',
          isProbabilistic: true,
          threshold: 0.3,
        }),
      )
    })

    act(() => {
      const settingsBtn = screen.getByTestId('overlay-settings-btn')
      fireEvent.click(settingsBtn)
    })

    const slider = screen.getByTestId('overlay-threshold-slider')
    expect(slider).toBeInTheDocument()
    expect(slider).toHaveValue('0.3')
  })

  it('updating threshold slider updates overlay threshold', () => {
    const { getCtx } = renderOverlayManager()

    act(() => {
      getCtx().addOverlay(
        makeOverlay({
          id: 'thresh-ov2',
          colormap: 'hot',
          isProbabilistic: true,
          threshold: 0,
        }),
      )
    })

    act(() => {
      const settingsBtn = screen.getByTestId('overlay-settings-btn')
      fireEvent.click(settingsBtn)
    })

    const slider = screen.getByTestId('overlay-threshold-slider')
    act(() => {
      fireEvent.change(slider, { target: { value: '0.5' } })
    })
    expect(getCtx().overlays[0].threshold).toBe(0.5)
  })
})

// ---------------------------------------------------------------------------
// OverlayManager – colormap selection
// ---------------------------------------------------------------------------

describe('OverlayManager – colormap selection', () => {
  it('renders colormap select when settings are shown', () => {
    const { getCtx } = renderOverlayManager()

    act(() => {
      getCtx().addOverlay(makeOverlay({ id: 'colmap-ov', colormap: 'red' }))
    })

    const settingsBtn = screen.getByTestId('overlay-settings-btn')
    act(() => {
      fireEvent.click(settingsBtn)
    })

    expect(screen.getByTestId('overlay-colormap-select')).toBeInTheDocument()
  })

  it('changing colormap select updates overlay colormap', () => {
    const { getCtx } = renderOverlayManager()

    act(() => {
      getCtx().addOverlay(makeOverlay({ id: 'colmap-ov2', colormap: 'green' }))
    })

    const settingsBtn = screen.getByTestId('overlay-settings-btn')
    act(() => {
      fireEvent.click(settingsBtn)
    })

    const select = screen.getByTestId('overlay-colormap-select')
    act(() => {
      fireEvent.change(select, { target: { value: 'violet' } })
    })
    expect(getCtx().overlays[0].colormap).toBe('violet')
  })
})

// ---------------------------------------------------------------------------
// OverlayManager – colorbar limits
// ---------------------------------------------------------------------------

describe('OverlayManager – colorbar limits', () => {
  it('renders colorbar min/max inputs when settings are shown', () => {
    const { getCtx } = renderOverlayManager()

    act(() => {
      getCtx().addOverlay(makeOverlay({ id: 'cb-ov', colorbarMin: 0.1, colorbarMax: 0.9 }))
    })

    const settingsBtn = screen.getByTestId('overlay-settings-btn')
    act(() => {
      fireEvent.click(settingsBtn)
    })

    expect(screen.getByTestId('overlay-colorbar-min-input')).toBeInTheDocument()
    expect(screen.getByTestId('overlay-colorbar-max-input')).toBeInTheDocument()
  })

  it('updating colorbar min input updates overlay colorbarMin', () => {
    const { getCtx } = renderOverlayManager()

    act(() => {
      getCtx().addOverlay(makeOverlay({ id: 'cb-ov2', colorbarMin: 0, colorbarMax: 1 }))
    })

    const settingsBtn = screen.getByTestId('overlay-settings-btn')
    act(() => {
      fireEvent.click(settingsBtn)
    })

    const minInput = screen.getByTestId('overlay-colorbar-min-input')
    act(() => {
      fireEvent.change(minInput, { target: { value: '0.25' } })
    })
    expect(getCtx().overlays[0].colorbarMin).toBe(0.25)
  })

  it('updating colorbar max input updates overlay colorbarMax', () => {
    const { getCtx } = renderOverlayManager()

    act(() => {
      getCtx().addOverlay(makeOverlay({ id: 'cb-ov3', colorbarMin: 0, colorbarMax: 1 }))
    })

    const settingsBtn = screen.getByTestId('overlay-settings-btn')
    act(() => {
      fireEvent.click(settingsBtn)
    })

    const maxInput = screen.getByTestId('overlay-colorbar-max-input')
    act(() => {
      fireEvent.change(maxInput, { target: { value: '0.8' } })
    })
    expect(getCtx().overlays[0].colorbarMax).toBe(0.8)
  })
})

// ---------------------------------------------------------------------------
// OverlayManager – download overlay
// ---------------------------------------------------------------------------

describe('OverlayManager – download overlay', () => {
  it('renders download button for each overlay', () => {
    const { getCtx } = renderOverlayManager()

    act(() => {
      getCtx().addOverlay(makeOverlay({ id: 'dl-ov', name: 'Downloadable Overlay' }))
    })

    expect(screen.getByTestId('overlay-download-btn')).toBeInTheDocument()
  })

  it('clicking download button triggers download', () => {
    const { getCtx } = renderOverlayManager()
    const dlSpy = vitest.spyOn(document, 'createElement')

    act(() => {
      getCtx().addOverlay(makeOverlay({ id: 'dl-ov2', name: 'TestDownload', base64: 'dGVzdA==' }))
    })

    const downloadBtn = screen.getByTestId('overlay-download-btn')
    act(() => {
      fireEvent.click(downloadBtn)
    })

    expect(dlSpy).toHaveBeenCalled()
    dlSpy.mockRestore()
  })
})
