/**
 * OverlayManager.test.tsx
 *
 * Tests for the OverlayManager component and its sub-components:
 *   1. OverlayManager: renders overlays, hides when empty, remove buttons
 *   2. ColorBar: renders gradient and labels for probabilistic overlays
 *   3. BrainOverlayContext: addOverlay, removeOverlay, MAX_OVERLAYS cap
 */
import { render, screen, fireEvent, act } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
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

/** Build a test overlay. */
function makeOverlay(partial: Partial<BrainOverlay> = {}): BrainOverlay {
  return {
    id: partial.id ?? 'test:0:0',
    name: partial.name ?? 'Test Region',
    base64: partial.base64 ?? 'dGVzdA==',
    colormap: partial.colormap ?? 'hot',
    isProbabilistic: partial.isProbabilistic ?? false,
  }
}

/** Renders OverlayManager inside a BrainOverlayProvider. */
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
// OverlayManager – visibility
// ---------------------------------------------------------------------------

describe('OverlayManager – visibility', () => {
  it('does NOT render when no overlays are active', () => {
    renderOverlayManager()
    expect(screen.queryByTestId('overlay-manager')).not.toBeInTheDocument()
  })

  it('renders when at least one overlay is active', () => {
    const { getCtx } = renderOverlayManager()

    act(() => {
      getCtx().addOverlay(makeOverlay())
    })

    expect(screen.getByTestId('overlay-manager')).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// OverlayManager – overlay list
// ---------------------------------------------------------------------------

describe('OverlayManager – overlay list', () => {
  it('shows overlay name in the list', () => {
    const { getCtx } = renderOverlayManager()

    act(() => {
      getCtx().addOverlay(makeOverlay({ name: 'Left Motor Cortex' }))
    })

    expect(screen.getByText('Left Motor Cortex')).toBeInTheDocument()
  })

  it('shows count of active overlays in header', () => {
    const { getCtx } = renderOverlayManager()

    act(() => {
      getCtx().addOverlay(makeOverlay({ id: 'a', name: 'Region A' }))
      getCtx().addOverlay(makeOverlay({ id: 'b', name: 'Region B' }))
    })

    expect(screen.getByTestId('overlay-manager')).toHaveTextContent('2')
  })

  it('renders an overlay-item for each overlay', () => {
    const { getCtx } = renderOverlayManager()

    act(() => {
      getCtx().addOverlay(makeOverlay({ id: 'a', name: 'Region A' }))
      getCtx().addOverlay(makeOverlay({ id: 'b', name: 'Region B' }))
    })

    const items = screen.getAllByTestId('overlay-item')
    expect(items).toHaveLength(2)
  })
})

// ---------------------------------------------------------------------------
// OverlayManager – remove overlay
// ---------------------------------------------------------------------------

describe('OverlayManager – remove overlay', () => {
  it('renders a remove button for each overlay', () => {
    const { getCtx } = renderOverlayManager()

    act(() => {
      getCtx().addOverlay(makeOverlay({ id: 'r1', name: 'Region 1' }))
    })

    expect(screen.getByTestId('overlay-remove-btn')).toBeInTheDocument()
  })

  it('clicking remove button removes that overlay', () => {
    const { getCtx } = renderOverlayManager()

    act(() => {
      getCtx().addOverlay(makeOverlay({ id: 'r1', name: 'Region 1' }))
    })

    const removeBtn = screen.getByTestId('overlay-remove-btn')
    act(() => {
      fireEvent.click(removeBtn)
    })

    expect(screen.queryByTestId('overlay-manager')).not.toBeInTheDocument()
    expect(getCtx().overlays).toHaveLength(0)
  })

  it('clicking remove removes only the clicked overlay', () => {
    const { getCtx } = renderOverlayManager()

    act(() => {
      getCtx().addOverlay(makeOverlay({ id: 'r1', name: 'Region 1' }))
      getCtx().addOverlay(makeOverlay({ id: 'r2', name: 'Region 2' }))
    })

    const removeBtns = screen.getAllByTestId('overlay-remove-btn')
    // Remove first overlay
    act(() => {
      fireEvent.click(removeBtns[0])
    })

    expect(getCtx().overlays).toHaveLength(1)
    expect(getCtx().overlays[0].id).toBe('r2')
  })

  it('clicking "Clear all" removes all overlays', () => {
    const { getCtx } = renderOverlayManager()

    act(() => {
      getCtx().addOverlay(makeOverlay({ id: 'r1', name: 'Region 1' }))
      getCtx().addOverlay(makeOverlay({ id: 'r2', name: 'Region 2' }))
    })

    const clearBtn = screen.getByTestId('overlay-clear-all-btn')
    act(() => {
      fireEvent.click(clearBtn)
    })

    expect(getCtx().overlays).toHaveLength(0)
    expect(screen.queryByTestId('overlay-manager')).not.toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// ColorBar – probabilistic overlays
// ---------------------------------------------------------------------------

describe('ColorBar – probabilistic overlays', () => {
  it('renders color bar with data-testid', () => {
    render(<ColorBar colormap="hot" />)
    expect(screen.getByTestId('color-bar')).toBeInTheDocument()
  })

  it('renders the gradient element', () => {
    render(<ColorBar colormap="hot" />)
    expect(screen.getByTestId('color-bar-gradient')).toBeInTheDocument()
  })

  it('shows default low label "0"', () => {
    render(<ColorBar colormap="hot" />)
    const bar = screen.getByTestId('color-bar')
    expect(bar).toHaveTextContent('0')
  })

  it('shows default high label "1"', () => {
    render(<ColorBar colormap="hot" />)
    const bar = screen.getByTestId('color-bar')
    expect(bar).toHaveTextContent('1')
  })

  it('shows custom labels', () => {
    render(<ColorBar colormap="blue" lowLabel="0.0" highLabel="p-val" />)
    const bar = screen.getByTestId('color-bar')
    expect(bar).toHaveTextContent('0.0')
    expect(bar).toHaveTextContent('p-val')
  })

  it('renders ColorBar for probabilistic overlays in OverlayManager', () => {
    const { getCtx } = renderOverlayManager()

    act(() => {
      getCtx().addOverlay(
        makeOverlay({
          id: 'prob:0',
          name: 'Probability map',
          colormap: 'hot',
          isProbabilistic: true,
        }),
      )
    })

    expect(screen.getByTestId('color-bar')).toBeInTheDocument()
  })

  it('does NOT render ColorBar for non-probabilistic overlays', () => {
    const { getCtx } = renderOverlayManager()

    act(() => {
      getCtx().addOverlay(
        makeOverlay({
          id: 'plain:0',
          name: 'Plain region',
          colormap: 'blue',
          isProbabilistic: false,
        }),
      )
    })

    expect(screen.queryByTestId('color-bar')).not.toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// BrainOverlayContext – overlay management
// ---------------------------------------------------------------------------

describe('BrainOverlayContext – overlay management', () => {
  it('starts with no overlays', () => {
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

    expect(capturedCtx!.overlays).toHaveLength(0)
  })

  it('addOverlay adds an overlay', () => {
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
      capturedCtx!.addOverlay(makeOverlay({ id: 'x1' }))
    })

    expect(capturedCtx!.overlays).toHaveLength(1)
  })

  it('addOverlay does not add duplicate (same id)', () => {
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
      capturedCtx!.addOverlay(makeOverlay({ id: 'dup' }))
      capturedCtx!.addOverlay(makeOverlay({ id: 'dup' }))
    })

    expect(capturedCtx!.overlays).toHaveLength(1)
  })

  it(`removes oldest overlay when ${MAX_OVERLAYS} is exceeded`, () => {
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

    // Add MAX_OVERLAYS + 1 overlays
    act(() => {
      for (let i = 0; i <= MAX_OVERLAYS; i++) {
        capturedCtx!.addOverlay(makeOverlay({ id: `ov:${i}`, name: `ov${i}` }))
      }
    })

    expect(capturedCtx!.overlays).toHaveLength(MAX_OVERLAYS)
    // First overlay (ov:0) should have been evicted
    expect(capturedCtx!.overlays[0].id).toBe('ov:1')
  })

  it('removeOverlay removes the overlay with the given id', () => {
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
      capturedCtx!.addOverlay(makeOverlay({ id: 'keep' }))
      capturedCtx!.addOverlay(makeOverlay({ id: 'remove-me' }))
    })

    act(() => {
      capturedCtx!.removeOverlay('remove-me')
    })

    expect(capturedCtx!.overlays).toHaveLength(1)
    expect(capturedCtx!.overlays[0].id).toBe('keep')
  })

  it('clearOverlays removes all overlays', () => {
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
      capturedCtx!.addOverlay(makeOverlay({ id: 'a' }))
      capturedCtx!.addOverlay(makeOverlay({ id: 'b' }))
    })

    act(() => {
      capturedCtx!.clearOverlays()
    })

    expect(capturedCtx!.overlays).toHaveLength(0)
  })
})

// ---------------------------------------------------------------------------
// nextColormap utility
// ---------------------------------------------------------------------------

describe('nextColormap', () => {
  it('returns "hot" when no overlays exist', () => {
    expect(nextColormap([])).toBe('hot')
  })

  it('returns a colormap not already in use', () => {
    const existing: BrainOverlay[] = [makeOverlay({ colormap: 'hot' })]
    const next = nextColormap(existing)
    expect(next).not.toBe('hot')
  })

  it('cycles back to first color when all are used', () => {
    const allColors: BrainOverlay[] = [
      'hot', 'blue', 'green', 'red', 'yellow', 'cyan', 'pink', 'violet',
    ].map((colormap, i) => makeOverlay({ id: `o${i}`, colormap: colormap as BrainOverlay['colormap'] }))

    // All colors used – should cycle back to first
    const next = nextColormap(allColors)
    expect(next).toBe('hot')
  })
})
