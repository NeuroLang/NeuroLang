/**
 * EngineSwitchHandler.test.tsx
 *
 * Tests for engine switching behaviour:
 *   1. Switching engine clears the query (visual builder + code editor)
 *   2. Switching engine clears brain overlays
 *   3. Switching engine shows "Switching engine…" transition state
 *   4. Switching to the same engine is a no-op (no clear)
 *   5. setSelectedEngine (permalink) does NOT trigger the banner
 */
import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, afterEach } from 'vitest'
import { EngineProvider } from '../../context/EngineContext'
import { QueryProvider } from '../../context/QueryContext'
import { ExecutionProvider } from '../../context/ExecutionContext'
import { BrainOverlayProvider } from '../../context/BrainOverlayContext'
import { useEngine } from '../../context/useEngine'
import { useQuery } from '../../context/useQuery'
import { useBrainOverlay } from '../../context/useBrainOverlay'
import EngineSwitchHandler from '../EngineSwitchHandler'

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/** Renders EngineSwitchHandler inside all required providers. */
function renderWithProviders(ui?: React.ReactNode) {
  return render(
    <EngineProvider>
      <QueryProvider>
        <ExecutionProvider>
          <BrainOverlayProvider>
            <EngineSwitchHandler />
            {ui}
          </BrainOverlayProvider>
        </ExecutionProvider>
      </QueryProvider>
    </EngineProvider>,
  )
}

/**
 * Test component that exposes context state for assertions and buttons to
 * mutate it.
 */
function StateHarness() {
  const { switchEngine } = useEngine()
  const { datalogText, setDatalogText, model, refresh } = useQuery()
  const { overlays, addOverlay } = useBrainOverlay()

  return (
    <div>
      <div data-testid="datalog-text">{datalogText}</div>
      <div data-testid="overlay-count">{overlays.length}</div>

      {/* Add a predicate to the visual builder */}
      <button
        data-testid="add-predicate"
        onClick={() => {
          model.addPredicate('PeakReported', ['x', 'y', 'z', 's'])
          refresh()
        }}
      >
        Add predicate
      </button>

      {/* Set raw datalog text */}
      <button
        data-testid="set-datalog"
        onClick={() => setDatalogText('ans(x) :- PeakReported(x, y, z, s)')}
      >
        Set datalog text
      </button>

      {/* Add a dummy overlay */}
      <button
        data-testid="add-overlay"
        onClick={() =>
          addOverlay({
            id: 'test-overlay',
            name: 'Test',
            base64: 'abc123',
            colormap: 'blue',
            isProbabilistic: false,
          })
        }
      >
        Add overlay
      </button>

      {/* Switch engine */}
      <button
        data-testid="switch-neurosynth"
        onClick={() => switchEngine('neurosynth')}
      >
        Switch to neurosynth
      </button>
      <button
        data-testid="switch-destrieux"
        onClick={() => switchEngine('destrieux')}
      >
        Switch to destrieux
      </button>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Query clearing tests
// ---------------------------------------------------------------------------

describe('EngineSwitchHandler – query clearing', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('clears the code editor text when the engine switches', async () => {
    renderWithProviders(<StateHarness />)

    // Set some datalog text.
    await userEvent.click(screen.getByTestId('set-datalog'))
    expect(screen.getByTestId('datalog-text')).toHaveTextContent(
      'ans(x) :- PeakReported(x, y, z, s)',
    )

    // Switch engine → handler clears the text.
    await userEvent.click(screen.getByTestId('switch-neurosynth'))

    expect(screen.getByTestId('datalog-text')).toHaveTextContent('')
  })

  it('clears the visual query builder predicates when the engine switches', async () => {
    renderWithProviders(<StateHarness />)

    // Add a predicate to the visual builder.
    await userEvent.click(screen.getByTestId('add-predicate'))
    // After refresh(), datalogText reflects the model.
    const textAfterAdd = screen.getByTestId('datalog-text').textContent ?? ''
    expect(textAfterAdd.trim().length).toBeGreaterThan(0)

    // Switch engine → handler resets the model and clears the text.
    await userEvent.click(screen.getByTestId('switch-neurosynth'))

    expect(screen.getByTestId('datalog-text')).toHaveTextContent('')
  })
})

// ---------------------------------------------------------------------------
// Overlay clearing tests
// ---------------------------------------------------------------------------

describe('EngineSwitchHandler – overlay clearing', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('removes all brain overlays when the engine switches', async () => {
    renderWithProviders(<StateHarness />)

    // Add an overlay.
    await userEvent.click(screen.getByTestId('add-overlay'))
    expect(screen.getByTestId('overlay-count')).toHaveTextContent('1')

    // Switch engine → handler clears overlays.
    await userEvent.click(screen.getByTestId('switch-neurosynth'))

    expect(screen.getByTestId('overlay-count')).toHaveTextContent('0')
  })
})

// ---------------------------------------------------------------------------
// No-op when switching to the same engine
// ---------------------------------------------------------------------------

describe('EngineSwitchHandler – same engine no-op', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('does not clear query when switching to the already-selected engine', async () => {
    renderWithProviders(<StateHarness />)

    // Select neurosynth first — this triggers the handler and clears things.
    await userEvent.click(screen.getByTestId('switch-neurosynth'))
    expect(screen.getByTestId('datalog-text')).toHaveTextContent('')

    // Now set some datalog text.
    await userEvent.click(screen.getByTestId('set-datalog'))
    expect(screen.getByTestId('datalog-text')).toHaveTextContent(
      'ans(x) :- PeakReported(x, y, z, s)',
    )

    // Switch to the SAME engine — should be a no-op.
    await userEvent.click(screen.getByTestId('switch-neurosynth'))

    // Text should remain unchanged.
    expect(screen.getByTestId('datalog-text')).toHaveTextContent(
      'ans(x) :- PeakReported(x, y, z, s)',
    )
  })
})

// ---------------------------------------------------------------------------
// Switching transition banner (isSwitching state in EngineContext)
// ---------------------------------------------------------------------------

describe('Engine switching transition banner', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('isSwitching is false by default', () => {
    function ReadIsSwitching() {
      const { isSwitching } = useEngine()
      return <div data-testid="is-switching">{String(isSwitching)}</div>
    }

    render(
      <EngineProvider>
        <ReadIsSwitching />
      </EngineProvider>,
    )

    expect(screen.getByTestId('is-switching')).toHaveTextContent('false')
  })

  it('isSwitching becomes true immediately after switchEngine and reverts after timeout', async () => {
    // Use real timers with a short enough duration that we can just wait.
    // We override SWITCHING_DURATION by relying on the real 800ms timeout.
    // Use real timers + just check that isSwitching changes synchronously.
    function SwitchAndRead() {
      const { isSwitching, switchEngine } = useEngine()
      return (
        <div>
          <div data-testid="is-switching">{String(isSwitching)}</div>
          <button
            data-testid="do-switch"
            onClick={() => switchEngine('neurosynth')}
          >
            Switch
          </button>
        </div>
      )
    }

    render(
      <EngineProvider>
        <SwitchAndRead />
      </EngineProvider>,
    )

    expect(screen.getByTestId('is-switching')).toHaveTextContent('false')

    await userEvent.click(screen.getByTestId('do-switch'))

    // isSwitching should be true right after the click.
    expect(screen.getByTestId('is-switching')).toHaveTextContent('true')

    // Wait for the real 800ms to elapse so isSwitching reverts to false.
    await waitFor(
      () => {
        expect(screen.getByTestId('is-switching')).toHaveTextContent('false')
      },
      { timeout: 2000 },
    )
  })

  it('setSelectedEngine (permalink/silent path) does NOT set isSwitching', async () => {
    function SilentSetAndRead() {
      const { isSwitching, setSelectedEngine } = useEngine()
      return (
        <div>
          <div data-testid="is-switching">{String(isSwitching)}</div>
          <button
            data-testid="do-set"
            onClick={() => setSelectedEngine('neurosynth')}
          >
            Set silently
          </button>
        </div>
      )
    }

    render(
      <EngineProvider>
        <SilentSetAndRead />
      </EngineProvider>,
    )

    await userEvent.click(screen.getByTestId('do-set'))

    // isSwitching must remain false — permalink loading should not flash banner.
    expect(screen.getByTestId('is-switching')).toHaveTextContent('false')
  })
})
