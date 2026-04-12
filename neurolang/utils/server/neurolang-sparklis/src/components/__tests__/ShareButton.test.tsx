/**
 * ShareButton.test.tsx
 *
 * Tests for the ShareButton component and permalink URL logic.
 *
 * Covers:
 *   1. ShareButton renders when engine and query are set
 *   2. Clicking ShareButton generates a permalink URL with base64-encoded query
 *   3. The generated URL follows the format #/<engine>?q=<base64query>
 *   4. Clicking ShareButton calls writeToClipboard with the URL
 *   5. Shows 'Copied!' tooltip/feedback after clicking
 *   6. The 'Copied!' feedback disappears after a short delay
 *   7-15. permalink utilities: parse, build, error handling
 */
import React from 'react'
import { render, screen, waitFor, act } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, afterEach } from 'vitest'
import ShareButton from '../ShareButton'
import { EngineProvider } from '../../context/EngineContext'
import { QueryProvider } from '../../context/QueryContext'
import { useEngine } from '../../context/useEngine'
import { useQuery } from '../../context/useQuery'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * A component that sets the engine + query directly in the context on mount,
 * making the ShareButton immediately enabled.
 */
function ContextInitializer({
  engine,
  query,
}: {
  engine: string | null
  query: string
}): null {
  const { setSelectedEngine } = useEngine()
  const { setDatalogText } = useQuery()

  // Use layout effect so state is set synchronously before children render
  React.useLayoutEffect(() => {
    if (engine !== null) setSelectedEngine(engine)
    if (query) setDatalogText(query)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return null
}

interface RenderOptions {
  engine?: string | null
  query?: string
  writeToClipboard?: (text: string) => Promise<void>
  getHref?: () => string
}

function renderShareButton({
  engine = null,
  query = '',
  writeToClipboard,
  getHref,
}: RenderOptions = {}) {
  return render(
    <EngineProvider>
      <QueryProvider>
        <ContextInitializer engine={engine} query={query} />
        <ShareButton writeToClipboard={writeToClipboard} getHref={getHref} />
      </QueryProvider>
    </EngineProvider>,
  )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('ShareButton', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  // -------------------------------------------------------------------------
  // 1. Renders when engine and query are set
  // -------------------------------------------------------------------------

  it('renders Share button', async () => {
    renderShareButton({ engine: 'neurosynth', query: 'ans(x) :- PeakReported(x, y, z, s)' })

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /share/i })).toBeInTheDocument()
    })
  })

  // -------------------------------------------------------------------------
  // 2. Generates permalink URL with base64-encoded query
  // -------------------------------------------------------------------------

  it('generates a permalink URL with base64-encoded query on click', async () => {
    const user = userEvent.setup()
    const query = 'ans(x) :- PeakReported(x, y, z, s)'
    const writeToClipboard = vi.fn().mockResolvedValue(undefined)
    const getHref = () => 'http://localhost:3100/'

    renderShareButton({ engine: 'neurosynth', query, writeToClipboard, getHref })

    // Wait for button to be enabled
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /share/i })).not.toBeDisabled()
    })
    await user.click(screen.getByRole('button', { name: /share/i }))

    expect(writeToClipboard).toHaveBeenCalledOnce()
    const calledWith: string = writeToClipboard.mock.calls[0][0] as string

    // URL should contain the hash
    expect(calledWith).toContain('#/neurosynth')

    // The q param should be base64-encoded query
    const hashPart = calledWith.split('#')[1]
    const qParam = new URLSearchParams(hashPart.split('?')[1]).get('q')
    expect(qParam).not.toBeNull()
    const decoded = atob(qParam!)
    expect(decoded).toBe(query)
  })

  // -------------------------------------------------------------------------
  // 3. URL format is #/<engine>?q=<base64query>
  // -------------------------------------------------------------------------

  it('uses the correct URL hash format', async () => {
    const user = userEvent.setup()
    const query = 'ans(x) :- Study(x, y)'
    const writeToClipboard = vi.fn().mockResolvedValue(undefined)
    const getHref = () => 'http://localhost:3100/'

    renderShareButton({ engine: 'destrieux', query, writeToClipboard, getHref })

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /share/i })).not.toBeDisabled()
    })
    await user.click(screen.getByRole('button', { name: /share/i }))

    const calledWith: string = writeToClipboard.mock.calls[0][0] as string
    expect(calledWith).toMatch(/#\/destrieux\?q=/)
  })

  // -------------------------------------------------------------------------
  // 4. Calls writeToClipboard with the generated URL
  // -------------------------------------------------------------------------

  it('calls writeToClipboard with the permalink URL', async () => {
    const user = userEvent.setup()
    const writeToClipboard = vi.fn().mockResolvedValue(undefined)

    renderShareButton({
      engine: 'neurosynth',
      query: 'ans(x) :- PeakReported(x, y, z, s)',
      writeToClipboard,
      getHref: () => 'http://localhost:3100/',
    })

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /share/i })).not.toBeDisabled()
    })
    await user.click(screen.getByRole('button', { name: /share/i }))

    expect(writeToClipboard).toHaveBeenCalledOnce()
  })

  // -------------------------------------------------------------------------
  // 5. Shows 'Copied!' tooltip after clicking
  // -------------------------------------------------------------------------

  it("shows 'Copied!' feedback after clicking the Share button", async () => {
    const user = userEvent.setup()
    const writeToClipboard = vi.fn().mockResolvedValue(undefined)

    renderShareButton({
      engine: 'neurosynth',
      query: 'ans(x) :- PeakReported(x, y, z, s)',
      writeToClipboard,
    })

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /share/i })).not.toBeDisabled()
    })
    await user.click(screen.getByRole('button', { name: /share/i }))

    await waitFor(() => {
      expect(screen.getByText(/copied/i)).toBeInTheDocument()
    })
  })

  // -------------------------------------------------------------------------
  // 6. 'Copied!' feedback disappears after delay
  // -------------------------------------------------------------------------

  it("'Copied!' feedback disappears after a short delay", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })

    try {
      const writeToClipboard = vi.fn().mockResolvedValue(undefined)

      renderShareButton({
        engine: 'neurosynth',
        query: 'ans(x) :- PeakReported(x, y, z, s)',
        writeToClipboard,
      })

      // Wait for button to be enabled using real async polling
      await waitFor(() => {
        expect(screen.getByRole('button', { name: /share/i })).not.toBeDisabled()
      })

      // Click the button
      act(() => {
        screen.getByRole('button', { name: /share/i }).click()
      })

      // Feedback should appear
      await waitFor(() => {
        expect(screen.getByText(/copied/i)).toBeInTheDocument()
      })

      // Advance time past the tooltip duration (2 seconds)
      act(() => {
        vi.advanceTimersByTime(2500)
      })

      await waitFor(() => {
        expect(screen.queryByText(/copied/i)).not.toBeInTheDocument()
      })
    } finally {
      vi.useRealTimers()
    }
  })
})

// ---------------------------------------------------------------------------
// Permalink loading tests (hash-based routing on app load)
// ---------------------------------------------------------------------------

import { parsePermalinkHash, buildPermalinkUrl } from '../../utils/permalink'

describe('permalink utilities', () => {
  // -------------------------------------------------------------------------
  // 7. Parsing valid permalink hash
  // -------------------------------------------------------------------------

  it('parses a valid permalink hash and returns engine + query', () => {
    const query = 'ans(x) :- PeakReported(x, y, z, s)'
    const encoded = btoa(query)
    const hash = `#/neurosynth?q=${encoded}`

    const result = parsePermalinkHash(hash)
    expect(result).not.toBeNull()
    expect(result!.engine).toBe('neurosynth')
    expect(result!.query).toBe(query)
  })

  it('parses a permalink hash for destrieux engine', () => {
    const query = 'ans(x, y) :- LeftRegion(x, y)'
    const encoded = btoa(query)
    const hash = `#/destrieux?q=${encoded}`

    const result = parsePermalinkHash(hash)
    expect(result).not.toBeNull()
    expect(result!.engine).toBe('destrieux')
    expect(result!.query).toBe(query)
  })

  // -------------------------------------------------------------------------
  // 8. buildPermalinkUrl constructs the correct URL
  // -------------------------------------------------------------------------

  it('builds a permalink URL with engine and base64-encoded query', () => {
    const query = 'ans(x) :- Study(x, y)'
    const base = 'http://localhost:3100/'

    const url = buildPermalinkUrl(base, 'neurosynth', query)
    expect(url).toContain('#/neurosynth?q=')
    const hashPart = url.split('#')[1]
    const qParam = new URLSearchParams(hashPart.split('?')[1]).get('q')
    expect(qParam).not.toBeNull()
    expect(atob(qParam!)).toBe(query)
  })

  // -------------------------------------------------------------------------
  // 9. Handles invalid base64 gracefully
  // -------------------------------------------------------------------------

  it('returns null for invalid base64 in URL hash', () => {
    const hash = '#/neurosynth?q=!!!invalid_base64!!!'
    const result = parsePermalinkHash(hash)
    expect(result).toBeNull()
  })

  // -------------------------------------------------------------------------
  // 10. Handles missing query param gracefully
  // -------------------------------------------------------------------------

  it('returns null when the q param is missing from hash', () => {
    const hash = '#/neurosynth'
    const result = parsePermalinkHash(hash)
    expect(result).toBeNull()
  })

  it('returns null when the hash has no search params', () => {
    const hash = '#/neurosynth?'
    const result = parsePermalinkHash(hash)
    expect(result).toBeNull()
  })

  // -------------------------------------------------------------------------
  // 11. Handles missing engine gracefully
  // -------------------------------------------------------------------------

  it('returns null when the engine portion is missing from hash', () => {
    const query = 'ans(x) :- Study(x, y)'
    const hash = `#?q=${btoa(query)}`
    const result = parsePermalinkHash(hash)
    expect(result).toBeNull()
  })

  it('returns null for an empty hash', () => {
    const result = parsePermalinkHash('')
    expect(result).toBeNull()
  })

  it('returns null for a hash without the leading /', () => {
    const query = 'ans(x) :- Study(x, y)'
    const hash = `#neurosynth?q=${btoa(query)}`
    const result = parsePermalinkHash(hash)
    // We require the #/ prefix for correctness
    expect(result).toBeNull()
  })
})
