/**
 * ShareButton.tsx
 *
 * A button that generates a shareable permalink URL for the current query.
 *
 * Behavior:
 *   - Generates a URL with the current query encoded in the URL hash:
 *     <baseUrl>#/<engine>?q=<base64encodedquery>
 *   - Copies the URL to the clipboard
 *   - Shows 'Copied!' tooltip/feedback for 2 seconds after clicking
 *
 * The button is only active when both an engine is selected and the query is non-empty.
 *
 * Props:
 *   - writeToClipboard: optional custom function to write text to clipboard
 *     (defaults to navigator.clipboard.writeText). Useful for testing.
 *   - getHref: optional custom function to get the current URL
 *     (defaults to () => window.location.href). Useful for testing.
 */
import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useEngine } from '../context/useEngine'
import { useQuery } from '../context/useQuery'
import { buildPermalinkUrl } from '../utils/permalink'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Duration in milliseconds to show the 'Copied!' feedback. */
const COPIED_FEEDBACK_DURATION_MS = 2000

// ---------------------------------------------------------------------------
// Default clipboard writer
// ---------------------------------------------------------------------------

async function defaultWriteToClipboard(text: string): Promise<void> {
  await navigator.clipboard.writeText(text)
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface ShareButtonProps {
  /** Optional clipboard writer (for testing). Defaults to navigator.clipboard.writeText. */
  writeToClipboard?: (text: string) => Promise<void>
  /** Optional URL getter (for testing). Defaults to () => window.location.href. */
  getHref?: () => string
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

function ShareButton({
  writeToClipboard = defaultWriteToClipboard,
  getHref = () => window.location.href,
}: ShareButtonProps = {}): React.ReactElement {
  const { selectedEngine } = useEngine()
  const { datalogText } = useQuery()
  const [copied, setCopied] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Clear the timer on unmount to avoid state updates after unmount.
  useEffect(() => {
    return () => {
      if (timerRef.current !== null) {
        clearTimeout(timerRef.current)
      }
    }
  }, [])

  const canShare = !!selectedEngine && !!datalogText.trim()

  const handleClick = useCallback(async () => {
    if (!canShare) return

    const url = buildPermalinkUrl(getHref(), selectedEngine!, datalogText)

    try {
      await writeToClipboard(url)
    } catch {
      // Clipboard API may be unavailable in some environments; fail silently.
    }

    // Show 'Copied!' feedback
    setCopied(true)

    // Clear any existing timer
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current)
    }

    // Reset after the feedback duration
    timerRef.current = setTimeout(() => {
      timerRef.current = null
      setCopied(false)
    }, COPIED_FEEDBACK_DURATION_MS)
  }, [canShare, selectedEngine, datalogText, writeToClipboard, getHref])

  return (
    <div className="share-button-container">
      <button
        className={`share-btn${!canShare ? ' share-btn--disabled' : ''}`}
        onClick={handleClick}
        disabled={!canShare}
        aria-label="Share query permalink"
        data-testid="share-btn"
        title={
          !selectedEngine
            ? 'Select an engine first'
            : !datalogText.trim()
              ? 'Enter a query first'
              : 'Copy permalink to clipboard'
        }
      >
        🔗 Share
      </button>
      {copied && (
        <span
          className="share-btn-copied"
          role="status"
          aria-live="polite"
          data-testid="share-copied-tooltip"
        >
          Copied!
        </span>
      )}
    </div>
  )
}

export default ShareButton
