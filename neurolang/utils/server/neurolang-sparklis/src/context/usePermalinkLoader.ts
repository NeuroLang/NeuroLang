/**
 * usePermalinkLoader.ts
 *
 * A hook that reads the URL hash on mount and, if it contains a valid
 * permalink (see permalink.ts for format), sets the engine and loads
 * the encoded query into the editor + visual builder.
 *
 * This hook should be called once near the root of the app (e.g., inside App.tsx
 * or the Layout component) after the contexts it depends on are available.
 */
import { useEffect } from 'react'
import { parsePermalinkHash } from '../utils/permalink'
import { useEngine } from './useEngine'
import { useQuery } from './useQuery'

export function usePermalinkLoader(): void {
  const { setSelectedEngine } = useEngine()
  const { setDatalogText } = useQuery()

  useEffect(() => {
    const hash = window.location.hash
    const result = parsePermalinkHash(hash)
    if (result === null) {
      // No valid permalink in the URL hash — nothing to do.
      return
    }

    // Set the engine
    setSelectedEngine(result.engine)

    // Load the query into the editor (which triggers bidirectional sync)
    setDatalogText(result.query)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // Only run once on mount — intentionally empty deps
}
