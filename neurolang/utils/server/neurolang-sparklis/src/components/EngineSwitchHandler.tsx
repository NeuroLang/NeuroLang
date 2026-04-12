/**
 * EngineSwitchHandler.tsx
 *
 * Headless component that coordinates the side effects of an engine switch:
 *   1. Clears the current query (visual builder model + code editor text)
 *   2. Clears the execution results (resets to idle)
 *   3. Removes all brain viewer overlays (atlas is reloaded by BrainViewer)
 *
 * This component renders nothing; it only runs effects.  Mount it once inside
 * the providers that it depends on (see App.tsx).
 *
 * Note: Schema refresh and example-query reload are handled automatically by
 * SchemaContext and ExampleQueries respectively, which both depend on
 * `selectedEngine` and re-fetch whenever it changes.
 */
import { useEffect, useRef } from 'react'
import { useEngine } from '../context/useEngine'
import { useQuery } from '../context/useQuery'
import { useExecution } from '../context/useExecution'
import { useBrainOverlay } from '../context/useBrainOverlay'

function EngineSwitchHandler(): null {
  const { selectedEngine } = useEngine()
  const { model, refresh, setDatalogText } = useQuery()
  const { resetExecution } = useExecution()
  const { clearOverlays } = useBrainOverlay()

  // Track whether this is the initial mount so we don't clear state on
  // first render (when the page loads with no engine selected yet).
  const isFirstMount = useRef(true)

  useEffect(() => {
    // Skip clearing on the very first mount.
    if (isFirstMount.current) {
      isFirstMount.current = false
      return
    }

    // Engine changed — clear all transient state.

    // 1. Clear the visual query builder and code editor.
    model.reset()
    // Wrap with builder-update flag so the debounced parser is suppressed.
    setDatalogText('')
    refresh()

    // 2. Clear query execution results / errors.
    resetExecution()

    // 3. Remove all brain viewer overlays (atlas reload is handled by
    //    BrainViewer's own useEffect([selectedEngine])).
    clearOverlays()
  }, [selectedEngine]) // eslint-disable-line react-hooks/exhaustive-deps
  // ^ Deliberately omitting stable callbacks from deps (they won't change
  //   across the provider's lifetime, but listing them would cause TS to
  //   complain about exhaustive-deps without benefit).

  return null
}

export default EngineSwitchHandler
