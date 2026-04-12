/**
 * PermalinkLoader.tsx
 *
 * A side-effect-only component that loads a query from the URL hash on mount.
 *
 * This component renders nothing; it only triggers the usePermalinkLoader hook
 * which reads the URL hash and, if it encodes a query, sets the engine and
 * loads the query into the editor.
 *
 * Placement: inside the context providers (EngineProvider + QueryProvider)
 * so that the hooks have access to the required context values.
 */
import { usePermalinkLoader } from '../context/usePermalinkLoader'

function PermalinkLoader(): null {
  usePermalinkLoader()
  return null
}

export default PermalinkLoader
