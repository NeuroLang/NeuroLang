/**
 * OverlayManager.tsx
 *
 * Lists active brain overlays with their names and colors, provides remove
 * buttons for each overlay, and renders a color bar for probabilistic
 * (VBROverlay) overlays showing the probability scale.
 *
 * The component reads from and writes to the BrainOverlayContext.
 */
import React, { useState } from 'react'
import { useBrainOverlay } from '../context/useBrainOverlay'
import { type BrainOverlay, type OverlayColormap } from '../context/BrainOverlayContext'

// ---------------------------------------------------------------------------
// Colormap gradient definitions (CSS linear-gradient)
// ---------------------------------------------------------------------------

/** Map from OverlayColormap to a representative CSS gradient string. */
const COLORMAP_GRADIENT: Record<OverlayColormap, string> = {
  hot: 'linear-gradient(to right, #000000, #8b0000, #ff4500, #ff8c00, #ffff00, #ffffff)',
  blue: 'linear-gradient(to right, #000033, #0000cc, #0080ff, #80c0ff, #ffffff)',
  green: 'linear-gradient(to right, #001500, #006400, #00aa00, #80ff80, #ffffff)',
  red: 'linear-gradient(to right, #1a0000, #880000, #ff0000, #ff9999, #ffffff)',
  yellow: 'linear-gradient(to right, #1a1a00, #888800, #e0e000, #ffff00, #ffffff)',
  cyan: 'linear-gradient(to right, #001a1a, #008888, #00e0e0, #80ffff, #ffffff)',
  pink: 'linear-gradient(to right, #1a001a, #880088, #e000e0, #ff80ff, #ffffff)',
  violet: 'linear-gradient(to right, #0a0028, #5500bb, #8800ff, #cc80ff, #ffffff)',
}

/** Map from OverlayColormap to display name. */
const COLORMAP_NAMES: Record<OverlayColormap, string> = {
  hot: 'Hot',
  blue: 'Blue',
  green: 'Green',
  red: 'Red',
  yellow: 'Yellow',
  cyan: 'Cyan',
  pink: 'Pink',
  violet: 'Violet',
}

/** A swatch color for the overlay list item (solid color representative). */
const COLORMAP_SWATCH: Record<OverlayColormap, string> = {
  hot: '#ff4500',
  blue: '#0080ff',
  green: '#00aa00',
  red: '#ff0000',
  yellow: '#e0e000',
  cyan: '#00e0e0',
  pink: '#e000e0',
  violet: '#8800ff',
}

// ---------------------------------------------------------------------------
// ColorBar
// ---------------------------------------------------------------------------

interface ColorBarProps {
  /** The colormap to render. */
  colormap: OverlayColormap
  /** Label for the low end of the scale. */
  lowLabel?: string
  /** Label for the high end of the scale. */
  highLabel?: string
}

/**
 * Renders a horizontal color gradient bar with labels at each end.
 * Used to show the probability scale for VBROverlay results.
 */
export function ColorBar({
  colormap,
  lowLabel = '0',
  highLabel = '1',
}: ColorBarProps): React.ReactElement {
  const gradient = COLORMAP_GRADIENT[colormap]
  return (
    <div className="color-bar" data-testid="color-bar">
      <span className="color-bar-label color-bar-label--low">{lowLabel}</span>
      <div
        className="color-bar-gradient"
        style={{ background: gradient }}
        aria-label={`Color map: ${colormap}`}
        data-testid="color-bar-gradient"
      />
      <span className="color-bar-label color-bar-label--high">{highLabel}</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// OverlayControls
// ---------------------------------------------------------------------------

interface OverlayControlsProps {
  overlay: BrainOverlay
  onToggleVisibility: (id: string) => void
  onRemove: (id: string) => void
  onThresholdChange: (id: string, threshold: number) => void
  onColormapChange: (id: string, colormap: OverlayColormap) => void
  onColorbarMinChange: (id: string, min: number) => void
  onColorbarMaxChange: (id: string, max: number) => void
  onDownload: (id: string) => void
}

function OverlayControls({
  overlay,
  onToggleVisibility,
  onRemove,
  onThresholdChange,
  onColormapChange,
  onColorbarMinChange,
  onColorbarMaxChange,
  onDownload,
}: OverlayControlsProps): React.ReactElement {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const swatch = COLORMAP_SWATCH[overlay.colormap]

  return (
    <div className="overlay-item" data-testid="overlay-item">
      <div className="overlay-item-header">
        <button
          className="overlay-visibility-toggle"
          onClick={() => onToggleVisibility(overlay.id)}
          aria-label={overlay.visible ? 'Hide overlay' : 'Show overlay'}
          title={overlay.visible ? 'Hide overlay' : 'Show overlay'}
          data-testid="overlay-visibility-toggle"
          style={{
            opacity: overlay.visible ? 1 : 0.4,
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            fontSize: '16px',
            padding: '4px',
          }}
        >
          {overlay.visible ? '👁' : '👁‍🗨'}
        </button>
        <span
          className="overlay-item-swatch"
          style={{ backgroundColor: swatch, opacity: overlay.visible ? 1 : 0.4 }}
          aria-hidden="true"
        />
        <span
          className="overlay-item-name"
          title={overlay.name}
          style={{ opacity: overlay.visible ? 1 : 0.4 }}
        >
          {overlay.name}
        </span>
        <div className="overlay-item-actions">
          <button
            className="overlay-item-download-btn"
            onClick={() => onDownload(overlay.id)}
            aria-label={`Download overlay ${overlay.name}`}
            title="Download as NIfTI"
            data-testid="overlay-download-btn"
          >
            ⬇
          </button>
          <button
            className="overlay-item-settings-btn"
            onClick={() => setShowAdvanced(!showAdvanced)}
            aria-label={showAdvanced ? 'Hide settings' : 'Show settings'}
            title="Overlay settings"
            data-testid="overlay-settings-btn"
          >
            ⚙
          </button>
          <button
            className="overlay-item-remove-btn"
            onClick={() => onRemove(overlay.id)}
            aria-label={`Remove overlay ${overlay.name}`}
            title="Remove overlay"
            data-testid="overlay-remove-btn"
          >
            ✕
          </button>
        </div>
      </div>

      {overlay.isProbabilistic && (
        <ColorBar
          colormap={overlay.colormap}
          lowLabel={(overlay.colorbarMin ?? 0).toFixed(2)}
          highLabel={overlay.colorbarMax.toFixed(2)}
        />
      )}

      {showAdvanced && (
        <div className="overlay-item-advanced" data-testid="overlay-advanced-controls">
          <div className="overlay-control-row">
            <label className="overlay-control-label">
              Threshold: {overlay.threshold.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={overlay.threshold}
              onChange={(e) => onThresholdChange(overlay.id, parseFloat(e.target.value))}
              className="overlay-control-slider"
              data-testid="overlay-threshold-slider"
            />
          </div>

          <div className="overlay-control-row">
            <label className="overlay-control-label">Colormap</label>
            <select
              value={overlay.colormap}
              onChange={(e) => onColormapChange(overlay.id, e.target.value as OverlayColormap)}
              className="overlay-control-select"
              data-testid="overlay-colormap-select"
            >
              {Object.entries(COLORMAP_NAMES).map(([value, name]) => (
                <option key={value} value={value}>
                  {name}
                </option>
              ))}
            </select>
          </div>

          <div className="overlay-control-row">
            <label className="overlay-control-label">Colorbar Min</label>
            <input
              type="number"
              step="0.01"
              value={overlay.colorbarMin}
              onChange={(e) => onColorbarMinChange(overlay.id, parseFloat(e.target.value))}
              className="overlay-control-input"
              data-testid="overlay-colorbar-min-input"
            />
          </div>

          <div className="overlay-control-row">
            <label className="overlay-control-label">Colorbar Max</label>
            <input
              type="number"
              step="0.01"
              value={overlay.colorbarMax}
              onChange={(e) => onColorbarMaxChange(overlay.id, parseFloat(e.target.value))}
              className="overlay-control-input"
              data-testid="overlay-colorbar-max-input"
            />
          </div>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// OverlayItem
// ---------------------------------------------------------------------------

interface OverlayItemProps {
  overlay: BrainOverlay
  onToggleVisibility: (id: string) => void
  onRemove: (id: string) => void
  onThresholdChange: (id: string, threshold: number) => void
  onColormapChange: (id: string, colormap: OverlayColormap) => void
  onColorbarMinChange: (id: string, min: number) => void
  onColorbarMaxChange: (id: string, max: number) => void
  onDownload: (id: string) => void
}

function OverlayItem({
  overlay,
  onToggleVisibility,
  onRemove,
  onThresholdChange,
  onColormapChange,
  onColorbarMinChange,
  onColorbarMaxChange,
  onDownload,
}: OverlayItemProps): React.ReactElement {
  return (
    <OverlayControls
      overlay={overlay}
      onToggleVisibility={onToggleVisibility}
      onRemove={onRemove}
      onThresholdChange={onThresholdChange}
      onColormapChange={onColormapChange}
      onColorbarMinChange={onColorbarMinChange}
      onColorbarMaxChange={onColorbarMaxChange}
      onDownload={onDownload}
    />
  )
}

// ---------------------------------------------------------------------------
// OverlayManager
// ---------------------------------------------------------------------------

export interface OverlayManagerProps {
  /** Optional test ID for easier querying in tests. */
  'data-testid'?: string
}

function OverlayManager(props: OverlayManagerProps): React.ReactElement | null {
  const {
    overlays,
    removeOverlay,
    clearOverlays,
    toggleOverlayVisibility,
    updateOverlayThreshold,
    updateOverlayColormap,
    updateOverlayColorbarLimits,
    downloadOverlay,
  } = useBrainOverlay()

  if (overlays.length === 0) return null

  const testId = props['data-testid'] ?? 'overlay-manager'

  return (
    <div className="overlay-manager" data-testid={testId}>
      <div className="overlay-manager-header">
        <span className="overlay-manager-title">
          Overlays ({overlays.length})
        </span>
        <button
          className="overlay-manager-clear-btn"
          onClick={clearOverlays}
          aria-label="Remove all overlays"
          title="Remove all overlays"
          data-testid="overlay-clear-all-btn"
        >
          Clear all
        </button>
      </div>
      <div className="overlay-manager-list">
        {overlays.map((overlay) => (
          <OverlayItem
            key={overlay.id}
            overlay={overlay}
            onToggleVisibility={toggleOverlayVisibility}
            onRemove={removeOverlay}
            onThresholdChange={updateOverlayThreshold}
            onColormapChange={updateOverlayColormap}
            onColorbarMinChange={(id, min) =>
              updateOverlayColorbarLimits(id, min, overlay.colorbarMax)
            }
            onColorbarMaxChange={(id, max) =>
              updateOverlayColorbarLimits(id, overlay.colorbarMin, max)
            }
            onDownload={downloadOverlay}
          />
        ))}
      </div>
    </div>
  )
}

export default OverlayManager
