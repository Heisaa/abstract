import { useCallback, useEffect, useLayoutEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type Ref } from 'react'

type ZoomMode = 'fit' | number

type PreviewProps = {
  canvasRef: Ref<HTMLCanvasElement>
  hasImage: boolean
  // Changing this resets zoom and pan (e.g. when a new image loads).
  resetToken: number
  outputWidth: number
  outputHeight: number
  split: { enabled: boolean; position: number }
  onSplitChange: (position: number) => void
  eyedropper: boolean
  // Normalized output coordinates (0..1) of an eyedropper click.
  onPick: (u: number, v: number) => void
  busy: boolean
}

const MIN_ZOOM = 0.05
const MAX_ZOOM = 16

export function Preview({
  canvasRef,
  hasImage,
  resetToken,
  outputWidth,
  outputHeight,
  split,
  onSplitChange,
  eyedropper,
  onPick,
  busy,
}: PreviewProps) {
  const frameRef = useRef<HTMLDivElement | null>(null)
  const stageRef = useRef<HTMLDivElement | null>(null)
  const [frameSize, setFrameSize] = useState({ width: 0, height: 0 })
  const [zoom, setZoom] = useState<ZoomMode>('fit')
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [lastResetToken, setLastResetToken] = useState(resetToken)
  if (resetToken !== lastResetToken) {
    setLastResetToken(resetToken)
    setZoom('fit')
    setPan({ x: 0, y: 0 })
  }
  const dragRef = useRef<{ pointerId: number; startX: number; startY: number; panX: number; panY: number; moved: boolean } | null>(null)
  const splitDragRef = useRef<number | null>(null)

  useLayoutEffect(() => {
    const frame = frameRef.current
    if (!frame) {
      return
    }
    const observer = new ResizeObserver(([entry]) => {
      setFrameSize({ width: entry.contentRect.width, height: entry.contentRect.height })
    })
    observer.observe(frame)
    return () => observer.disconnect()
  }, [])

  const fitScale =
    outputWidth > 0 && frameSize.width > 0 ? Math.min((frameSize.width - 32) / outputWidth, (frameSize.height - 32) / outputHeight) : 1
  const scale = zoom === 'fit' ? Math.max(MIN_ZOOM, fitScale) : zoom
  const effectivePan = zoom === 'fit' ? { x: 0, y: 0 } : pan

  const fit = useCallback(() => {
    setZoom('fit')
    setPan({ x: 0, y: 0 })
  }, [])

  const zoomAround = useCallback(
    (nextScale: number, clientX: number, clientY: number) => {
      const frame = frameRef.current
      if (!frame) {
        return
      }
      const clamped = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, nextScale))
      const rect = frame.getBoundingClientRect()
      const cursorX = clientX - rect.left - rect.width / 2
      const cursorY = clientY - rect.top - rect.height / 2
      const ratio = clamped / scale
      setPan({
        x: cursorX - (cursorX - effectivePan.x) * ratio,
        y: cursorY - (cursorY - effectivePan.y) * ratio,
      })
      setZoom(clamped)
    },
    [effectivePan.x, effectivePan.y, scale],
  )

  // Wheel zoom needs a non-passive listener to prevent page scroll.
  useEffect(() => {
    const frame = frameRef.current
    if (!frame || !hasImage) {
      return
    }
    const onWheel = (event: WheelEvent) => {
      event.preventDefault()
      zoomAround(scale * Math.exp(-event.deltaY * 0.0015), event.clientX, event.clientY)
    }
    frame.addEventListener('wheel', onWheel, { passive: false })
    return () => frame.removeEventListener('wheel', onWheel)
  }, [hasImage, scale, zoomAround])

  const outputCoordinates = (clientX: number, clientY: number) => {
    const stage = stageRef.current
    if (!stage) {
      return undefined
    }
    const rect = stage.getBoundingClientRect()
    const u = (clientX - rect.left) / rect.width
    const v = (clientY - rect.top) / rect.height
    return u >= 0 && u <= 1 && v >= 0 && v <= 1 ? { u, v } : undefined
  }

  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!hasImage || event.button !== 0) {
      return
    }
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      panX: effectivePan.x,
      panY: effectivePan.y,
      moved: false,
    }
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  const onPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current
    if (!drag || drag.pointerId !== event.pointerId) {
      return
    }
    const dx = event.clientX - drag.startX
    const dy = event.clientY - drag.startY
    if (!drag.moved && Math.hypot(dx, dy) < 4) {
      return
    }
    drag.moved = true
    if (zoom === 'fit') {
      setZoom(scale)
    }
    setPan({ x: drag.panX + dx, y: drag.panY + dy })
  }

  const onPointerUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current
    dragRef.current = null
    if (!drag || drag.pointerId !== event.pointerId) {
      return
    }
    if (!drag.moved && eyedropper) {
      const point = outputCoordinates(event.clientX, event.clientY)
      if (point) {
        onPick(point.u, point.v)
      }
    }
  }

  const onSplitPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.stopPropagation()
    splitDragRef.current = event.pointerId
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  const onSplitPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (splitDragRef.current !== event.pointerId || !stageRef.current) {
      return
    }
    const rect = stageRef.current.getBoundingClientRect()
    onSplitChange(Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)))
  }

  const displayWidth = outputWidth * scale
  const displayHeight = outputHeight * scale

  return (
    <div
      ref={frameRef}
      className={`canvas-frame ${eyedropper ? 'is-picking' : ''} ${hasImage ? 'has-image' : ''}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={() => {
        dragRef.current = null
      }}
      onDoubleClick={fit}
    >
      {!hasImage ? <div className="empty-state">No image loaded — open, paste, or drop one</div> : null}
      <div
        ref={stageRef}
        className="canvas-stage"
        style={{
          width: displayWidth,
          height: displayHeight,
          transform: `translate(-50%, -50%) translate(${effectivePan.x}px, ${effectivePan.y}px)`,
          visibility: hasImage ? 'visible' : 'hidden',
        }}
      >
        <canvas ref={canvasRef} aria-label="Filtered image preview" className={scale >= 2 ? 'pixelated' : ''} />
        {hasImage && split.enabled ? (
          <div
            className="split-handle"
            style={{ left: `${split.position * 100}%` }}
            onPointerDown={onSplitPointerDown}
            onPointerMove={onSplitPointerMove}
            onPointerUp={() => {
              splitDragRef.current = null
            }}
            role="slider"
            aria-label="Compare split position"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(split.position * 100)}
          >
            <span>Original | Filter</span>
          </div>
        ) : null}
      </div>
      {busy ? <div className="busy-indicator">Rendering…</div> : null}
      {hasImage ? (
        <div className="zoom-toolbar" onPointerDown={(event) => event.stopPropagation()} onDoubleClick={(event) => event.stopPropagation()}>
          <button type="button" onClick={fit} className={zoom === 'fit' ? 'active' : ''}>
            Fit
          </button>
          <button
            type="button"
            onClick={() => {
              setZoom(1)
              setPan({ x: 0, y: 0 })
            }}
          >
            100%
          </button>
          <button type="button" aria-label="Zoom out" onClick={() => setZoom(Math.max(MIN_ZOOM, scale / 1.25))}>
            −
          </button>
          <button type="button" aria-label="Zoom in" onClick={() => setZoom(Math.min(MAX_ZOOM, scale * 1.25))}>
            +
          </button>
          <span>{Math.round(scale * 100)}%</span>
        </div>
      ) : null}
    </div>
  )
}
