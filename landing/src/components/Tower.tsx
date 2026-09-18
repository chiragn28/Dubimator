// Turns the glowing slab in hero.mp4 into a skyscraper. The camera in the video never moves,
// so the slab sits at fixed frame coordinates (1280x720): left edge x=655, right edge x=841,
// top edge sloping from y=147 to y=177, base lost in the clouds around y=530. The SVG uses the
// same viewBox with `slice`, which is exactly the video's `object-fit: cover`, so the tower
// lands on the slab at every viewport size.

const LEFT = 655
const WIDTH = 186
const TOP = 147
const HEIGHT = 390
const SKEW_DEG = 9.16 // atan(30 / 186): the slab's top edge drops 30px across its width

const COLS = 12
const ROWS = 43
const WIN_W = 10
const WIN_H = 5
const COL_STEP = 15
const ROW_STEP = 8.6

type Window = { x: number; y: number; kind: 'lit' | 'dim' | 'off'; delay: number }

// Deterministic so the façade looks the same on every visit.
function hash(i: number) {
  let h = (i + 1) * 2654435761
  h ^= h >>> 15
  h = Math.imul(h, 2246822519)
  h ^= h >>> 13
  return (h >>> 0) / 4294967296
}

const windows: Window[] = []
for (let r = 0; r < ROWS; r++) {
  for (let c = 0; c < COLS; c++) {
    const i = r * COLS + c
    const roll = hash(i)
    windows.push({
      x: 5 + c * COL_STEP,
      y: 12 + r * ROW_STEP,
      kind: roll < 0.62 ? 'lit' : roll < 0.85 ? 'dim' : 'off',
      // Lower floors light first, with a little scatter, as the tower rises past them.
      delay: 1.2 + (ROWS - r) * 0.055 + hash(i + 977) * 0.5,
    })
  }
}

const fill = { lit: '#ffd27a', dim: '#f2a54a', off: '#1a2c3d' }
const opacity = { lit: 0.95, dim: 0.55, off: 0.9 }

export default function Tower() {
  return (
    <svg
      className="tower"
      viewBox="0 0 1280 720"
      preserveAspectRatio="xMidYMid slice"
      aria-hidden="true"
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
    >
      <defs>
        {/* The base fades out where the clouds pass in front of the slab. */}
        <linearGradient id="tower-fade" gradientUnits="userSpaceOnUse" x1="0" y1="330" x2="0" y2="450">
          <stop offset="0" stopColor="#fff" />
          <stop offset="1" stopColor="#fff" stopOpacity="0" />
        </linearGradient>
        <mask id="tower-mask">
          <rect x="0" y="0" width="1280" height="720" fill="url(#tower-fade)" />
        </mask>
        <linearGradient id="tower-glass" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="#0d2033" />
          <stop offset="0.55" stopColor="#12283c" />
          <stop offset="1" stopColor="#0a1a2a" />
        </linearGradient>
        <linearGradient id="tower-sweep" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#fff" stopOpacity="0" />
          <stop offset="0.5" stopColor="#fff" stopOpacity="0.28" />
          <stop offset="1" stopColor="#fff" stopOpacity="0" />
        </linearGradient>
        <clipPath id="tower-face">
          <rect x="0" y="0" width={WIDTH} height={HEIGHT} />
        </clipPath>
      </defs>

      <g mask="url(#tower-mask)">
        <g className="tower-rise">
          {/* Side face, receding to the right. */}
          <polygon
            points={`${LEFT + WIDTH},${TOP + 30} ${LEFT + WIDTH + 26},${TOP + 42} ${LEFT + WIDTH + 26},${TOP + HEIGHT + 20} ${LEFT + WIDTH},${TOP + HEIGHT + 30}`}
            fill="#071320"
            opacity="0.92"
          />

          {/* Front face, drawn upright then skewed to match the slab's perspective. */}
          <g transform={`translate(${LEFT} ${TOP}) skewY(${SKEW_DEG})`}>
            <rect x="0" y="0" width={WIDTH} height={HEIGHT} fill="url(#tower-glass)" opacity="0.9" />
            {/* Vertical mullions */}
            {Array.from({ length: COLS - 1 }, (_, c) => (
              <line key={c} x1={5 + (c + 1) * COL_STEP - 2.5} y1="0" x2={5 + (c + 1) * COL_STEP - 2.5} y2={HEIGHT} stroke="#1d3449" strokeWidth="1" />
            ))}
            {windows.map((w, i) => (
              <rect
                key={i}
                className={w.kind === 'off' ? undefined : 'tower-win'}
                x={w.x}
                y={w.y}
                width={WIN_W}
                height={WIN_H}
                rx="0.6"
                fill={fill[w.kind]}
                opacity={opacity[w.kind]}
                style={{ animationDelay: `${w.delay}s` }}
              />
            ))}
            <g clipPath="url(#tower-face)">
              <rect className="tower-sweep" x="0" y="0" width={WIDTH} height="90" fill="url(#tower-sweep)" />
            </g>
            {/* Crown band */}
            <rect x="0" y="0" width={WIDTH} height="6" fill="#ffd27a" opacity="0.8" />
          </g>

          {/* Spire and aviation beacon, centred on the sloping top edge. */}
          <polygon points="741,161 755,163 748,62" fill="#0a1a2a" />
          <line x1="748" y1="62" x2="748" y2="30" stroke="#0a1a2a" strokeWidth="2" />
          <line x1="748" y1="62" x2="748" y2="30" stroke="#ffd27a" strokeWidth="0.6" opacity="0.7" />
          <circle className="tower-beacon" cx="748" cy="28" r="3" fill="#ff3b3b" />
          <circle className="tower-beacon" cx="748" cy="28" r="7" fill="#ff3b3b" opacity="0.25" />
        </g>
      </g>
    </svg>
  )
}
