/** Кривая качества политики: q из кадров. Заливка до курсора. */

const W = 1000;
const H = 48;
const TOP = 8;
const BASE = 40;

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function clampY(y) {
  return Math.max(0.08, Math.min(0.96, y));
}

function keysFromQ(qs, steps) {
  const n = Math.max(1, steps || 1);
  const keys = [];
  let last = 0.5;
  for (let i = 0; i <= n; i += 1) {
    const q = qs[i];
    if (typeof q === 'number' && Number.isFinite(q)) last = clampY(q);
    keys.push([i / n, last]);
  }
  if (!keys.length) keys.push([0, 0.5], [1, 0.5]);
  return keys;
}

function lerpKeys(t, keys) {
  const x = clamp01(t);
  for (let i = 0; i < keys.length - 1; i += 1) {
    const [x0, y0] = keys[i];
    const [x1, y1] = keys[i + 1];
    if (x >= x0 && x <= x1) {
      const u = (x - x0) / (x1 - x0 || 1);
      const s = u * u * (3 - 2 * u);
      return y0 + (y1 - y0) * s;
    }
  }
  return keys[keys.length - 1][1];
}

export function curveY(t, qs = [], steps = 1) {
  return BASE - lerpKeys(t, keysFromQ(qs, steps)) * (BASE - TOP);
}

/** Доля качества политики на шаге, 0..100. Нет кадра — null. */
export function curveAdvantagePct(qs, step, steps) {
  const n = Math.max(0, steps || 0);
  const at = Math.max(0, Math.min(n, Number(step) || 0));
  let q = null;
  for (let i = at; i >= 0; i -= 1) {
    const v = qs[i];
    if (typeof v === 'number' && Number.isFinite(v)) {
      q = v;
      break;
    }
  }
  if (q == null) return null;
  return Math.round(Math.max(0, Math.min(1, q)) * 100);
}

function catmull(pts) {
  if (!pts.length) return '';
  let d = `M ${pts[0].x.toFixed(1)} ${pts[0].y.toFixed(1)}`;
  for (let i = 0; i < pts.length - 1; i += 1) {
    const p0 = pts[Math.max(0, i - 1)];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[Math.min(pts.length - 1, i + 2)];
    const c1x = p1.x + (p2.x - p0.x) / 6;
    const c1y = p1.y + (p2.y - p0.y) / 6;
    const c2x = p2.x - (p3.x - p1.x) / 6;
    const c2y = p2.y - (p3.y - p1.y) / 6;
    d += ` C ${c1x.toFixed(1)} ${c1y.toFixed(1)}, ${c2x.toFixed(1)} ${c2y.toFixed(1)}, ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`;
  }
  return d;
}

export function buildSignalSvg(nowT, ballT, qs = [], marks = [], steps = 1) {
  const keys = keysFromQ(qs, steps);
  const pts = keys.map(([t, y]) => ({
    x: clamp01(t) * W,
    y: BASE - y * (BASE - TOP),
  }));
  const line = catmull(pts);
  const fill = `${line} L ${W} ${BASE} L 0 ${BASE} Z`;
  const nowX = clamp01(nowT) * W;
  const uid = `p${Math.round(nowX)}`;
  const stems = marks.map((m) => {
    const x = m.left * W;
    const y = curveY(m.left, qs, steps);
    return `<line class="tl-stem tone-${m.tone}" x1="${x.toFixed(1)}" y1="${y.toFixed(1)}" x2="${x.toFixed(1)}" y2="${BASE}" />`;
  }).join('');

  return `
    <svg class="tl-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">
      <defs>
        <linearGradient id="${uid}-g" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#b8b0f0" stop-opacity="0.7"/>
          <stop offset="100%" stop-color="#8b7fd4" stop-opacity="0.12"/>
        </linearGradient>
        <clipPath id="${uid}-c"><rect x="0" y="0" width="${nowX.toFixed(1)}" height="${H}"/></clipPath>
      </defs>
      <path d="${fill}" fill="url(#${uid}-g)" clip-path="url(#${uid}-c)"/>
      <path d="${line}" fill="none" stroke="#c4bdf5" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/>
      ${stems}
    </svg>
  `;
}
