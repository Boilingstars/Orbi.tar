export function seriesChart(series, opts = {}) {
  if (!series.length) return '<p class="empty">нет ряда</p>';

  const n = series.length;
  const lastStep = n - 1;
  const xMin = opts.xMin ?? 0;
  const xMax = Math.max(opts.xMax ?? lastStep, lastStep, xMin + 1);
  const xNow = clampNum(opts.xNow ?? lastStep, xMin, xMax);

  const dataMin = Math.min(...series);
  const dataMax = Math.max(...series);
  let ymin = opts.yMin;
  let ymax = opts.yMax;
  if (ymin == null || ymax == null) {
    const pad = ((dataMax - dataMin) || 1) * 0.15;
    ymin = (ymin == null ? dataMin : ymin) - (opts.yMin == null ? pad : 0);
    ymax = (ymax == null ? dataMax : ymax) + (opts.yMax == null ? pad : 0);
  }
  if (ymax <= ymin) {
    ymin -= 1;
    ymax += 1;
  }

  const W = 268;
  const H = 132;
  const L = 48;
  const R = 12;
  const T = 10;
  const B = 28;
  const pw = W - L - R;
  const ph = H - T - B;
  const refs = opts.refs || [];
  const xSpan = xMax - xMin;

  const xAt = (step) => L + ((step - xMin) / xSpan) * pw;
  const yAt = (v) => T + (1 - (v - ymin) / (ymax - ymin)) * ph;
  const clampY = (v) => Math.max(T, Math.min(T + ph, yAt(v)));

  const yTicks = opts.yTicks || ticksBetween(ymin, ymax, 4);
  const xTicks = opts.xTicks || ticksBetween(xMin, xMax, 4);

  const grid = [
    ...yTicks.map((v) => {
      const y = yAt(v);
      return `<line class="chart-grid" x1="${L}" y1="${y}" x2="${L + pw}" y2="${y}"/>`;
    }),
    ...xTicks.map((step) => {
      const x = xAt(step);
      return `<line class="chart-grid" x1="${x}" y1="${T}" x2="${x}" y2="${T + ph}"/>`;
    }),
  ].join('');

  const yLabels = yTicks.map((v, i) => {
    const y = yAt(v);
    let ty = y + 3;
    if (i === yTicks.length - 1) ty = y + 10;
    if (i === 0) ty = y - 3;
    return `<text class="chart-label" x="${L - 6}" y="${ty}" text-anchor="end">${fmt(v)}</text>`;
  }).join('');

  const xLabels = xTicks.map((step) => {
    const x = xAt(step);
    return `<text class="chart-label" x="${x}" y="${T + ph + 12}" text-anchor="middle">${step}</text>`;
  }).join('');

  const yAxis = opts.yAxis || '';
  const xAxis = opts.xAxis || 'шаги';
  const yAxisTitle = yAxis
    ? `<text class="chart-axis-name" transform="translate(11, ${T + ph / 2}) rotate(-90)" text-anchor="middle">${escapeXml(yAxis)}</text>`
    : '';
  const xAxisTitle = `<text class="chart-axis-name" x="${L + pw / 2}" y="${H - 3}" text-anchor="middle">${escapeXml(xAxis)}</text>`;

  const refLines = refs.filter((r) => r.value >= ymin && r.value <= ymax).map((r) => {
    const y = clampY(r.value);
    return `<line class="chart-ref ${r.cls || ''}" x1="${L}" y1="${y}" x2="${L + pw}" y2="${y}"/>`;
  }).join('');

  const pts = series.map((v, i) => `${xAt(i).toFixed(2)},${clampY(v).toFixed(2)}`).join(' ');
  const end = series[n - 1];
  const endX = xAt(lastStep);
  const endY = clampY(end);
  const nowX = xAt(xNow);
  const title = opts.title || '';
  const caption = Array.isArray(opts.caption)
    ? opts.caption.map(escapeXml).join('<br>')
    : escapeXml(opts.caption || `текущее состояние ${fmt(end)}`);

  return `
    <section class="chart-card">
      <h3 class="prop-card-h">${escapeXml(title)}</h3>
      <svg class="series-chart" viewBox="0 0 ${W} ${H}" width="100%" height="${H}" preserveAspectRatio="none" role="img" aria-label="${escapeXml(title)}">
        ${grid}
        ${refLines}
        <line class="chart-now-x" x1="${nowX}" y1="${T}" x2="${nowX}" y2="${T + ph}"/>
        ${yAxisTitle}${xAxisTitle}
        ${yLabels}${xLabels}
        <polyline class="chart-line" points="${pts}"/>
        <circle class="chart-now" cx="${endX}" cy="${endY}" r="2.2"/>
      </svg>
      <p class="chart-caption">${caption}</p>
    </section>
  `;
}

function ticksBetween(min, max, count) {
  if (max <= min) return [min];
  const step = Math.max(1, Math.round((max - min) / count));
  const out = [];
  for (let v = min; v < max; v += step) out.push(v);
  out.push(max);
  return uniqueInts(out);
}

function clampNum(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function fmt(v) {
  if (Math.abs(v - Math.round(v)) < 0.05) return String(Math.round(v));
  return v.toFixed(1);
}

function uniqueInts(arr) {
  return [...new Set(arr.map((n) => Math.round(n)))];
}

function escapeXml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}
