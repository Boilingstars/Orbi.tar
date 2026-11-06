export const AHEAD = 10;

export function relDelta(from, to, invert) {
  const a = Number(from) || 0;
  const b = Number(to) || 0;
  const diff = b - a;
  if (Math.abs(diff) < 1e-9) return { pct: 0, dir: 'up', flat: true };
  const base = Math.abs(a) < 1e-9 ? Math.abs(b) : Math.abs(a);
  const pct = Math.round((Math.abs(diff) / Math.max(base, 1e-9)) * 100);
  const better = invert ? diff < 0 : diff > 0;
  return { pct, dir: better ? 'up' : 'down', flat: false };
}

export function vsAhead(now, next, invert) {
  if (next == null) return null;
  return relDelta(now, next, invert);
}

export function vsAlt(ours, theirs, invert) {
  if (theirs == null) return null;
  return relDelta(theirs, ours, invert);
}

export function deltaHtml(delta) {
  if (!delta) return '<span class="met-delta is-flat">—</span>';
  if (delta.flat) return `<span class="met-delta is-flat">${delta.pct}%</span>`;
  const arrow = delta.dir === 'up' ? '↗' : '↘';
  return `<span class="met-delta is-${delta.dir}">${delta.pct}% ${arrow}</span>`;
}

export function formatUsd(n) {
  return (Number(n) || 0).toLocaleString('ru-RU');
}

export function uiToGoal(obj) {
  return obj === 'money' ? 'revenue' : 'priority';
}

export function oppositeUi(obj) {
  return obj === 'money' ? 'priorities' : 'money';
}
