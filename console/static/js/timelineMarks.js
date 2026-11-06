/** Каталог меток таймлайна: первый переход порога по уже пришедшим кадрам. */

export let MARK_RULES = [];

export async function loadMarkCatalog() {
  const res = await fetch('/src/timeline-marks.json', { cache: 'no-store' });
  if (!res.ok) return MARK_RULES;
  MARK_RULES = await res.json();
  return MARK_RULES;
}

export function detectMarks(snap, state) {
  if (!snap?.meta || !MARK_RULES.length) return [];
  const steps = Math.max(1, snap.meta.steps || state.scenario.steps || 1);
  const seed = snap.meta.jobs_seed ?? snap.meta.jobs.length;
  const potential = snap.meta.potential_seed_usd || snap.meta.potential_revenue_usd || 0;
  const reserve = snap.meta.reserve_soc_pct ?? state.model?.reserve_soc_pct ?? 30;
  const caps = (snap.meta.ids || []).map((id) => {
    const sat = state.satellites.find((row) => row.id === id);
    return sat?.capacity_wh || 1;
  });
  const uniqueLow = new Set();
  const done = new Array(seed).fill(false);
  const fired = new Set();
  const marks = [];

  for (let step = 0; step <= steps; step += 1) {
    const frame = snap.frames.get(step);
    if (!frame) break;
    const revenue = frame.metrics?.revenue_usd || 0;
    const missed = frame.metrics?.jobs_due_missed || 0;
    (frame.sat || []).forEach((row, i) => {
      const cap = caps[i] || 1;
      if ((100 * row[0]) / cap < reserve) uniqueLow.add(snap.meta.ids[i]);
    });
    for (const [index, , , flag] of frame.job || []) {
      if (index < seed) done[index] = Boolean(flag);
    }
    const originalDone = seed ? done.filter(Boolean).length / seed : 0;
    for (const rule of MARK_RULES) {
      if (fired.has(rule.id)) continue;
      let hit = false;
      if (rule.rule === 'revenue_frac') {
        hit = potential > 0 && revenue >= rule.threshold * potential;
      } else if (rule.rule === 'overdue_count') {
        const limit = rule.threshold < 1 ? rule.threshold * seed : rule.threshold;
        hit = missed >= limit;
      } else if (rule.rule === 'unique_below_reserve') {
        hit = uniqueLow.size >= rule.threshold;
      } else if (rule.rule === 'original_done_frac') {
        hit = originalDone >= rule.threshold;
      }
      if (!hit) continue;
      fired.add(rule.id);
      marks.push({
        id: rule.id,
        step,
        left: step / steps,
        tone: rule.tone,
        title: rule.title,
        body: rule.body,
      });
    }
  }
  return marks;
}
