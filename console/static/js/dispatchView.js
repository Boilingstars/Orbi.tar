import { dispatchSnapshot, onDispatch } from './api.js?v=whatif1';
import { AHEAD, vsAhead, vsAlt } from './metricsDelta.js';

const ACTIONS = ['idle', 'job', 'calibrate'];

export function watchDispatch(getView, onUpdate) {
  let scheduled = false;
  let last = 0;
  return onDispatch((msg) => {
    if (msg.type !== 'frame' && msg.type !== 'meta' && msg.type !== 'alt') return;
    const view = getView();
    const now = performance.now();
    const urgent = msg.type === 'meta' || msg.step === view;
    if (!urgent && now - last < 400) return;
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      last = performance.now();
      onUpdate();
    });
  });
}

export function applyDispatch(state) {
  const snap = dispatchSnapshot();
  if (!snap || !snap.meta) return false;
  const view = state.previewStep ?? state.step;
  if (!snap.frames.has(view)) return false;

  const meta = snap.meta;
  const loaded = prefix(snap.frames);
  const bySat = new Map(state.satellites.map((sat) => [sat.id, sat]));
  const byJob = new Map(state.jobs.map((job) => [job.id, job]));
  meta.jobs.forEach((spec) => {
    if (byJob.has(spec.id)) return;
    const job = {
      id: spec.id,
      kind: spec.kind,
      release_step: spec.release_step,
      deadline_step: spec.deadline_step,
      work_steps: spec.work_steps,
      eligible_satellites: spec.eligible || [],
      priority: spec.priority,
      value_usd: spec.value_usd,
      remaining_steps: spec.work_steps,
      completed_step: null,
      progress_by: [],
      status: 'waiting',
      assignable: false,
      first_open_step: null,
      xray: null,
    };
    state.jobs.push(job);
    byJob.set(spec.id, job);
  });
  const resolved = resolveJobs(meta, snap.frames, view);
  const access = resolveAccess(meta, snap.frames, view);

  meta.ids.forEach((id, i) => {
    const sat = bySat.get(id);
    if (!sat) return;
    const soc = [];
    const temp = [];
    const tape = [];
    for (let step = 0; step < loaded; step += 1) {
      const row = snap.frames.get(step).sat[i];
      const action = ACTIONS[row[3]] || 'idle';
      const jobId = row[4] >= 0 ? meta.jobs[row[4]].id : null;
      soc.push((100 * row[0]) / (sat.capacity_wh || 1));
      temp.push(row[1]);
      tape.push({ step, action, job_id: jobId });
    }
    const now = snap.frames.get(view).sat[i];
    const action = ACTIONS[now[3]] || 'idle';
    sat.energy_wh = now[0];
    sat.temp_c = now[1];
    sat.calibration_age_steps = now[2];
    sat.action = action;
    sat.job_id = now[4] >= 0 ? meta.jobs[now[4]].id : null;
    sat.assignable = access.take[i] === 1;
    sat.available = true;
    sat.idle_reason = (!sat.assignable && action === 'idle')
      ? 'нет задания, которое можно взять'
      : null;
    sat.rejected = [];
    sat.soc_series = soc;
    sat.temp_series = temp;
    sat.action_tape = tape;
  });

  meta.jobs.forEach((spec, i) => {
    const job = byJob.get(spec.id);
    const dyn = resolved[i];
    if (!job || !dyn) return;
    job.remaining_steps = dyn.remaining;
    job.completed_step = dyn.finished;
    job.progress_by = dyn.progress;
    job.status = statusOf(spec, dyn, view);
    job.assignable = access.open.has(i);
    job.first_open_step = access.firstOpen[i];
    job.xray = null;
  });

  const metrics = snap.frames.get(view).metrics;
  state.metrics = {
    ...metrics,
    jobs_total: meta.jobs_total,
    potential_revenue_usd: meta.potential_revenue_usd,
  };
  state.metricsDelta = resolveMetricsDelta(snap, view, metrics);
  return true;
}

function prefix(frames) {
  let n = 0;
  while (frames.has(n)) n += 1;
  return n;
}

function resolveAccess(meta, frames, view) {
  const n = meta.jobs.length;
  const firstOpen = new Array(n).fill(null);
  let take = (meta.ids || []).map(() => 1);
  let open = new Set();
  for (let step = 0; step <= view; step += 1) {
    const frame = frames.get(step);
    if (!frame) break;
    if (Array.isArray(frame.take) && frame.take.length === take.length) {
      take = frame.take;
    }
    if (Array.isArray(frame.open)) {
      open = new Set(frame.open);
      for (const index of frame.open) {
        if (index >= 0 && index < n && firstOpen[index] == null) firstOpen[index] = step;
      }
    }
  }
  return { firstOpen, open, take };
}

function resolveJobs(meta, frames, view) {
  const n = meta.jobs.length;
  const remaining = meta.jobs.map((job) => job.work_steps);
  const exec = new Array(n).fill(-1);
  const done = new Array(n).fill(0);
  const finished = new Array(n).fill(null);
  const progress = meta.jobs.map(() => []);
  for (let step = 0; step <= view; step += 1) {
    const frame = frames.get(step);
    if (!frame) break;
    for (const [index, rem, ex, flag] of frame.job) {
      remaining[index] = rem;
      exec[index] = ex;
      done[index] = flag;
      if (flag && finished[index] == null) finished[index] = step;
      if (!flag) finished[index] = null;
      if (ex >= 0) {
        const id = meta.ids[ex];
        const list = progress[index];
        if (!list.includes(id)) list.push(id);
      }
    }
  }
  return remaining.map((rem, i) => ({
    remaining: rem,
    exec: exec[i],
    done: done[i],
    finished: finished[i],
    progress: progress[i],
  }));
}

function statusOf(spec, dyn, step) {
  if (dyn.done) return 'done';
  if (spec.deadline_step <= step) return 'overdue';
  if (dyn.remaining < spec.work_steps || dyn.exec >= 0) return 'active';
  return 'waiting';
}

function resolveMetricsDelta(snap, view, now) {
  const laterStep = Math.min(view + AHEAD, snap.meta.steps);
  const later = snap.frames.get(laterStep);
  const alt = snap.alt && snap.alt.get(view);
  return {
    completed: vsAlt(now.jobs_completed, alt && alt.jobs_completed, false),
    overdue: vsAhead(now.jobs_due_missed, later && later.metrics.jobs_due_missed, true),
    revenue: vsAlt(now.revenue_usd, alt && alt.revenue_usd, false),
    brownout: vsAhead(now.brownout_satellite_steps, later && later.metrics.brownout_satellite_steps, true),
    reserve: vsAhead(now.below_reserve_satellite_steps, later && later.metrics.below_reserve_satellite_steps, true),
    blocked: null,
  };
}
