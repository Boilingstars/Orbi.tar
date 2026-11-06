import {
  createState, selectSatellite, selectJob, selectEvent, addOperatorEvent, cancelOperatorEvent,
  importOperatorEvents, patchOperatorEvent, addEventJob as appendEventJob,
  removeEventJob as dropEventJob, commitOperatorEvents, wireEvent,
} from './state.js';
import { renderTopBar } from './layout/TopBar.js?v=orbit1';
import { renderSatelliteTree } from './layout/SatelliteTree.js?v=palette1';
import { renderCenterStage } from './layout/CenterStage.js?v=whatif1';
import { renderInspector } from './layout/Inspector.js?v=palette1';
import { renderMetricsBar } from './layout/MetricsBar.js?v=whatif1';
import { renderLoadModal } from './views/LoadModal.js';
import { renderGoalModal } from './views/GoalModal.js?v=whatif1';
import { renderCompareView } from './views/CompareView.js';
import { applyDispatch, watchDispatch } from './dispatchView.js?v=whatif1';
import { resumeDispatch, prefetchWhatIf, clearWhatIf, dispatchSnapshot } from './api.js?v=whatif1';
import { beginTransfer, endTransfer } from './transferOverlay.js?v=resume-events';
import { loadMarkCatalog } from './timelineMarks.js';
import { AHEAD, uiToGoal, oppositeUi } from './metricsDelta.js';

const els = {
  console: document.getElementById('console'),
  top: document.getElementById('zone-top'),
  left: document.getElementById('zone-left'),
  d3: document.getElementById('zone-3d'),
  timeline: document.getElementById('zone-timeline'),
  right: document.getElementById('zone-right'),
  bottom: document.getElementById('zone-bottom'),
  modal: document.getElementById('modal-root'),
  eventModal: document.getElementById('event-modal-root'),
  compare: document.getElementById('compare-root'),
};

let state;
let playTimer = null;
let dispatchUnsub = null;
let stepping = false;

function stopPlayback() {
  if (playTimer) {
    clearTimeout(playTimer);
    playTimer = null;
  }
  if (state) state.running = false;
}

function startPlayback() {
  if (playTimer) {
    clearTimeout(playTimer);
    playTimer = null;
  }
  if (state.step >= state.scenario.steps) state.step = 0;
  state.previewStep = null;
  state.running = true;
  const tick = async () => {
    if (!state || !state.running) return;
    if (state.step >= state.scenario.steps) {
      stopPlayback();
      render();
      return;
    }
    if (!await goForward()) {
      stopPlayback();
      render();
      return;
    }
    render();
    if (state.running) playTimer = setTimeout(tick, 450);
  };
  tick();
}

const actions = {
  selectSatellite(id) {
    selectSatellite(state, id);
    render();
  },
  selectJob(id) {
    selectJob(state, id);
    render();
  },
  selectEvent(id) {
    selectEvent(state, id);
    render();
  },
  setSatFilter(v) { state.satFilter = v; render(); },
  setSatQuery(v) { state.satQuery = v; render(); },
  setJobFilter(v) { state.jobFilter = v; render(); },
  setJobQuery(v) { state.jobQuery = v; render(); },
  setTab(v) { state.inspectorTab = v; state.eventMenuOpen = false; render(); },
  setObjective(v) {
    if (v === state.objective) return;
    state.pendingObjective = v;
    state.overflowOpen = false;
    askWhatIf(state.step);
    render();
  },
  async confirmObjective() {
    if (!state.pendingObjective) return;
    const snap = dispatchSnapshot();
    if (!snap || !snap.frames.has(state.step)) return;
    const next = state.pendingObjective;
    state.objective = next;
    state.pendingObjective = null;
    state.whatIf = null;
    render();
    await switchPolicy(state.step);
    render();
  },
  cancelObjective() {
    state.pendingObjective = null;
    state.whatIf = null;
    render();
  },
  clearNotice() {
    state.notice = '';
    render();
  },
  toggleRun() {
    if (state.running) stopPlayback();
    else startPlayback();
    render();
  },
  async stepForward() {
    stopPlayback();
    await goForward();
    render();
  },
  async jumpTo(n) {
    stopPlayback();
    const max = state.scenario.steps;
    const raw = Number(n);
    const next = Number.isFinite(raw) ? Math.max(0, Math.min(max, Math.round(raw))) : state.step;
    if (next > state.step) {
      commitOperatorEvents(state, state.step);
      if (blockedAt(state.step)) {
        render();
        return;
      }
      if (!await recalculateFrom(state.step, next)) {
        render();
        return;
      }
    }
    state.step = next;
    state.previewStep = null;
    state.overflowOpen = false;
    state.eventComposeAt = state.step;
    state.eventMenuOpen = false;
    warmupWhatIf(state.step);
    render();
  },
  toggleOverflow() { state.overflowOpen = !state.overflowOpen; render(); },
  openLoad() { state.overflowOpen = false; state.loadOpen = true; render(); },
  closeLoad() { state.loadOpen = false; render(); },
  resetAndRun() {
    state.step = 0;
    state.previewStep = null;
    state.loadOpen = false;
    startPlayback();
    render();
  },
  exportResult() {
    state.overflowOpen = false;
    render();
    downloadResult();
  },
  openEvent() { state.overflowOpen = false; state.eventOpen = true; render(); },
  closeEvent() { state.eventOpen = false; render(); },
  openCompare() { state.overflowOpen = false; state.running = false; state.compareOpen = true; render(); },
  closeCompare() { state.compareOpen = false; render(); },
  toggleMetrics() {
    state.metricsOpen = !state.metricsOpen;
    els.console.classList.toggle('is-metrics-open', state.metricsOpen);
    render();
  },
  previewStep(n, mark) {
    stopPlayback();
    const max = state.scenario.steps;
    const next = Number(n);
    state.previewStep = Number.isFinite(next) ? Math.max(0, Math.min(max, Math.round(next))) : state.step;
    state.selectedMark = mark || null;
    if (mark) state.inspectorTab = 'event';
    render();
  },
  toggleEventMenu() {
    if (state.eventComposeAt !== state.step) return;
    state.eventMenuOpen = !state.eventMenuOpen;
    render();
  },
  addEvent(type) {
    addOperatorEvent(state, type);
    render();
  },
  importEvents(file) {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        importOperatorEvents(state, String(reader.result || ''));
      } catch (err) {
        state.notice = err && err.message ? err.message : 'Не удалось прочитать события';
      }
      render();
    };
    reader.readAsText(file);
  },
  patchEvent(id, field, value, index) {
    patchOperatorEvent(state, id, (ev) => applyEventField(ev, field, value, index));
    render();
  },
  addEventJob(id) {
    appendEventJob(state, id);
    render();
  },
  removeEventJob(id, index) {
    dropEventJob(state, id, index);
    render();
  },
  cancelEvent(id) {
    cancelOperatorEvent(state, id);
    render();
  },
};

window.selectSatellite = (id) => actions.selectSatellite(id);

function render() {
  if (state) applyDispatch(state);
  const active = document.activeElement;
  const restoreSatQ = active && active.matches && active.matches('[data-q]');
  const restoreJobQ = active && active.matches && active.matches('[data-job-q]');
  const restoreEvent = active && active.matches && active.matches('[data-field]');
  const restoreQ = restoreSatQ || restoreJobQ || restoreEvent;
  const caret = restoreQ && active.selectionStart != null ? active.selectionStart : null;
  const eventKey = restoreEvent
    ? [active.dataset.eid, active.dataset.field, active.dataset.index || ''].join(':')
    : '';

  renderTopBar(els.top, state, actions);
  renderSatelliteTree(els.left, state, actions);
  renderCenterStage(state, actions, els.d3, els.timeline);
  renderInspector(els.right, state, actions);
  renderMetricsBar(els.bottom, state, actions);
  renderLoadModal(els.modal, state, actions);
  renderGoalModal(els.eventModal, state, actions);
  renderCompareView(els.compare, state, actions);

  if (restoreSatQ || restoreJobQ) {
    const input = restoreJobQ
      ? els.right.querySelector('[data-job-q]')
      : els.left.querySelector('[data-q]');
    if (input) {
      input.focus();
      if (caret != null) input.setSelectionRange(caret, caret);
    }
  } else if (restoreEvent) {
    const [eid, field, index] = eventKey.split(':');
    const sel = `[data-field="${field}"][data-eid="${eid}"]${index === '' ? '' : `[data-index="${index}"]`}`;
    const input = els.right.querySelector(sel);
    if (input) {
      input.focus();
      if (caret != null && input.setSelectionRange) input.setSelectionRange(caret, caret);
    }
  }
}

export async function startConsole(snapshot) {
  if (dispatchUnsub) dispatchUnsub();
  state = createState(snapshot);
  window.__consoleState = state;
  await loadMarkCatalog();
  dispatchUnsub = watchDispatch(
    () => (state.previewStep ?? state.step),
    () => render(),
  );
  els.console.hidden = false;
  render();
  warmupWhatIf(state.step);
  window.addEventListener('resize', () => renderCenterStage(state, actions, els.d3, els.timeline));
}

function blockedAt(step) {
  return state.events.some((ev) => (
    ev.origin === 'operator' && ev.at_step === step && !ev.committed
  ));
}

function committedEvents() {
  return state.events
    .filter((ev) => ev.origin === 'operator' && ev.committed)
    .map(wireEvent);
}

function eventSig() {
  return committedEvents().map((ev) => `${ev.at_step}:${ev.id}`).join('|');
}

async function recalculateFrom(k, view) {
  const events = committedEvents();
  if (!events.some((ev) => ev.at_step === k)) return true;
  const sig = eventSig();
  if (state.resumedSig === sig) return true;
  if (stepping) return false;
  stepping = true;
  beginTransfer();
  try {
    clearWhatIf();
    const job = resumeDispatch({
      step: k,
      events,
      goal: state.objective === 'money' ? 'revenue' : 'priority',
    });
    await job.whenFrame(view);
    state.resumedSig = sig;
    warmupWhatIf(view);
    return true;
  } catch (err) {
    if (err && err.name === 'AbortError') return false;
    state.notice = err && err.message ? err.message : 'Не удалось пересчитать смену';
    return false;
  } finally {
    stepping = false;
    endTransfer();
  }
}

async function goForward() {
  if (state.step >= state.scenario.steps) return false;
  const k = state.step;
  commitOperatorEvents(state, k);
  if (blockedAt(k)) {
    stopPlayback();
    return false;
  }
  const next = k + 1;
  if (!await recalculateFrom(k, next)) return false;
  state.step = next;
  state.previewStep = null;
  state.selectedMark = null;
  state.eventComposeAt = next;
  state.eventMenuOpen = false;
  warmupWhatIf(next);
  return true;
}

function goalOf(obj) {
  return uiToGoal(obj);
}

function askWhatIf(step) {
  if (!state || !state.pendingObjective) return;
  const handle = prefetchWhatIf({
    step,
    goal: goalOf(state.pendingObjective),
    events: committedEvents(),
    horizon: AHEAD,
  });
  state.whatIf = handle.view;
  handle.done.then((view) => {
    if (!state || state.pendingObjective == null) return;
    if (state.whatIf && state.whatIf.key !== view.key) return;
    state.whatIf = view;
    render();
  });
}

function warmupWhatIf(step) {
  if (!state) return;
  const max = state.scenario.steps;
  const at = Math.max(0, Math.min(max, step));
  prefetchWhatIf({
    step: at,
    goal: goalOf(oppositeUi(state.objective)),
    events: committedEvents(),
    horizon: AHEAD,
    background: true,
  });
  if (state.running && at + 1 <= max) {
    prefetchWhatIf({
      step: at + 1,
      goal: goalOf(oppositeUi(state.objective)),
      events: committedEvents(),
      horizon: AHEAD,
      background: true,
    });
  }
}

async function switchPolicy(k) {
  if (stepping) return false;
  stepping = true;
  state.switchingGoal = true;
  beginTransfer();
  try {
    clearWhatIf();
    const job = resumeDispatch({
      step: k,
      events: committedEvents(),
      goal: state.objective === 'money' ? 'revenue' : 'priority',
    });
    await job.whenFrame(k);
    warmupWhatIf(k);
    return true;
  } catch (err) {
    if (err && err.name === 'AbortError') return false;
    state.notice = err && err.message ? err.message : 'Не удалось сменить политику';
    return false;
  } finally {
    state.switchingGoal = false;
    stepping = false;
    endTransfer();
  }
}

function applyEventField(ev, field, value, index) {
  if (field === 'satellite_ids') {
    ev.satellite_ids = String(value).split(/[\s,]+/).filter(Boolean);
    return;
  }
  if (field === 'end_step') {
    ev.end_step = Number(value);
    return;
  }
  if (!ev.jobs || index === '' || index == null) return;
  const job = ev.jobs[Number(index)];
  if (!job) return;
  if (field === 'eligible_satellites') {
    job.eligible_satellites = String(value).split(/[\s,]+/).filter(Boolean);
  } else if (field === 'value_usd') {
    job.value_usd = Number(value);
  } else if (['release_step', 'deadline_step', 'work_steps', 'priority'].includes(field)) {
    job[field] = Number(value);
  } else {
    job[field] = value;
  }
}

const RESULT_LIMIT = 120 * 1024 * 1024;

async function downloadResult() {
  try {
    const res = await fetch('/api/result');
    if (res.status === 409) {
      state.notice = 'Прогон ещё не готов';
      render();
      return;
    }
    if (res.status === 413) {
      state.notice = 'Результат больше 120 Мб';
      render();
      return;
    }
    if (!res.ok) {
      state.notice = 'Не удалось выгрузить результат';
      render();
      return;
    }
    const blob = await res.blob();
    if (blob.size > RESULT_LIMIT) {
      state.notice = 'Результат больше 120 Мб';
      render();
      return;
    }
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${state.scenario.id}.result.json`;
    a.click();
    URL.revokeObjectURL(url);
  } catch {
    state.notice = 'Не удалось выгрузить результат';
    render();
  }
}
