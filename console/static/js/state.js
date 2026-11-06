export function createState(snapshot) {
  return {
    scenario: snapshot.scenario,
    model: snapshot.model,
    step: snapshot.step,
    running: false,
    objective: snapshot.objective || 'priorities',
    satellites: snapshot.satellites,
    jobs: snapshot.jobs,
    events: snapshot.events,
    metrics: snapshot.metrics,
    selection: { kind: 'satellite', id: snapshot.satellites[0].id },
    satFilter: 'all',
    satQuery: '',
    jobFilter: 'all',
    jobQuery: '',
    inspectorTab: 'satellite',
    metricsOpen: false,
    previewStep: null,
    selectedMark: null,
    overflowOpen: false,
    loadOpen: false,
    eventOpen: false,
    eventMenuOpen: false,
    eventComposeAt: snapshot.step ?? 0,
    pendingObjective: null,
    whatIf: null,
    switchingGoal: false,
    notice: '',
    compareOpen: false,
  };
}

export function selectSatellite(state, id) {
  state.selection = { kind: 'satellite', id };
  state.inspectorTab = 'satellite';
}

export function selectJob(state, id) {
  state.selection = { kind: 'job', id };
  state.inspectorTab = 'job';
}

export function selectEvent(state, id) {
  state.selection = { kind: 'event', id };
  state.inspectorTab = 'event';
}

export function selectedSatellite(state) {
  const id = state.selection.kind === 'satellite' ? state.selection.id : null;
  return state.satellites.find((s) => s.id === id) || null;
}

export function operatorEvents(state) {
  return state.events
    .filter((ev) => ev.origin === 'operator')
    .sort((a, b) => a.seq - b.seq);
}

function nextEventId(events) {
  let max = 0;
  for (const ev of events) {
    const m = /^E-(\d+)$/.exec(ev.id || '');
    if (m) max = Math.max(max, Number(m[1]));
  }
  return `E-${String(max + 1).padStart(2, '0')}`;
}

function nextOperatorSeq(events) {
  let max = 0;
  for (const ev of events) {
    if (ev.origin === 'operator') max = Math.max(max, ev.seq || 0);
  }
  return max + 1;
}

export function addOperatorEvent(state, type) {
  if (state.eventComposeAt !== state.step) return;
  const id = nextEventId(state.events);
  const at_step = state.step;
  const sat = selectedSatellite(state)?.id || state.satellites[0]?.id || 'S01';
  const end = Math.min(state.scenario.steps, at_step + 24);
  const ev = {
    id,
    at_step,
    type,
    origin: 'operator',
    seq: nextOperatorSeq(state.events),
    committed: false,
    error: '',
  };
  if (type === 'add_jobs') {
    ev.jobs = [blankJob(id, at_step, sat, state.scenario.steps)];
  } else if (type === 'satellite_outage' || type === 'close_downlink') {
    ev.satellite_ids = [sat];
    ev.end_step = end;
  } else {
    return;
  }
  state.events.push(ev);
  selectEvent(state, id);
  state.eventMenuOpen = false;
}

function blankJob(eventId, at, sat, steps) {
  return {
    id: `URG-${eventId}`,
    kind: 'relay',
    release_step: at,
    deadline_step: Math.min(steps, at + 8),
    work_steps: 3,
    eligible_satellites: [sat],
    priority: 3,
    value_usd: 40,
  };
}

export function importOperatorEvents(state, text) {
  if (state.eventComposeAt !== state.step) {
    throw new Error('Событие можно ввести на текущем шаге');
  }
  const data = JSON.parse(text);
  if (data && data.schema_version && data.schema_version !== 'cosmo-B-events-1.0') {
    throw new Error('Нужна схема cosmo-B-events-1.0');
  }
  let list;
  if (Array.isArray(data)) list = data;
  else if (data && Array.isArray(data.events)) list = data.events;
  else if (data && typeof data === 'object' && data.type) list = [data];
  else throw new Error('В файле нет событий');
  if (!list.length) throw new Error('В файле нет событий');
  const known = new Set(state.satellites.map((sat) => sat.id));
  let lastId = null;
  for (const raw of list) {
    const ev = normalizeImported(raw, state, known);
    ev.origin = 'operator';
    ev.seq = nextOperatorSeq(state.events);
    ev.committed = false;
    state.events.push(ev);
    lastId = ev.id;
  }
  if (lastId) selectEvent(state, lastId);
  state.eventMenuOpen = false;
}

function normalizeImported(raw, state, known) {
  if (!raw || typeof raw !== 'object') throw new Error('Событие должно быть объектом');
  const type = raw.type;
  const at_step = state.step;
  const id = typeof raw.id === 'string' && raw.id.trim()
    ? raw.id.trim()
    : nextEventId(state.events);
  if (state.events.some((ev) => ev.id === id)) {
    throw new Error(`Событие ${id} уже есть`);
  }
  if (type === 'add_jobs') {
    const jobs = (raw.jobs || []).map((job) => ({
      id: job.id,
      kind: job.kind,
      release_step: intOr(job.release_step, at_step),
      deadline_step: intOr(job.deadline_step, Math.min(state.scenario.steps, at_step + 8)),
      work_steps: intOr(job.work_steps, 1),
      eligible_satellites: Array.isArray(job.eligible_satellites) ? job.eligible_satellites.slice() : [],
      priority: intOr(job.priority, 3),
      value_usd: Number(job.value_usd) || 0,
    }));
    if (!jobs.length) throw new Error('add_jobs нужен непустой список jobs');
    return { id, at_step, type, jobs, error: '' };
  }
  if (type === 'satellite_outage' || type === 'close_downlink') {
    const ids = (raw.satellite_ids || []).filter((sid) => known.has(sid));
    if (!ids.length) throw new Error('Нужен список satellite_ids');
    const end = intOr(raw.end_step, Math.min(state.scenario.steps, at_step + 24));
    return { id, at_step, type, satellite_ids: ids, end_step: end, error: '' };
  }
  throw new Error('Тип события: add_jobs, satellite_outage или close_downlink');
}

function intOr(value, fallback) {
  const n = Number(value);
  return Number.isInteger(n) ? n : fallback;
}

export function patchOperatorEvent(state, id, patch) {
  const ev = state.events.find((item) => item.id === id && item.origin === 'operator');
  if (!ev || ev.committed || ev.at_step !== state.step) return;
  patch(ev);
  ev.error = eventError(ev, state);
}

export function addEventJob(state, id) {
  patchOperatorEvent(state, id, (ev) => {
    if (ev.type !== 'add_jobs') return;
    const sat = selectedSatellite(state)?.id || state.satellites[0]?.id || 'S01';
    ev.jobs = ev.jobs || [];
    ev.jobs.push(blankJob(`${id}-${ev.jobs.length + 1}`, ev.at_step, sat, state.scenario.steps));
  });
}

export function removeEventJob(state, id, index) {
  patchOperatorEvent(state, id, (ev) => {
    if (ev.type !== 'add_jobs' || !ev.jobs || ev.jobs.length <= 1) return;
    ev.jobs.splice(index, 1);
  });
}

export function commitOperatorEvents(state, step) {
  for (const ev of state.events) {
    if (ev.origin === 'operator' && ev.at_step === step && !ev.committed) {
      ev.error = eventError(ev, state);
      if (!ev.error) ev.committed = true;
    }
  }
}

export function eventError(ev, state) {
  if (!ev || ev.origin !== 'operator') return '';
  const steps = state.scenario.steps;
  const known = new Set(state.satellites.map((sat) => sat.id));
  if (ev.type === 'add_jobs') {
    if (!Array.isArray(ev.jobs) || !ev.jobs.length) return 'Нужен непустой список заданий';
    for (const job of ev.jobs) {
      if (!job.id || !String(job.id).trim()) return 'У задания нет id';
      if (job.kind !== 'downlink' && job.kind !== 'relay') return 'kind: downlink или relay';
      if (!Number.isInteger(job.release_step) || job.release_step < ev.at_step) {
        return 'release_step не раньше шага события';
      }
      if (!Number.isInteger(job.deadline_step) || !(job.release_step < job.deadline_step && job.deadline_step <= steps)) {
        return 'Неверное окно deadline_step';
      }
      if (!Number.isInteger(job.work_steps) || job.work_steps <= 0
        || job.work_steps > job.deadline_step - job.release_step) {
        return 'Неверное work_steps';
      }
      if (!Array.isArray(job.eligible_satellites) || !job.eligible_satellites.length) {
        return 'Нужны eligible_satellites';
      }
      if (job.kind === 'downlink' && job.eligible_satellites.length !== 1) {
        return 'У downlink ровно один исполнитель';
      }
      if (!job.eligible_satellites.every((sid) => known.has(sid))) {
        return 'Неизвестный спутник в eligible_satellites';
      }
      if (![1, 2, 3].includes(job.priority)) return 'priority 1, 2 или 3';
      if (!Number.isFinite(job.value_usd) || job.value_usd < 0) return 'Неверное value_usd';
    }
    return '';
  }
  if (ev.type === 'satellite_outage' || ev.type === 'close_downlink') {
    const ids = ev.satellite_ids || [];
    if (!ids.length || new Set(ids).size !== ids.length) return 'Нужен список уникальных satellite_ids';
    if (!ids.every((sid) => known.has(sid))) return 'Неизвестный спутник';
    if (!Number.isInteger(ev.end_step) || !(ev.at_step < ev.end_step && ev.end_step <= steps)) {
      return 'Неверный end_step';
    }
    return '';
  }
  return 'Неизвестный тип';
}

export function wireEvent(ev) {
  const base = { id: ev.id, at_step: ev.at_step, type: ev.type };
  if (ev.type === 'add_jobs') {
    return {
      ...base,
      jobs: (ev.jobs || []).map((job) => ({
        id: job.id,
        kind: job.kind,
        release_step: job.release_step,
        deadline_step: job.deadline_step,
        work_steps: job.work_steps,
        eligible_satellites: [...(job.eligible_satellites || [])],
        priority: job.priority,
        value_usd: job.value_usd,
      })),
    };
  }
  return {
    ...base,
    satellite_ids: [...(ev.satellite_ids || [])],
    end_step: ev.end_step,
  };
}

export function cancelOperatorEvent(state, id) {
  const i = state.events.findIndex((ev) => ev.id === id && ev.origin === 'operator' && !ev.committed);
  if (i < 0) return;
  state.events.splice(i, 1);
  const rest = operatorEvents(state);
  if (state.selection.kind === 'event' && state.selection.id === id) {
    const last = rest[rest.length - 1];
    state.selection = last ? { kind: 'event', id: last.id } : { kind: 'event', id: null };
  }
  state.eventMenuOpen = false;
}
