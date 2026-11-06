import { buildSnapshot } from '../../src/mocks/snapshot.js';
import { beginTransfer, endTransfer } from './transferOverlay.js?v=resume-events';

export async function loadSession() {
  return buildSnapshot();
}

export async function postScenario(scenario) {
  beginTransfer();
  try {
    const res = await fetch('/api/scenario', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(scenario),
    });
    if (!res.ok) {
      let detail = '';
      try {
        const body = await res.json();
        detail = typeof body.detail === 'string' ? body.detail : '';
      } catch {
        detail = '';
      }
      throw new Error(detail || 'Сервер не принял сценарий');
    }
    return res.json();
  } finally {
    endTransfer();
  }
}

export async function fetchOrbits(step) {
  const res = await fetch(`/api/orbits?step=${encodeURIComponent(step)}`);
  if (!res.ok) throw new Error('Орбиты недоступны');
  return res.json();
}

let active = null;

export function trackSnapshot() {
  return active;
}

export function primeTracks() {
  if (active) active.abort.abort();
  const state = {
    abort: new AbortController(),
    frames: new Map(),
    orbits: null,
    ids: [],
    listeners: new Set(),
    failed: false,
  };
  active = state;
  readTracks(state).catch((err) => {
    if (err && err.name === 'AbortError') return;
    if (active !== state) return;
    state.failed = true;
    state.listeners.forEach((fn) => fn({ type: 'error' }));
  });
}

export function onTrack(fn) {
  if (!active) primeTracks();
  active.listeners.add(fn);
  return () => {
    if (active) active.listeners.delete(fn);
  };
}

async function readTracks(state) {
  const res = await fetch('/api/tracks', { signal: state.abort.signal });
  if (!res.ok || !res.body) throw new Error('Орбиты недоступны');
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl = buf.indexOf('\n');
    while (nl >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      nl = buf.indexOf('\n');
      if (!line || active !== state) continue;
      const msg = JSON.parse(line);
      if (msg.type === 'meta') state.ids = msg.ids;
      else if (msg.type === 'frame') state.frames.set(msg.step, msg);
      else if (msg.type === 'orbits') state.orbits = msg;
      state.listeners.forEach((fn) => fn(msg));
    }
  }
}

let dispatchActive = null;

export function dispatchSnapshot() {
  return dispatchActive;
}

export function primeDispatch() {
  if (dispatchActive) dispatchActive.abort.abort();
  const state = {
    abort: new AbortController(),
    frames: new Map(),
    alt: new Map(),
    meta: null,
    listeners: new Set(),
    failed: false,
  };
  dispatchActive = state;
  readDispatch(state, { url: '/api/dispatch' }).catch((err) => {
    if (err && err.name === 'AbortError') return;
    if (dispatchActive !== state) return;
    state.failed = true;
    state.listeners.forEach((fn) => fn({ type: 'error' }));
  });
  readAlt(state).catch((err) => {
    if (err && err.name === 'AbortError') return;
    if (dispatchActive !== state) return;
    state.listeners.forEach((fn) => fn({ type: 'error', detail: err.message }));
  });
}

export function resumeDispatch({ step, events, goal }) {
  const prev = dispatchActive;
  const abort = new AbortController();
  let state = null;
  const watchers = new Set();
  const ping = () => {
    for (const fn of [...watchers]) fn();
  };

  const done = (async () => {
    const res = await fetch('/api/dispatch/resume', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ step, events, goal }),
      signal: abort.signal,
    });
    if (!res.ok || !res.body) {
      throw new Error(await errorDetail(res, 'Диспетчер недоступен'));
    }
    if (prev) prev.abort.abort();
    const frames = new Map();
    const alt = new Map();
    if (prev) {
      for (const [at, frame] of prev.frames) {
        if (at < step) frames.set(at, frame);
      }
      for (const [at, metrics] of prev.alt || []) {
        if (at < step) alt.set(at, metrics);
      }
    }
    state = {
      abort,
      frames,
      alt,
      meta: prev?.meta ?? null,
      listeners: prev?.listeners ?? new Set(),
      failed: false,
    };
    state.listeners.add(ping);
    dispatchActive = state;
    ping();
    try {
      await Promise.all([
        readDispatch(state, { response: res }),
        readAlt(state),
      ]);
    } finally {
      state.listeners.delete(ping);
    }
  })();

  done.catch((err) => {
    if (err && err.name === 'AbortError') return;
    if (state) {
      state.failed = true;
      state.listeners.forEach((fn) => fn({ type: 'error', detail: err.message }));
    }
    ping();
  });

  return {
    abort,
    done,
    whenFrame(at) {
      return new Promise((resolve, reject) => {
        let settled = false;
        const finish = (err) => {
          if (settled) return;
          settled = true;
          watchers.delete(check);
          if (err) reject(err);
          else resolve();
        };
        const check = () => {
          if (state && state.failed) {
            finish(new Error('Не удалось пересчитать смену'));
            return;
          }
          if (state && state.frames.has(at)) finish();
        };
        watchers.add(check);
        check();
        done.then(() => {
          if (state && state.frames.has(at)) finish();
          else if (state && state.failed) finish(new Error('Не удалось пересчитать смену'));
          else finish(new Error('Прогон оборвался до нового шага'));
        }).catch((err) => {
          finish(err && err.name === 'AbortError' ? err : (err || new Error('Не удалось пересчитать смену')));
        });
      });
    },
  };
}

export function onDispatch(fn) {
  if (!dispatchActive) primeDispatch();
  dispatchActive.listeners.add(fn);
  return () => {
    if (dispatchActive) dispatchActive.listeners.delete(fn);
  };
}

async function errorDetail(res, fallback) {
  try {
    const body = await res.json();
    if (typeof body.detail === 'string' && body.detail) return body.detail;
  } catch {
    /* not JSON */
  }
  return fallback;
}

async function readDispatch(state, { url, init, response } = { url: '/api/dispatch' }) {
  const res = response || await fetch(url, { signal: state.abort.signal, ...(init || {}) });
  if (!res.ok || !res.body) throw new Error(await errorDetail(res, 'Диспетчер недоступен'));
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl = buf.indexOf('\n');
    while (nl >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      nl = buf.indexOf('\n');
      if (!line || dispatchActive !== state) continue;
      const msg = JSON.parse(line);
      if (msg.type === 'meta') state.meta = msg;
      else if (msg.type === 'frame') state.frames.set(msg.step, msg);
      state.listeners.forEach((fn) => fn(msg));
    }
  }
}

async function readAlt(state) {
  const res = await fetch('/api/dispatch/alt', { signal: state.abort.signal });
  if (!res.ok || !res.body) throw new Error(await errorDetail(res, 'Вторая политика недоступна'));
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl = buf.indexOf('\n');
    while (nl >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      nl = buf.indexOf('\n');
      if (!line || dispatchActive !== state) continue;
      const msg = JSON.parse(line);
      if (msg.type === 'alt' && msg.metrics) state.alt.set(msg.step, msg.metrics);
      state.listeners.forEach((fn) => fn(msg));
    }
  }
}

const whatifCache = new Map();
const whatifInflight = new Map();

export function clearWhatIf() {
  whatifCache.clear();
  whatifInflight.clear();
}

export function prefetchWhatIf({ step, goal, events = [], horizon = 10, background = false }) {
  const sig = (events || []).map((ev) => `${ev.at_step}:${ev.id}`).join('|');
  const key = `${step}|${goal}|${horizon}|${sig}`;
  if (whatifCache.has(key)) {
    const view = whatifCache.get(key);
    return { key, view, done: Promise.resolve(view) };
  }
  if (whatifInflight.has(key)) return whatifInflight.get(key);
  if (background && whatifInflight.size >= 2) {
    const view = {
      key, status: 'loading', step, horizon,
      stayStep: step + horizon, frames: new Map(), error: '',
    };
    return { key, view, done: Promise.resolve(view) };
  }
  const view = {
    key,
    status: 'loading',
    step,
    horizon,
    stayStep: step + horizon,
    frames: new Map(),
    error: '',
  };
  const handle = {
    key,
    view,
    done: readWhatIf(view, { step, goal, events, horizon }).then((ready) => {
      whatifInflight.delete(key);
      if (ready.status === 'ready') whatifCache.set(key, ready);
      return ready;
    }),
  };
  whatifInflight.set(key, handle);
  return handle;
}

async function readWhatIf(view, { step, goal, events, horizon }) {
  try {
    const res = await fetch('/api/dispatch/whatif', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ step, goal, events, horizon }),
    });
    if (!res.ok || !res.body) {
      view.status = 'error';
      view.error = await errorDetail(res, 'Не удалось сравнить политики');
      return view;
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl = buf.indexOf('\n');
      while (nl >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        nl = buf.indexOf('\n');
        if (!line) continue;
        const msg = JSON.parse(line);
        if (msg.type === 'whatif_meta') {
          view.stayStep = msg.stay_step;
          view.horizon = msg.horizon;
        } else if (msg.type === 'whatif' && msg.metrics) {
          view.frames.set(msg.step, msg.metrics);
          if (msg.stay_step != null) view.stayStep = msg.stay_step;
        } else if (msg.type === 'error') {
          view.status = 'error';
          view.error = msg.detail || 'Не удалось сравнить политики';
          return view;
        }
      }
    }
    if (!view.frames.has(view.stayStep) && view.frames.size) {
      const last = Math.max(...view.frames.keys());
      view.stayStep = last;
    }
    view.status = view.frames.size ? 'ready' : 'error';
    if (view.status === 'error' && !view.error) view.error = 'Нет кадров сравнения';
    return view;
  } catch (err) {
    view.status = 'error';
    view.error = err && err.message ? err.message : 'Не удалось сравнить политики';
    return view;
  }
}
