import { formatClock } from '../status.js?v=palette1';
import { buildSignalSvg, curveAdvantagePct } from '../timelineSignal.js?v=tl-adv';
import { detectMarks } from '../timelineMarks.js';
import { dispatchSnapshot } from '../api.js?v=whatif1';
import { syncGlobe } from '../globe.js?v=whatif1';

export function renderCenterStage(state, actions, d3El, tlEl) {
  syncGlobe(d3El, state, actions);
  renderTimeline(tlEl, state, actions);
}

function renderTimeline(el, state, actions) {
  const steps = state.scenario.steps || 1;
  const now = state.step;
  const view = state.previewStep ?? state.step;
  const nowT = now / steps;
  const ballT = view / steps;
  const snap = dispatchSnapshot();
  const qs = [];
  if (snap) {
    for (let i = 0; i <= steps; i += 1) {
      const frame = snap.frames.get(i);
      qs[i] = frame && typeof frame.q === 'number' ? frame.q : null;
    }
  }
  const marks = detectMarks(snap, state);
  const advantage = curveAdvantagePct(qs, now, steps);
  const advantageText = advantage == null ? '—' : `${advantage}%`;
  el.innerHTML = `
    <div class="timeline-head">
      <span>Ожидается <span class="num">${steps}</span> шагов · Меток событий <span class="num">${marks.length}</span> · Преимущество текущей политики: <span class="num">${advantageText}</span></span>
      <span class="num">${formatClock(view)}</span>
    </div>
    <div class="timeline-track" data-track>
      ${buildSignalSvg(nowT, ballT, qs, marks, steps)}
      ${marks.map((m) => `
        <button type="button" class="tl-dot tone-${m.tone}" data-mark="${m.id}" data-step="${m.step}" style="left:${m.left * 100}%" title="${m.title}"></button>
      `).join('')}
      <div class="tl-stick" style="left:${nowT * 100}%"></div>
      <div class="tl-ball" style="left:${ballT * 100}%"></div>
    </div>
    <div class="timeline-hint">Текущий шаг <span class="num">${now}</span> · Вы навелись на <span class="num">${view}</span> шаг</div>
  `;

  const track = el.querySelector('[data-track]');
  const setFromEvent = (e) => {
    const rect = track.getBoundingClientRect();
    const t = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    actions.previewStep(Math.round(t * steps));
  };
  track.addEventListener('click', setFromEvent);
  track.querySelectorAll('.tl-dot').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      const mark = marks.find((row) => row.id === btn.dataset.mark);
      actions.previewStep(Number(btn.dataset.step), mark);
    });
  });
}
