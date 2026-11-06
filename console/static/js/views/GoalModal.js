import { dispatchSnapshot } from '../api.js?v=whatif1';
import { AHEAD, deltaHtml, formatUsd, relDelta } from '../metricsDelta.js';

const LABELS = { priorities: 'Приоритеты', money: 'Деньги' };

export function renderGoalModal(root, state, actions) {
  if (state.notice) {
    root.innerHTML = `
      <div class="modal-backdrop" data-close>
        <div class="modal" role="dialog">
          <h2>Сообщение</h2>
          <p class="empty">${escapeHtml(state.notice)}</p>
          <div class="actions">
            <button type="button" class="btn" data-close>Закрыть</button>
          </div>
        </div>
      </div>`;
    root.querySelectorAll('[data-close]').forEach((n) => {
      n.addEventListener('click', (e) => { if (e.target === n) actions.clearNotice(); });
    });
    return;
  }
  if (!state.pendingObjective) {
    root.innerHTML = '';
    return;
  }
  const snap = dispatchSnapshot();
  const k = state.step;
  const ready = snap && snap.frames && snap.frames.has(k);
  const stayStep = Math.min(k + AHEAD, snap && snap.meta ? snap.meta.steps : k + AHEAD);
  const stayLater = snap && snap.frames.get(stayStep);
  const what = state.whatIf;
  const switchLater = what && what.frames && what.frames.get(stayStep);
  const waiting = !ready || !stayLater || !what || what.status === 'loading' || !switchLater;
  const failed = what && what.status === 'error';
  const rows = waiting && !failed ? '' : effectRows(stayLater && stayLater.metrics, switchLater);
  const okDisabled = !ready || Boolean(state.switchingGoal);

  root.innerHTML = `
    <div class="modal-backdrop" data-close>
      <div class="modal modal-goal" role="dialog">
        <h2>Смена цели</h2>
        <p>Действительно ли вы хотите поменять цель смены с шага ${k}?</p>
        <p class="goal-fromto">${LABELS[state.objective] || state.objective} → ${LABELS[state.pendingObjective] || state.pendingObjective}</p>
        <p class="goal-effect">Ожидаемый эффект через ${AHEAD} шагов, если сменить сейчас:</p>
        ${failed ? `<p class="empty">${escapeHtml(what.error || 'Не удалось сравнить')}</p>` : ''}
        ${waiting && !failed ? '<p class="goal-wait">считаем…</p>' : ''}
        ${rows}
        <div class="actions">
          <button type="button" class="btn" data-close>Отмена</button>
          <button type="button" class="btn btn-go" data-ok ${okDisabled ? 'disabled' : ''}>Сменить</button>
        </div>
      </div>
    </div>`;
  root.querySelectorAll('[data-close]').forEach((n) => {
    n.addEventListener('click', (e) => { if (e.target === n) actions.cancelObjective(); });
  });
  const ok = root.querySelector('[data-ok]');
  if (ok && !ok.disabled) {
    ok.addEventListener('click', (e) => {
      e.stopPropagation();
      actions.confirmObjective();
    });
  }
}

function effectRows(stay, next) {
  if (!stay || !next) return '';
  const items = [
    {
      label: 'Выполнено P3 в срок',
      stay: `${stay.critical_jobs_completed_on_time}/${stay.critical_jobs_due}`,
      next: `${next.critical_jobs_completed_on_time}/${next.critical_jobs_due}`,
      delta: relDelta(stay.critical_jobs_completed_on_time, next.critical_jobs_completed_on_time, false),
    },
    {
      label: 'Просрочено',
      stay: String(stay.jobs_due_missed),
      next: String(next.jobs_due_missed),
      delta: relDelta(stay.jobs_due_missed, next.jobs_due_missed, true),
    },
    {
      label: 'Выручка $',
      stay: formatUsd(stay.revenue_usd),
      next: formatUsd(next.revenue_usd),
      delta: relDelta(stay.revenue_usd, next.revenue_usd, false),
    },
    {
      label: 'Критическое состояние заряда',
      stay: String(stay.brownout_satellite_steps),
      next: String(next.brownout_satellite_steps),
      delta: relDelta(stay.brownout_satellite_steps, next.brownout_satellite_steps, true),
    },
    {
      label: 'Заряд ниже резерва',
      stay: String(stay.below_reserve_satellite_steps),
      next: String(next.below_reserve_satellite_steps),
      delta: relDelta(stay.below_reserve_satellite_steps, next.below_reserve_satellite_steps, true),
    },
    {
      label: 'Отклонённых команд',
      stay: String(stay.blocked_command_count),
      next: String(next.blocked_command_count),
      delta: relDelta(stay.blocked_command_count, next.blocked_command_count, true),
    },
  ];
  return `
    <div class="goal-effect-table">
      ${items.map((item) => `
        <div class="goal-effect-row">
          <span class="goal-effect-label">${item.label}</span>
          <span class="goal-effect-num">${item.stay}</span>
          <span class="goal-effect-arrow">→</span>
          <span class="goal-effect-num">${item.next}</span>
          ${deltaHtml(item.delta)}
        </div>
      `).join('')}
    </div>`;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/"/g, '&quot;');
}
