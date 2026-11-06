import { AHEAD, deltaHtml, formatUsd } from '../metricsDelta.js';

export function renderMetricsBar(el, state, actions) {
  const m = state.metrics;
  const d = state.metricsDelta || {};
  const rev = formatUsd(m.revenue_usd);
  const items = [
    {
      delta: d.completed,
      title: 'против второй политики на этом шаге',
      label: `Выполнено <span class="num">${m.critical_jobs_completed_on_time}/${m.critical_jobs_due}</span> задач высшего приоритета`,
    },
    {
      delta: d.overdue,
      title: `через ${AHEAD} шагов по текущей политике`,
      label: `Просрочено <span class="num">${m.jobs_due_missed}</span> задач`,
    },
    {
      delta: d.revenue,
      title: 'против второй политики на этом шаге',
      label: `Выручка $<span class="num">${rev}</span>`,
    },
    {
      delta: d.brownout,
      title: `через ${AHEAD} шагов по текущей политике`,
      label: `Критическое состояние заряда <span class="num">${m.brownout_satellite_steps}</span>`,
    },
    {
      delta: d.reserve,
      title: `через ${AHEAD} шагов по текущей политике`,
      label: `Заряд ниже резерва <span class="num">${m.below_reserve_satellite_steps}</span>`,
    },
    {
      delta: d.blocked,
      title: '',
      label: `Отклонённых команд <span class="num">${m.blocked_command_count}</span>`,
    },
  ];
  el.innerHTML = `
    <div class="metrics-line" data-toggle>
      ${items.map((item, i) => `
        ${i ? '<span class="sep">·</span>' : ''}
        <span class="met-item" title="${item.title}">
          ${deltaHtml(item.delta)}
          <span class="met-label">${item.label}</span>
        </span>
      `).join('')}
    </div>
    <div class="metrics-panel">
      <div class="metrics-grid">
        <div><span class="k">Заданий всего</span><span class="v">${m.jobs_total}</span></div>
        <div><span class="k">Выполнено</span><span class="v">${m.jobs_completed}</span></div>
        <div><span class="k">Потенциал</span><span class="v">$${formatUsd(m.potential_revenue_usd)}</span></div>
      </div>
    </div>
  `;
  el.querySelector('[data-toggle]').addEventListener('click', actions.toggleMetrics);
}
