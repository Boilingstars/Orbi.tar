import { actionLabel, actionTone, satelliteStatus, socPct, socTone, treeActionLabel, satFaded, isOverheated } from '../status.js?v=palette1';

const ACTION_RANK = { job: 0, calibrate: 1, idle: 2, out: 3 };

export function renderSatelliteTree(el, state, actions) {
  const q = state.satQuery.trim().toLowerCase();
  const rows = state.satellites.filter((sat) => {
    const st = satelliteStatus(sat, state.model);
    if (state.satFilter === 'critical' && st !== 'critical') return false;
    if (state.satFilter === 'problem' && st === 'ok') return false;
    const label = `${treeActionLabel(sat)} ${actionLabel(sat)}`.toLowerCase();
    if (q && !sat.id.toLowerCase().includes(q) && !(sat.job_id || '').toLowerCase().includes(q) && !label.includes(q)) {
      return false;
    }
    return true;
  }).sort((a, b) => {
    const fa = satFaded(a) ? 1 : 0;
    const fb = satFaded(b) ? 1 : 0;
    if (fa !== fb) return fa - fb;
    const ra = ACTION_RANK[actionTone(a)] ?? 2;
    const rb = ACTION_RANK[actionTone(b)] ?? 2;
    if (ra !== rb) return ra - rb;
    return a.id.localeCompare(b.id, 'ru');
  });

  const selectedId = state.selection.kind === 'satellite' ? state.selection.id : null;

  el.innerHTML = `
    <div class="tree-card">
      <div class="tree-toolbar">
        <div class="tree-filters">
          <button type="button" class="filter-btn ${state.satFilter === 'all' ? 'is-on' : ''}" data-f="all">Все</button>
          <button type="button" class="filter-btn ${state.satFilter === 'problem' ? 'is-on' : ''}" data-f="problem">Проблемные</button>
          <button type="button" class="filter-btn ${state.satFilter === 'critical' ? 'is-on' : ''}" data-f="critical">Критические</button>
        </div>
        <input type="search" placeholder="поиск" value="${state.satQuery}" data-q>
      </div>
      <div class="sat-list">
        <div class="sat-cols">
          <span></span>
          <span>ID</span>
          <span>заряд</span>
          <span title="температура">темп., °C</span>
          <span title="число шагов после калибровки">калибровка</span>
          <span>статус</span>
        </div>
        ${rows.map((sat) => {
          const st = satelliteStatus(sat, state.model);
          const charge = socTone(sat, state.model);
          const soc = socPct(sat);
          const action = treeActionLabel(sat);
          return `
            <button type="button" class="sat-row ${sat.id === selectedId ? 'is-selected' : ''} ${satFaded(sat) ? 'is-faded' : ''}" data-id="${sat.id}" title="${satFaded(sat) ? 'Сейчас нельзя назначить задание' : ''}">
              <span class="dot ${st}" title="${st}"></span>
              <span class="num sat-id">${sat.id}</span>
              <span class="soc-cell is-${charge}">
                <span class="soc-bar"><span style="width:${Math.max(0, Math.min(100, soc))}%"></span></span>
                <span class="num">${soc.toFixed(0)}%</span>
              </span>
              <span class="num ${isOverheated(sat) ? 'is-hot' : ''}">${sat.temp_c.toFixed(0)}°</span>
              <span class="num" title="число шагов после калибровки">${sat.calibration_age_steps}</span>
              <span class="action-cell is-${actionTone(sat)}" title="${actionLabel(sat)}">${action}</span>
            </button>`;
        }).join('')}
      </div>
    </div>
  `;

  el.querySelectorAll('[data-f]').forEach((btn) => {
    btn.addEventListener('click', () => actions.setSatFilter(btn.dataset.f));
  });
  el.querySelector('[data-q]').addEventListener('input', (e) => actions.setSatQuery(e.target.value));
  el.querySelectorAll('.sat-row').forEach((row) => {
    row.addEventListener('click', () => actions.selectSatellite(row.dataset.id));
  });
}
