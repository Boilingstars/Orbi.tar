export function renderLoadModal(root, state, actions) {
  if (!state.loadOpen) {
    root.innerHTML = '';
    return;
  }
  const s = state.scenario;
  const m = state.metrics;
  root.innerHTML = `
    <div class="modal-backdrop" data-close>
      <div class="modal" role="dialog">
        <h2>Загрузка сценария ${s.id}</h2>
        <div class="prop-grid">
          <span class="k">Кол-во спутников</span><span class="v">${state.satellites.length}</span>
          <span class="k">Шагов в смене</span><span class="v">${s.steps} × ${s.step_s} с</span>
          <span class="k">Всего заданий</span><span class="v">${m.jobs_total}</span>
          <span class="k">Потенциальная выручка</span><span class="v">$${m.potential_revenue_usd}</span>
        </div>
        <p class="insp-h">цель</p>
        <div class="obj-toggle">
          <button type="button" data-obj="priorities" class="${state.objective === 'priorities' ? 'is-on' : ''}">Приоритеты</button>
          <button type="button" data-obj="money" class="${state.objective === 'money' ? 'is-on' : ''}">Деньги</button>
        </div>
        <div class="actions">
          <button type="button" class="btn" data-close>Закрыть</button>
          <button type="button" class="btn btn-go" data-start>Запустить с шага 0</button>
        </div>
      </div>
    </div>
  `;
  root.querySelectorAll('[data-close]').forEach((n) => n.addEventListener('click', (e) => {
    if (e.target === n) actions.closeLoad();
  }));
  root.querySelectorAll('[data-obj]').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      actions.setObjective(btn.dataset.obj);
    });
  });
  root.querySelector('[data-start]').addEventListener('click', (e) => {
    e.stopPropagation();
    actions.resetAndRun();
  });
}
