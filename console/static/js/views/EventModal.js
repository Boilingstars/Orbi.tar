export function renderEventModal(root, state, actions) {
  if (!state.eventOpen) {
    root.innerHTML = '';
    return;
  }
  root.innerHTML = `
    <div class="modal-backdrop" data-close>
      <div class="modal" role="dialog">
        <h2>Ввести событие</h2>
        <p class="empty">Форма валидации — следующий этап. Типы: add_jobs / satellite_outage / close_downlink. at_step = ${state.step}.</p>
        <div class="actions">
          <button type="button" class="btn" data-close>Закрыть</button>
        </div>
      </div>
    </div>
  `;
  root.querySelectorAll('[data-close]').forEach((n) => {
    n.addEventListener('click', (e) => { if (e.target === n) actions.closeEvent(); });
  });
}
