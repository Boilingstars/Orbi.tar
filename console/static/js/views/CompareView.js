export function renderCompareView(el, state, actions) {
  if (!state.compareOpen) {
    el.hidden = true;
    el.innerHTML = '';
    return;
  }
  el.hidden = false;
  el.innerHTML = `
    <div class="timeline-head">
      <strong>Сравнение вариантов</strong>
      <span>полноэкранный режим · вне основной сетки</span>
    </div>
    <p class="empty">Два таймлайна, diff-таблица и прогон fork() — следующий этап. Сейчас заглушка двух целей: Приоритеты vs Деньги от шага ${state.step}.</p>
    <div class="actions" style="margin-top:12px">
      <button type="button" class="btn" data-close>Вернуться к пульту</button>
    </div>
  `;
  el.querySelector('[data-close]').addEventListener('click', actions.closeCompare);
}
