import { actionLabel, eventTypeLabel, idleReasonLabel, jobKindLabel, jobPriorityLabel, jobStatusLabel, reasonLabel, socPct, tapeActionLabel, jobFaded, jobListRank, accessLabel, isOverheated } from '../status.js?v=palette1';
import { explainJob } from '../xray.js';
import { operatorEvents, selectedSatellite } from '../state.js';
import { seriesChart } from '../chart.js';

export function renderInspector(el, state, actions) {
  const canAdd = state.inspectorTab === 'event' && state.eventComposeAt === state.step;
  el.innerHTML = `
    <div class="insp-tabs">
      <button type="button" data-tab="satellite" class="${state.inspectorTab === 'satellite' ? 'is-on' : ''}">Спутник</button>
      <button type="button" data-tab="job" class="${state.inspectorTab === 'job' ? 'is-on' : ''}">Задания</button>
      <button type="button" data-tab="event" class="${state.inspectorTab === 'event' ? 'is-on' : ''}">События</button>
      ${state.inspectorTab === 'event' ? `
        <div class="event-add-wrap">
          <button type="button" class="btn event-add" data-add ${canAdd ? '' : 'disabled'} title="${canAdd ? 'Добавить событие' : 'Событие можно ввести после «Шаг вперёд»'}">+</button>
          <button type="button" class="btn" data-upload ${canAdd ? '' : 'disabled'}>JSON</button>
          ${state.eventMenuOpen && canAdd ? `
            <div class="event-add-menu">
              <button type="button" data-etype="add_jobs">Новое срочное задание</button>
              <button type="button" data-etype="satellite_outage">Спутник недоступен</button>
              <button type="button" data-etype="close_downlink">Отмена сеанса</button>
            </div>` : ''}
          <input id="event-json" type="file" accept="application/json,.json" hidden>
        </div>` : ''}
    </div>
    <div class="insp-body" data-body></div>
  `;
  el.querySelectorAll('[data-tab]').forEach((btn) => {
    btn.addEventListener('click', () => actions.setTab(btn.dataset.tab));
  });
  const addBtn = el.querySelector('[data-add]');
  if (addBtn && !addBtn.disabled) {
    addBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      actions.toggleEventMenu();
    });
  }
  el.querySelectorAll('[data-etype]').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      actions.addEvent(btn.dataset.etype);
    });
  });
  const file = el.querySelector('#event-json');
  const uploadBtn = el.querySelector('[data-upload]');
  if (file) {
    file.addEventListener('change', () => {
      const picked = file.files && file.files[0];
      file.value = '';
      if (picked) actions.importEvents(picked);
    });
  }
  if (uploadBtn) {
    uploadBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      file?.click();
    });
  }
  const body = el.querySelector('[data-body]');
  if (state.inspectorTab === 'satellite') renderSat(body, state);
  else if (state.inspectorTab === 'job') renderJob(body, state, actions);
  else renderEvent(body, state, actions);
}

function renderSat(el, state) {
  const sat = selectedSatellite(state) || state.satellites.find((s) => s.id === state.selection.id);
  if (state.selection.kind !== 'satellite' || !sat) {
    el.innerHTML = '<p class="empty">Выберите спутник в дереве или на 3D.</p>';
    return;
  }
  const soc = socPct(sat);
  el.innerHTML = `
    <div class="sat-props">
      <section class="prop-card">
        <h3 class="prop-card-h">Параметры</h3>
        <div class="prop-grid">
          <span class="k">Идентификатор</span><span class="v">${sat.id}</span>
          <span class="k">Ёмкость батареи</span><span class="v">${sat.capacity_wh.toFixed(1)} <span class="u">Вт·ч</span></span>
          <span class="k">Заряд</span><span class="v">${soc.toFixed(2)} <span class="u">%</span></span>
          <span class="k">Начальная температура</span><span class="v ${isOverheated(sat) ? 'is-hot' : ''}">${sat.temp_c.toFixed(2)} <span class="u">°C</span></span>
          <span class="k">Базовая мощность</span><span class="v">${sat.base_w} <span class="u">Вт</span></span>
          <span class="k">Мощность нагревателя</span><span class="v">${sat.heater_w} <span class="u">Вт</span></span>
        </div>
      </section>
      <section class="prop-card">
        <h3 class="prop-card-h">Доп. мощность</h3>
        <div class="prop-grid">
          <span class="k">Калибровка</span><span class="v">${sat.calibration_w} <span class="u">Вт</span></span>
          <span class="k">Передача на Землю</span><span class="v">${sat.downlink_w} <span class="u">Вт</span></span>
          <span class="k">Ретрансляция</span><span class="v">${sat.relay_w} <span class="u">Вт</span></span>
        </div>
      </section>
      <section class="prop-card">
        <h3 class="prop-card-h">Состояние</h3>
        <div class="prop-grid">
          <div class="prop-pair is-long">
            <span class="k">Число шагов после последней калибровки</span>
            <span class="v">${sat.calibration_age_steps}</span>
          </div>
          <span class="k">Текущее действие</span><span class="v">${actionLabel(sat)}</span>
          <span class="k">Можно взять задание</span><span class="v">${sat.assignable === false ? 'нет' : 'да'}</span>
        </div>
      </section>
      ${seriesChart(sat.soc_series, {
        title: 'Заряд, %',
        yMin: 0,
        yMax: 100,
        yTicks: [0, 25, 50, 75, 100],
        yAxis: 'заряд, %',
        xAxis: 'шаги',
        xMin: 0,
        xMax: state.scenario.steps,
        xNow: state.previewStep ?? state.step,
        refs: [
          { value: state.model.reserve_soc_pct, cls: 'is-reserve' },
          { value: state.model.critical_soc_pct, cls: 'is-critical' },
        ],
        caption: [
          `текущее состояние ${soc.toFixed(0)}% · резерв ${state.model.reserve_soc_pct}%`,
          `критическое ${state.model.critical_soc_pct}%`,
        ],
      })}
      ${seriesChart(sat.temp_series, {
        title: 'Температура, °C',
        yMin: -150,
        yMax: 150,
        yTicks: [-150, -75, 0, 75, 150],
        yAxis: 'температура, °C',
        xAxis: 'шаги',
        xMin: 0,
        xMax: state.scenario.steps,
        xNow: state.previewStep ?? state.step,
        refs: [
          { value: 130, cls: 'is-critical' },
          { value: -130, cls: 'is-critical' },
        ],
        caption: `текущее состояние ${sat.temp_c.toFixed(1)}°C · критическое ±130°C`,
      })}
      <section class="prop-card">
        <h3 class="prop-card-h">История</h3>
        ${logGrid('шаг', 'статус', sat.action_tape.map((a) => [a.step, tapeActionLabel(a)]))}
      </section>
      <section class="prop-card">
        <h3 class="prop-card-h">Отклонённые команды</h3>
        ${logGrid('шаг', 'причина', sat.rejected.map((r) => [r.step, reasonLabel(r.reason)]))}
      </section>
      <section class="prop-card">
        <h3 class="prop-card-h">Статус</h3>
        <p class="prop-note">${idleReasonLabel(sat)}</p>
      </section>
    </div>
  `;
}

function logGrid(leftH, rightH, rows) {
  if (!rows.length) return '<p class="empty">нет</p>';
  return `
    <div class="tape-grid">
      <span class="th">${leftH}</span><span class="th is-val">${rightH}</span>
      ${rows.map(([left, right]) => `<span class="k">${left}</span><span class="v">${right}</span>`).join('')}
    </div>`;
}

function renderJob(el, state, actions) {
  const q = (state.jobQuery || '').trim().toLowerCase();
  const jobs = state.jobs.filter((j) => {
    if (state.jobFilter === 'p3' && j.priority !== 3) return false;
    if (state.jobFilter === 'overdue' && j.status !== 'overdue') return false;
    if (state.jobFilter === 'done' && j.status !== 'done') return false;
    if (q && !j.id.toLowerCase().includes(q)
      && !String(j.priority).includes(q)
      && !jobStatusLabel(j.status).toLowerCase().includes(q)
      && !(j.kind || '').toLowerCase().includes(q)) return false;
    return true;
  });
  jobs.sort((a, b) => {
    const ra = jobListRank(a);
    const rb = jobListRank(b);
    if (ra !== rb) return ra - rb;
    return a.id.localeCompare(b.id, 'ru');
  });
  const selectedId = state.selection.kind === 'job' ? state.selection.id : null;
  const job = state.jobs.find((j) => j.id === selectedId) || null;
  const xray = explainJob(job);

  el.innerHTML = `
    <div class="jobs-pane">
      <section class="prop-card job-list-card">
        <div class="tree-toolbar job-toolbar">
          <div class="tree-filters">
            <button type="button" class="filter-btn ${state.jobFilter === 'all' ? 'is-on' : ''}" data-jf="all">Все</button>
            <button type="button" class="filter-btn ${state.jobFilter === 'done' ? 'is-on' : ''}" data-jf="done">Выполненные</button>
            <button type="button" class="filter-btn ${state.jobFilter === 'overdue' ? 'is-on' : ''}" data-jf="overdue">Просроченные</button>
            <button type="button" class="filter-btn ${state.jobFilter === 'p3' ? 'is-on' : ''}" data-jf="p3">Высший приоритет</button>
          </div>
          <input type="search" placeholder="поиск" value="${state.jobQuery || ''}" data-job-q>
        </div>
        <div class="job-list">
          <div class="job-cols">
            <span>ID</span>
            <span>Приоритет</span>
            <span>Статус</span>
          </div>
          ${jobs.map((j) => `
            <button type="button" class="job-row ${j.id === selectedId ? 'is-selected' : ''} ${jobFaded(j) ? 'is-faded' : ''}" data-jid="${j.id}" title="${jobFaded(j) ? (j.status === 'overdue' ? accessLabel(j) : 'Сейчас нельзя назначить') : ''}">
              <span class="num">${j.id}</span>
              <span class="num">${j.priority}</span>
              <span class="status-${j.status}">${jobStatusLabel(j.status)}</span>
            </button>`).join('')}
        </div>
      </section>
      ${job ? jobDetail(job, xray) : ''}
    </div>
  `;
  el.querySelectorAll('[data-jf]').forEach((btn) => {
    btn.addEventListener('click', () => actions.setJobFilter(btn.dataset.jf));
  });
  el.querySelector('[data-job-q]').addEventListener('input', (e) => actions.setJobQuery(e.target.value));
  el.querySelectorAll('[data-jid]').forEach((row) => {
    row.addEventListener('click', () => actions.selectJob(row.dataset.jid));
  });
  el.querySelector('.job-row.is-selected')?.scrollIntoView({ block: 'nearest' });
}

function jobDetail(job, xray) {
  const done = job.work_steps - job.remaining_steps;
  const by = job.progress_by.join(', ') || '—';
  return `
    <section class="prop-card job-detail">
      <h3 class="prop-card-h">${job.id}</h3>
      <div class="prop-grid">
        <span class="k">Тип работы</span><span class="v">${jobKindLabel(job.kind)}</span>
        <span class="k">Минимальный шаг для исполнения</span><span class="v">${job.release_step}</span>
        <span class="k">Максимальный шаг для исполнения</span><span class="v">${job.deadline_step}</span>
        <span class="k">Требуемое кол-во шагов</span><span class="v">${job.work_steps}</span>
        <span class="k">Исполнители</span><span class="v">${job.eligible_satellites.join(', ') || '—'}</span>
        <span class="k">Приоритет</span><span class="v">${jobPriorityLabel(job.priority)}</span>
        <span class="k">Выручка</span><span class="v">$${job.value_usd.toFixed(2)}</span>
        <span class="k">Прогресс</span><span class="v">${done}/${job.work_steps} шагов · ${by}</span>
        <span class="k">Статус</span><span class="v status-${job.status}">${jobStatusLabel(job.status)}</span>
        <span class="k">Была в доступе</span><span class="v">${accessLabel(job)}</span>
      </div>
    </section>
    <section class="prop-card job-xray">
      <h3 class="prop-card-h">Анализ текущего состояния</h3>
      ${xray ? `<ul class="xray">${xray.map((s) => `<li>${s}</li>`).join('')}</ul>` : '<p class="empty">задание выполняется или выполнено.</p>'}
    </section>
  `;
}

function renderEvent(el, state, actions) {
  const list = operatorEvents(state);
  const canEdit = state.eventComposeAt === state.step;
  const mark = state.selectedMark;
  const markHtml = mark ? `
    <section class="prop-card mark-card tone-${mark.tone}">
      <h3 class="prop-card-h">${escapeHtml(mark.title)}</h3>
      <p class="mark-body">${escapeHtml(mark.body || '')}</p>
    </section>` : '';
  if (!list.length) {
    const empty = canEdit
      ? `<p class="empty empty-split"><span>Текущий шаг <span class="num">${state.step}</span> ·</span> <span>Нажмите + или загрузите JSON события.</span></p>`
      : '<p class="empty empty-break">Кликните метку на таймлайне или нажмите<br>«Шаг вперёд», чтобы ввести событие.</p>';
    el.innerHTML = markHtml + empty;
    return;
  }
  el.innerHTML = markHtml + list.map((ev) => eventCard(ev, state, canEdit && !ev.committed && ev.at_step === state.step)).join('');
  el.querySelectorAll('[data-cancel]').forEach((btn) => {
    btn.addEventListener('click', () => actions.cancelEvent(btn.dataset.cancel));
  });
  el.querySelectorAll('[data-add-job]').forEach((btn) => {
    btn.addEventListener('click', () => actions.addEventJob(btn.dataset.addJob));
  });
  el.querySelectorAll('[data-del-job]').forEach((btn) => {
    btn.addEventListener('click', () => actions.removeEventJob(btn.dataset.delJob, Number(btn.dataset.index)));
  });
  el.querySelectorAll('[data-field]').forEach((input) => {
    const apply = () => {
      actions.patchEvent(input.dataset.eid, input.dataset.field, input.value, input.dataset.index);
    };
    input.addEventListener('change', apply);
    input.addEventListener('blur', apply);
  });
}

function eventCard(ev, state, editable) {
  const locked = editable ? '' : 'disabled';
  const jobs = ev.jobs || ev.payload?.jobs || [];
  const satIds = ev.satellite_ids || ev.payload?.satellite_ids || [];
  const endStep = ev.end_step ?? ev.payload?.end_step ?? '';
  const head = `
    <div class="prop-card-head">
      <h3 class="prop-card-h">Событие № ${ev.seq}</h3>
      ${editable ? `<button type="button" class="event-cancel" data-cancel="${ev.id}">Отмена</button>` : ''}
    </div>
    <div class="prop-grid">
      <span class="k">Идентификатор</span><span class="v">${escapeHtml(ev.id)}</span>
      <span class="k">Тип</span><span class="v">${eventTypeLabel(ev.type)}</span>
      <span class="k">Шаг</span><span class="v">${ev.at_step}</span>
      <span class="k">Статус</span><span class="v">${ev.committed ? 'сохранено' : 'черновик'}</span>
    </div>`;
  let body = '';
  if (ev.type === 'add_jobs') {
    body = jobs.map((job, i) => `
      <section class="event-job">
        <div class="prop-card-head">
          <h3 class="prop-card-h">Задание ${i + 1}</h3>
          ${editable && jobs.length > 1 ? `<button type="button" class="event-cancel" data-del-job="${ev.id}" data-index="${i}">Убрать</button>` : ''}
        </div>
        <div class="prop-grid event-form">
          ${field('Идентификатор', 'id', job.id, ev.id, i, locked)}
          <span class="k">Тип работы</span>
          <span class="v">${selectField('kind', job.kind, [['relay', 'ретрансляция'], ['downlink', 'передача на Землю']], ev.id, i, locked)}</span>
          ${field('Минимальный шаг', 'release_step', job.release_step, ev.id, i, locked, 'number')}
          ${field('Максимальный шаг', 'deadline_step', job.deadline_step, ev.id, i, locked, 'number')}
          ${field('Шагов работы', 'work_steps', job.work_steps, ev.id, i, locked, 'number')}
          ${field('Исполнители', 'eligible_satellites', (job.eligible_satellites || []).join(', '), ev.id, i, locked)}
          ${field('Приоритет', 'priority', job.priority, ev.id, i, locked, 'number')}
          ${field('Выручка, USD', 'value_usd', job.value_usd, ev.id, i, locked, 'number')}
        </div>
      </section>`).join('');
    if (editable) {
      body += `<button type="button" class="btn" data-add-job="${ev.id}">Добавить задание</button>`;
    }
  } else {
    body = `
      <div class="prop-grid event-form">
        ${field('Спутники', 'satellite_ids', satIds.join(', '), ev.id, '', locked)}
        ${field('Шаг окончания', 'end_step', endStep, ev.id, '', locked, 'number')}
      </div>`;
  }
  const err = ev.error ? `<p class="splash-error">${escapeHtml(ev.error)}</p>` : '';
  return `<section class="prop-card">${head}${body}${err}</section>`;
}

function field(label, name, value, eid, index, locked, type = 'text') {
  const idx = index === '' || index == null ? '' : ` data-index="${index}"`;
  return `<span class="k">${label}</span><span class="v"><input ${locked} type="${type}" data-field="${name}" data-eid="${eid}"${idx} value="${escapeHtml(String(value ?? ''))}"></span>`;
}

function selectField(name, value, options, eid, index, locked) {
  const idx = index === '' || index == null ? '' : ` data-index="${index}"`;
  return `<select ${locked} data-field="${name}" data-eid="${eid}"${idx}>${
    options.map(([code, label]) => `<option value="${code}" ${code === value ? 'selected' : ''}>${label}</option>`).join('')
  }</select>`;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
