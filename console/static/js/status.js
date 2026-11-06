export const TEMP_LIMIT_C = 130;

export function socPct(sat) {
  return (100 * sat.energy_wh) / sat.capacity_wh;
}

export function isOverheated(sat) {
  return Boolean(sat) && Math.abs(sat.temp_c) >= TEMP_LIMIT_C;
}

export function socTone(sat, model) {
  if (!sat || !sat.available) return 'unavailable';
  const soc = socPct(sat);
  if (soc < model.critical_soc_pct) return 'critical';
  if (soc < model.reserve_soc_pct) return 'reserve';
  return 'ok';
}

export function satelliteStatus(sat, model) {
  if (!sat || !sat.available) return 'unavailable';
  if (isOverheated(sat) || socTone(sat, model) === 'critical') return 'critical';
  if (socTone(sat, model) === 'reserve') return 'reserve';
  return 'ok';
}

export function formatClock(step) {
  const min = step * 5;
  const h = Math.floor(min / 60);
  const m = min % 60;
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
}

export function formatShiftDuration(steps, stepS = 300) {
  const hours = (steps * stepS) / 3600;
  const label = Number.isInteger(hours)
    ? String(hours)
    : String(Math.round(hours * 10) / 10).replace('.', ',');
  return `${label} ч`;
}

const ACTION_LABELS = {
  idle: 'ожидание',
  job: 'выполнение задания',
  calibrate: 'калибровка',
  cal: 'калибровка',
  out: 'недоступен',
};

const REASON_LABELS = {
  energy_reserve: 'нехватка заряда',
  not_eligible: 'недопустимый спутник',
  no_calibration: 'не калиброван',
  calibration_required: 'не калиброван',
  wrong_time: 'вне допустимого интервала времени',
  already_done: 'уже выполнено',
  no_contact: 'нет связи',
  temp_out: 'критическая температура',
  thermal_limit: 'критическая температура',
  conflict: 'конфликт',
  outage: 'недоступен',
  satellite_unavailable: 'недоступен',
  satellite_outage: 'отказ КА',
  unknown_job: 'нет задания',
  bad_action: 'ошибка команды',
};

const REASON_CODES = Object.keys(REASON_LABELS).sort((a, b) => b.length - a.length);

export function actionCodeLabel(action) {
  if (action == null || action === '') return ACTION_LABELS.idle;
  return ACTION_LABELS[action] || action;
}

export function actionLabel(sat) {
  if (!sat || !sat.available) return ACTION_LABELS.out;
  const label = actionCodeLabel(sat.action);
  if (sat.action === 'job' && sat.job_id) return `${label} ${sat.job_id}`;
  return label;
}

export function treeActionLabel(sat) {
  if (!sat || !sat.available) return ACTION_LABELS.out;
  if (sat.action === 'calibrate') return ACTION_LABELS.calibrate;
  if (sat.action === 'job') return 'выполнение';
  return ACTION_LABELS.idle;
}

export function actionTone(sat) {
  if (!sat || sat.available === false) return 'out';
  if (sat.action === 'job') return 'job';
  if (sat.action === 'calibrate') return 'calibrate';
  return 'idle';
}

export function tapeActionLabel(entry) {
  const label = actionCodeLabel(entry.action);
  if (entry.action === 'job' && entry.job_id) return `${label} ${entry.job_id}`;
  return label;
}

export function reasonLabel(reason) {
  if (!reason) return '';
  return REASON_LABELS[reason] || reason;
}

export function satFaded(sat) {
  if (!sat) return false;
  if (sat.action === 'job' || sat.action === 'calibrate') return false;
  return sat.assignable === false;
}

export function jobNeverOpen(job) {
  return job && job.first_open_step == null;
}

export function jobFaded(job) {
  if (!job) return false;
  if (job.status === 'active' || job.status === 'done') return false;
  if (job.status === 'overdue') return jobNeverOpen(job);
  return job.assignable === false;
}

export function jobListRank(job) {
  const never = jobNeverOpen(job);
  if (job.status === 'active') return 0;
  if (job.status === 'waiting' && job.assignable) return 1;
  if (job.status === 'done') return 2;
  if (job.status === 'overdue' && !never) return 3;
  if (job.status === 'overdue' && never) return 4;
  if (job.status === 'infeasible') return 5;
  return 6;
}

export function accessLabel(job) {
  if (!job) return '';
  if (job.first_open_step == null) return 'Не была в доступе';
  return `Была в доступе на шаге ${job.first_open_step}`;
}

export function idleReasonLabel(sat) {
  if (!sat || !sat.available) return 'недоступен';
  if (sat.idle_reason) return reasonLabel(sat.idle_reason);
  if (sat.action === 'calibrate') return 'калибруется';
  if (sat.action === 'job') return 'выполняет задачу';
  if (sat.action === 'idle') return 'свободен';
  return actionCodeLabel(sat.action);
}

export function localizeReasonText(text) {
  if (!text) return text;
  let s = String(text);
  for (const code of REASON_CODES) {
    if (s.includes(code)) s = s.split(code).join(REASON_LABELS[code]);
  }
  return s;
}

const XRAY_PHRASES = [
  ['в eligible_satellites', 'среди исполнителей'],
  ['eligible_satellites', 'исполнители'],
  ['work_steps', 'требуемое кол-во шагов'],
  ['release_step', 'минимальный шаг для исполнения'],
  ['deadline_step', 'максимальный шаг для исполнения'],
  ['remaining_steps', 'оставшиеся шаги'],
  ['progress_by', 'прогресс'],
  ['value_usd', 'выручка'],
  ['completed_step', 'шаг завершения'],
  ['infeasible', 'невыполнимо'],
  ['overdue', 'просрочено'],
  ['waiting', 'ожидание'],
  ['downlink', 'передача сигнала на землю'],
  ['relay', 'ретрансляция'],
  ['priority', 'приоритет'],
  ['status', 'статус'],
  ['kind', 'тип работы'],
  ['Дедлайн', 'Максимальный шаг для исполнения'],
];

const XRAY_SENTENCES = [
  [
    'Максимальный шаг для исполнения наступил до завершения требуемого кол-ва шагов',
    'Задание не успевает завершиться: нужно больше шагов, чем осталось до дедлайна.',
  ],
  [
    'Максимальный шаг для исполнения наступил до завершения требуемое кол-во шагов',
    'Задание не успевает завершиться: нужно больше шагов, чем осталось до дедлайна.',
  ],
  ['Окно контакта не покрыло оставшийся объём', 'Связь со спутником оборвалась.'],
  [
    'Невыполнимо: требуемое кол-во шагов не умещается в остаток окна',
    'Не хватает времени: работы больше, чем осталось шагов до дедлайна.',
  ],
  ['недоступен (отказ КА)', 'недоступен.'],
  ['Недостаточно данных трассировки для цепочки причин', 'недостаточно данных.'],
];

export function localizeXrayText(text) {
  let s = localizeReasonText(text);
  for (const [en, ru] of XRAY_PHRASES) {
    if (s.includes(en)) s = s.split(en).join(ru);
  }
  for (const [from, to] of XRAY_SENTENCES) {
    if (s.includes(from)) s = s.split(from).join(to);
  }
  return s;
}

export function jobKindLabel(kind) {
  return {
    relay: 'ретрансляция',
    downlink: 'передача сигнала на землю',
  }[kind] || kind;
}

export function jobPriorityLabel(priority) {
  const rank = { 1: 'низкий', 2: 'средний', 3: 'высший' }[priority];
  return rank ? `${priority} — ${rank}` : String(priority);
}

export function jobStatusLabel(status) {
  return {
    waiting: 'ожидание',
    active: 'занят',
    done: 'выполнено',
    overdue: 'просрочено',
    infeasible: 'невыполнимо',
  }[status] || status;
}

export function eventTypeLabel(type) {
  return {
    add_jobs: 'Новое срочное задание',
    satellite_outage: 'Спутник недоступен',
    close_downlink: 'Отмена сеанса',
  }[type] || type;
}
