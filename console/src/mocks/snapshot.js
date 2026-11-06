/** Снимок смены для UI без бэкенда. Поля совпадают с Observation/summary. */

const STEPS = 288;
const STEP_S = 300;
const STEP = 42;
const RESERVE = 30;
const CRITICAL = 20;

function pad(n, w = 2) {
  return String(n).padStart(w, '0');
}

function series(len, start, drift, jitter) {
  const out = [];
  let v = start;
  for (let k = 0; k < len; k += 1) {
    v += drift + Math.sin(k / 7) * jitter;
    out.push(Math.round(v * 100) / 100);
  }
  return out;
}

function buildSatellites() {
  const satellites = [];
  for (let i = 1; i <= 48; i += 1) {
    const id = `S${pad(i)}`;
    const capacity_wh = i % 2 ? 140 : 120;
    const bucket = i % 12;
    let soc = 58 + (i % 25);
    let available = true;
    let action = 'idle';
    let job_id = null;
    let idle_reason = null;

    if (bucket === 0) {
      soc = 16 + (i % 3);
      idle_reason = 'energy_reserve';
    } else if (bucket === 1) {
      soc = 24 + (i % 4);
      idle_reason = 'energy_reserve';
    } else if (bucket === 2) {
      soc = 61;
      available = false;
    } else if (bucket === 3) {
      action = 'calibrate';
    } else if (bucket === 4) {
      action = 'job';
      job_id = `JOB-${pad(i, 4)}`;
    } else if (bucket === 5) {
      idle_reason = 'calibration_required';
    }

    const energy_wh = (capacity_wh * soc) / 100;
    const rejected = idle_reason
      ? [{ step: STEP - 1, requested: { action: 'job', job_id: `JOB-${pad(i, 4)}` }, reason: idle_reason }]
      : [];

    satellites.push({
      id,
      capacity_wh,
      energy_wh: Math.round(energy_wh * 1000) / 1000,
      temp_c: Math.round((12 + (i % 18) + (bucket === 0 ? 8 : 0)) * 100) / 100,
      base_w: 18,
      heater_w: 30,
      calibration_w: 20,
      downlink_w: 90,
      relay_w: 65,
      calibration_age_steps: bucket === 5 ? 51 : (i * 3) % 47,
      available,
      action,
      job_id,
      idle_reason,
      rejected,
      action_tape: [
        { step: STEP - 3, action: 'idle' },
        { step: STEP - 2, action: bucket === 3 ? 'calibrate' : 'idle' },
        { step: STEP - 1, action },
        { step: STEP, action, job_id },
      ],
      soc_series: series(STEP + 1, soc + 8, -0.12, 1.1),
      temp_series: series(STEP + 1, 18, 0.02, 0.4),
    });
  }
  return satellites;
}

function buildJobs() {
  const kinds = ['downlink', 'relay'];
  const jobs = [];
  for (let i = 1; i <= 80; i += 1) {
    const id = `JOB-${pad(i, 4)}`;
    const kind = kinds[i % 2];
    const sat = `S${pad(((i - 1) % 48) + 1)}`;
    const release = (i * 3) % 40;
    const work = 1 + (i % 4);
    const deadline = Math.min(STEPS, release + work + 8 + (i % 12));
    const bucket = i % 5;
    let status = 'waiting';
    let remaining = work;
    let completed_step = null;
    let xray = null;

    if (bucket === 0) {
      status = 'done';
      remaining = 0;
      completed_step = release + work;
    } else if (bucket === 1 && deadline <= STEP) {
      status = 'overdue';
      remaining = 1;
      xray = [
        'Задание не успевает завершиться: нужно больше шагов, чем осталось до дедлайна.',
        'Последний отказ: нехватка заряда на ' + sat,
        'Связь со спутником оборвалась.',
      ];
    } else if (bucket === 2) {
      status = 'active';
      remaining = Math.max(1, work - 1);
    } else if (bucket === 3) {
      status = 'infeasible';
      xray = [
        'Не хватает времени: работы больше, чем осталось шагов до дедлайна.',
        sat + ' недоступен.',
        'Альтернатив среди исполнителей нет',
      ];
    } else if (release > STEP) {
      status = 'waiting';
    } else {
      status = 'waiting';
    }

    jobs.push({
      id,
      kind,
      release_step: release,
      deadline_step: deadline,
      work_steps: work,
      remaining_steps: remaining,
      completed_step,
      eligible_satellites: kind === 'downlink' ? [sat] : [sat, `S${pad((i % 48) + 1)}`],
      priority: (i % 3) + 1,
      value_usd: Math.round((10 + (i % 17) * 4.17) * 100) / 100,
      status,
      progress_by: status === 'done' || status === 'active' ? [sat] : [],
      xray,
    });
  }
  return jobs;
}

function buildEvents() {
  return [
    {
      id: 'E-01',
      at_step: 24,
      type: 'add_jobs',
      payload: {
        id: 'E-01',
        at_step: 24,
        type: 'add_jobs',
        jobs: [{
          id: 'URG-P-01',
          kind: 'relay',
          release_step: 24,
          deadline_step: 32,
          work_steps: 3,
          eligible_satellites: ['S08', 'S10', 'S12'],
          priority: 3,
          value_usd: 40,
        }],
      },
      reaction: {
        accepted: true,
        changed: 'В план добавлено 1 задание приоритета 3',
        kept: 'Текущие downlink-окна S01–S07 сохранены',
        lost: 'Сдвинут relay S12 на +2 шага',
      },
    },
    {
      id: 'E-02',
      at_step: 36,
      type: 'satellite_outage',
      payload: {
        id: 'E-02',
        at_step: 36,
        type: 'satellite_outage',
        satellite_ids: ['S08', 'S10'],
        end_step: 90,
      },
      reaction: {
        accepted: true,
        changed: 'S08, S10 исключены из назначения до шага 90',
        kept: 'JOB-0008 перенесён на S12',
        lost: 'JOB-0010 просрочен, −16.67 USD',
      },
    },
    {
      id: 'E-03',
      at_step: 42,
      type: 'close_downlink',
      payload: {
        id: 'E-03',
        at_step: 42,
        type: 'close_downlink',
        satellite_ids: ['S01', 'S02', 'S03'],
        end_step: 60,
      },
      reaction: {
        accepted: true,
        changed: 'Downlink S01–S03 закрыт до шага 60',
        kept: 'Relay-задания не затронуты',
        lost: '2 шага работы JOB-0001 отложены',
      },
    },
  ];
}

export function buildSnapshot() {
  return observationFromScenario({
    schema_version: 'cosmo-B-ops-1.0',
    scenario: {
      id: 'P02_shift',
      title: 'Суточная смена: 48 аппаратов (мок)',
      steps: STEPS,
      step_s: STEP_S,
    },
    model: {
      reserve_soc_pct: RESERVE,
      critical_soc_pct: CRITICAL,
      calibration_valid_steps: 48,
      downlink_parallel_limit: 2,
    },
    step: STEP,
    objective: 'priorities',
    satellites: buildSatellites(),
    jobs: buildJobs(),
    events: buildEvents(),
    metrics: {
      brownout_satellite_steps: 3,
      below_reserve_satellite_steps: 42,
      blocked_command_count: 7,
    },
  });
}

/** Снимок пульта из выбранного JSON: те же поля UI, данные из файла. */
export function observationFromScenario(data, listed) {
  const model = {
    reserve_soc_pct: data.model?.reserve_soc_pct ?? RESERVE,
    critical_soc_pct: data.model?.critical_soc_pct ?? CRITICAL,
    calibration_valid_steps: data.model?.calibration_valid_steps ?? 48,
    downlink_parallel_limit: data.model?.downlink_parallel_limit ?? 2,
  };
  const steps = data.time?.steps ?? data.scenario?.steps;
  const stepS = data.time?.step_s ?? data.scenario?.step_s ?? STEP_S;
  const id = data.meta?.id ?? data.scenario?.id ?? 'scenario';
  const title = data.meta?.title ?? data.scenario?.title ?? id;
  const step = Number.isInteger(data.step) ? data.step : 0;
  let satellites = (data.satellites || []).map((sat) => enrichSatFromFile(sat, step, model));
  const rawJobs = Array.isArray(data.jobs) ? data.jobs : [];
  let jobs = rawJobs.map((job) => enrichJobFromFile(job, step));
  if (listed) {
    const bySat = new Map(satellites.map((sat) => [sat.id, sat]));
    const byJob = new Map(jobs.map((job) => [job.id, job]));
    satellites = listed.satellites.map((id) => bySat.get(id)).filter(Boolean);
    jobs = listed.jobs.map((id) => byJob.get(id)).filter(Boolean);
  }
  const events = (data.events || []).map(enrichEventFromFile);
  return {
    schema_version: data.schema_version || 'cosmo-B-ops-1.0',
    scenario: { id, title, steps, step_s: stepS },
    model,
    step,
    objective: data.objective || 'priorities',
    satellites,
    jobs,
    events,
    metrics: mergeMetrics(data.metrics, jobs, satellites, model, step),
  };
}

function enrichSatFromFile(sat, step, model) {
  const capacity_wh = sat.capacity_wh || 1;
  const socPct = sat.initial_soc_pct ?? (
    sat.energy_wh != null ? (100 * sat.energy_wh) / capacity_wh : 0
  );
  const energy_wh = sat.energy_wh ?? (capacity_wh * socPct) / 100;
  const temp_c = sat.temp_c ?? sat.initial_temp_c ?? 0;
  const calibration_age_steps = sat.calibration_age_steps ?? sat.initial_calibration_age_steps ?? 0;
  const available = sat.available !== false;
  let action = sat.action;
  let job_id = sat.job_id ?? null;
  let idle_reason = sat.idle_reason ?? null;
  if (!action) {
    action = 'idle';
    if (!available) idle_reason = idle_reason || 'satellite_unavailable';
  }
  return {
    ...sat,
    capacity_wh,
    energy_wh,
    temp_c,
    calibration_age_steps,
    available,
    action,
    job_id,
    idle_reason,
    rejected: sat.rejected || [],
    action_tape: sat.action_tape || [],
    soc_series: sat.soc_series || [],
    temp_series: sat.temp_series || [],
  };
}

function enrichJobFromFile(job, step) {
  const work = job.work_steps ?? 1;
  let status = job.status;
  let remaining = job.remaining_steps;
  let completed_step = job.completed_step ?? null;
  if (!status) {
    if (completed_step != null) {
      status = 'done';
      remaining = 0;
    } else if (job.deadline_step <= step) {
      status = 'overdue';
      remaining = remaining ?? 1;
    } else if ((job.release_step ?? 0) > step) {
      status = 'waiting';
      remaining = remaining ?? work;
    } else {
      status = 'waiting';
      remaining = remaining ?? work;
    }
  }
  return {
    ...job,
    kind: job.kind || 'relay',
    work_steps: work,
    remaining_steps: remaining ?? work,
    completed_step,
    eligible_satellites: job.eligible_satellites || [],
    priority: job.priority || 1,
    value_usd: typeof job.value_usd === 'number' ? job.value_usd : 0,
    status,
    progress_by: job.progress_by || [],
    xray: job.xray ?? null,
  };
}

function enrichEventFromFile(ev) {
  return {
    ...ev,
    payload: ev.payload || { id: ev.id, at_step: ev.at_step, type: ev.type },
    reaction: ev.reaction || {
      accepted: true,
      changed: '—',
      kept: '—',
      lost: '—',
    },
  };
}

function mergeMetrics(given, jobs, satellites, model, step) {
  const computed = summarize(jobs, satellites, model, step);
  return {
    ...computed,
    ...given,
    brownout_satellite_steps: computed.brownout_satellite_steps,
    below_reserve_satellite_steps: computed.below_reserve_satellite_steps,
  };
}

function summarize(jobs, satellites, model, step) {
  const due = jobs.filter((j) => j.deadline_step <= step);
  const overdue = jobs.filter((j) => j.status === 'overdue');
  const done = jobs.filter((j) => j.status === 'done');
  const criticalDue = due.filter((j) => j.priority === 3);
  const criticalDone = criticalDue.filter((j) => j.status === 'done');
  const revenue = done.reduce((s, j) => s + (j.value_usd || 0), 0);
  const potential = jobs.reduce((s, j) => s + (j.value_usd || 0), 0);
  const below = satellites.filter((sat) => {
    if (sat.available === false) return false;
    const soc = (100 * sat.energy_wh) / sat.capacity_wh;
    return soc < model.reserve_soc_pct;
  }).length;
  const critical = satellites.filter((sat) => {
    if (sat.available === false) return false;
    const soc = (100 * sat.energy_wh) / sat.capacity_wh;
    return soc < model.critical_soc_pct;
  }).length;
  const blocked = satellites.reduce((n, sat) => n + (sat.rejected?.length || 0), 0);
  return {
    critical_jobs_completed_on_time: criticalDone.length,
    critical_jobs_due: criticalDue.length,
    jobs_due_missed: overdue.length,
    revenue_usd: Math.round(revenue),
    potential_revenue_usd: Math.round(potential),
    brownout_satellite_steps: critical,
    below_reserve_satellite_steps: below,
    blocked_command_count: blocked,
    jobs_total: jobs.length,
    jobs_completed: done.length,
  };
}
