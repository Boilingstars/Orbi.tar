import {
  formatBytes,
  isScenarioFile,
  parseScenarioText,
  readFileAsText,
  validateScenario,
} from './splashValidate.js';

const DEMO_URL = new URL('../../../src/mocks/P01_intro.json', import.meta.url).href;

export function mountSplash(root, { onStartShift }) {
  if (!root) return;
  const ui = {
    phase: 'empty',
    fileName: '',
    fileSize: 0,
    preview: null,
    error: '',
    scenario: null,
  };

  function set(patch) {
    Object.assign(ui, patch);
    draw();
  }

  async function ingestFile(file) {
    if (!file) return;
    if (!isScenarioFile(file)) {
      set({
        phase: 'error',
        fileName: file.name || '',
        fileSize: file.size || 0,
        preview: null,
        scenario: null,
        error: 'Поддерживается только JSON сценария смены',
      });
      return;
    }
    set({
      phase: 'validating',
      fileName: file.name,
      fileSize: file.size,
      preview: null,
      scenario: null,
      error: '',
    });
    try {
      const text = await readFileAsText(file);
      const { data, preview } = parseScenarioText(text);
      set({ phase: 'valid', preview, scenario: data, error: '' });
    } catch (err) {
      set({
        phase: 'error',
        preview: null,
        scenario: null,
        error: humanError(err),
      });
    }
  }

  async function loadDemo() {
    set({
      phase: 'validating',
      fileName: 'P01_intro.json',
      fileSize: 0,
      preview: null,
      scenario: null,
      error: '',
    });
    try {
      const res = await fetch(DEMO_URL, { cache: 'no-store' });
      if (!res.ok) throw new Error('Не удалось загрузить демо-сценарий');
      const buf = await res.arrayBuffer();
      const text = new TextDecoder().decode(buf);
      const { data, preview } = parseScenarioText(text);
      set({
        phase: 'valid',
        fileSize: buf.byteLength,
        preview,
        scenario: data,
        error: '',
      });
    } catch (err) {
      set({
        phase: 'error',
        preview: null,
        scenario: null,
        error: humanError(err, 'Не удалось загрузить демо-сценарий'),
      });
    }
  }

  function draw() {
    const canStart = ui.phase === 'valid' && ui.scenario;
    const revenue = ui.preview?.potentialRevenue;
    const revenueText = revenue == null ? '—' : `$${Math.round(revenue)}`;

    root.innerHTML = `
      <div class="splash">
        <div class="splash-panel">
          <h1>Orbi.tar</h1>
          <p class="splash-lead">Автономное управление спутниковой группировкой</p>

          <div class="splash-drop" data-drop tabindex="0">
            <input id="splash-file" class="splash-file" type="file" accept="application/json,.json" tabindex="-1">
            <p>Перетащите JSON сценария смены</p>
            <label class="btn" for="splash-file">Выбрать файл</label>
          </div>

          ${ui.phase === 'validating' ? '<p class="splash-status">Проверка файла…</p>' : ''}
          ${ui.phase === 'error' && ui.error ? `<p class="splash-error">${escapeHtml(ui.error)}</p>` : ''}

          ${ui.phase === 'valid' && ui.preview ? `
            <div class="splash-preview">
              <div class="prop-grid">
                <span class="k">Имя файла</span><span class="v">${escapeHtml(ui.fileName)}</span>
                <span class="k">Размер</span><span class="v">${formatBytes(ui.fileSize)}</span>
                <span class="k">Кол-во спутников</span><span class="v">${ui.preview.satelliteCount}</span>
                <span class="k">Шагов в смене</span><span class="v">${ui.preview.steps}</span>
                <span class="k">Всего заданий</span><span class="v">${ui.preview.jobCount}</span>
                <span class="k">Потенциальная выручка</span><span class="v">${revenueText}</span>
              </div>
            </div>` : ''}

          <div class="splash-actions">
            <button type="button" class="btn" data-demo>Загрузить демо-сценарий</button>
            <button type="button" class="btn btn-go" data-start ${canStart ? '' : 'disabled'}>Запустить смену</button>
          </div>
        </div>
      </div>
    `;

    const drop = root.querySelector('[data-drop]');
    const input = root.querySelector('#splash-file');
    input.addEventListener('change', () => ingestFile(input.files && input.files[0]));
    drop.addEventListener('dragenter', (e) => { e.preventDefault(); markDrag(drop, true); });
    drop.addEventListener('dragover', (e) => { e.preventDefault(); markDrag(drop, true); });
    drop.addEventListener('dragleave', (e) => {
      if (!drop.contains(e.relatedTarget)) markDrag(drop, false);
    });
    drop.addEventListener('drop', (e) => {
      e.preventDefault();
      markDrag(drop, false);
      ingestFile(e.dataTransfer.files && e.dataTransfer.files[0]);
    });
    root.querySelector('[data-demo]').addEventListener('click', loadDemo);
    root.querySelector('[data-start]').addEventListener('click', async () => {
      if (!canStart) return;
      try {
        await onStartShift(ui.scenario);
      } catch (err) {
        set({
          phase: 'error',
          error: humanError(err, 'Сервер не принял сценарий'),
        });
      }
    });
  }

  draw();

  root.addEventListener('dragover', (e) => e.preventDefault());
  root.addEventListener('drop', (e) => e.preventDefault());
}

function markDrag(drop, on) {
  drop.classList.toggle('is-drag', on);
}

function humanError(err, fallback) {
  const msg = err && err.message ? String(err.message) : '';
  if (!msg || /typeerror|cannot read|undefined|stack|failed to fetch|networkerror/i.test(msg)) {
    return fallback || 'Файл не подходит как сценарий смены';
  }
  return msg;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export { validateScenario };
