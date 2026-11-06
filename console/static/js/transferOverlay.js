let depth = 0;
let root;

function ensure() {
  if (root) return root;
  root = document.getElementById('transfer-overlay');
  if (root) return root;
  root = document.createElement('div');
  root.id = 'transfer-overlay';
  root.hidden = true;
  root.setAttribute('aria-hidden', 'true');
  root.innerHTML = `
    <div class="transfer-box" role="status" aria-live="polite" aria-label="Пересчёт политики">
      <span class="transfer-mark">
        <span class="transfer-ring"></span>
        <span class="transfer-core"></span>
      </span>
      <span class="transfer-label">Пересчёт политики</span>
    </div>
  `;
  document.body.append(root);
  return root;
}

export function beginTransfer() {
  depth += 1;
  const el = ensure();
  el.hidden = false;
  el.setAttribute('aria-busy', 'true');
  document.body.classList.add('is-transfer');
}

export function endTransfer() {
  depth = Math.max(0, depth - 1);
  if (depth > 0) return;
  const el = ensure();
  el.hidden = true;
  el.removeAttribute('aria-busy');
  document.body.classList.remove('is-transfer');
}
