const MIN_SATS = 1;
const MAX_SATS = 48;
const MIN_STEPS = 1;
const MAX_STEPS = 288;
const ORBIT_KEYS = ['raan_deg', 'period_min', 'inclination_deg', 'altitude_km'];

export function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export function isScenarioFile(file) {
  if (!file) return false;
  const name = (file.name || '').toLowerCase();
  return name.endsWith('.json') || file.type === 'application/json';
}

export function parseScenarioText(text) {
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error('File is not JSON');
  }
  return validateScenario(data);
}

export function validateScenario(data) {
  if (data == null || typeof data !== 'object' || Array.isArray(data)) {
    throw new Error('Scenario must be a JSON object');
  }
  const satellites = data.satellites;
  if (!Array.isArray(satellites)) {
    throw new Error('Scenario has no satellite list');
  }
  if (satellites.length < MIN_SATS || satellites.length > MAX_SATS) {
    throw new Error(`Satellite count must be ${MIN_SATS}..${MAX_SATS}`);
  }
  for (const sat of satellites) {
    if (!sat || typeof sat.id !== 'string' || !sat.id) {
      throw new Error('Each satellite needs an id');
    }
    const orbit = sat.orbit || sat;
    for (const key of ORBIT_KEYS) {
      if (typeof orbit[key] !== 'number') {
        throw new Error(`Satellite ${sat.id} needs orbit.${key}`);
      }
    }
  }

  const steps = data.time?.steps ?? data.scenario?.steps;
  if (!Number.isInteger(steps) || steps < MIN_STEPS || steps > MAX_STEPS) {
    throw new Error(`Step count must be ${MIN_STEPS}..${MAX_STEPS}`);
  }

  const jobCount = Number.isInteger(data.jobs?.initial) ? data.jobs.initial : 0;
  return {
    data,
    preview: {
      satelliteCount: satellites.length,
      steps,
      jobCount,
      potentialRevenue: null,
    },
  };
}

export function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ''));
    reader.onerror = () => reject(new Error('Could not read file'));
    reader.readAsText(file);
  });
}
