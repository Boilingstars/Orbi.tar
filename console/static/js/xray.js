import { localizeXrayText } from './status.js';

export function explainJob(job) {
  if (!job || job.status === 'done' || job.status === 'active') return null;
  if (job.xray && job.xray.length) return job.xray.map(localizeXrayText);
  return ['недостаточно данных.'];
}
