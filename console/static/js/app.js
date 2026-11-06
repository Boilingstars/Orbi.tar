import { mountSplash } from './views/SplashScreen.js?v=orbit1';
import { beginTransfer, endTransfer } from './transferOverlay.js?v=resume-events';

const splashRoot = document.getElementById('splash-root');
const consoleEl = document.getElementById('console');
if (consoleEl) consoleEl.hidden = true;

function waitFirst(onEvent, ready) {
  return new Promise((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      off();
      resolve();
    };
    const off = onEvent((msg) => {
      if (ready(msg)) finish();
    });
    setTimeout(finish, 20000);
  });
}

mountSplash(splashRoot, {
  async onStartShift(scenario) {
    beginTransfer();
    try {
      const [{ snapshotFromScenario }, { startConsole }, { postScenario, onTrack, onDispatch }] = await Promise.all([
        import('./scenario.js'),
        import('./consoleApp.js?v=orbit1'),
        import('./api.js?v=whatif1'),
      ]);
      const listed = await postScenario(scenario);
      const tracksReady = waitFirst(onTrack, (msg) => msg.type === 'orbits' || msg.type === 'error');
      const dispatchReady = waitFirst(onDispatch, (msg) => msg.type === 'frame' || msg.type === 'error');
      splashRoot.innerHTML = '';
      splashRoot.hidden = true;
      startConsole(snapshotFromScenario(scenario, listed));
      await Promise.all([tracksReady, dispatchReady]);
    } finally {
      endTransfer();
    }
  },
});
