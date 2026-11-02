// Tema claro/escuro opcional: segue o sistema por padrão; o botão alterna e a escolha fica salva (com try/catch).
// Optional light/dark theme: follows the system by default; the button toggles and the choice is saved.
const CHAVE = 'rl-barco-tema';
const ouvintes = [];
export const onTema = (fn) => ouvintes.push(fn);

const sistemaEscuro = () => window.matchMedia?.('(prefers-color-scheme: dark)').matches ?? false;
export const temaAtual = () => document.documentElement.dataset.theme || (sistemaEscuro() ? 'dark' : 'light');

function aplicar(tema, salvar) {
  document.documentElement.dataset.theme = tema;
  if (salvar) { try { localStorage.setItem(CHAVE, tema); } catch (_) { /* sem storage */ } }
  const b = document.getElementById('tema-btn');
  if (b) b.setAttribute('aria-pressed', String(tema === 'dark'));
  ouvintes.forEach((fn) => fn(tema));
}

export function iniciarTema() {
  let salvo = null;
  try { salvo = localStorage.getItem(CHAVE); } catch (_) { /* sem storage */ }
  if (salvo === 'dark' || salvo === 'light') aplicar(salvo, false);
  else document.getElementById('tema-btn')?.setAttribute('aria-pressed', String(sistemaEscuro()));
  document.getElementById('tema-btn')?.addEventListener('click', () => aplicar(temaAtual() === 'dark' ? 'light' : 'dark', true));
  window.matchMedia?.('(prefers-color-scheme: dark)').addEventListener?.('change', () => { if (!document.documentElement.dataset.theme) ouvintes.forEach((fn) => fn(temaAtual())); });
}
