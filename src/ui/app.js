// Ponto de entrada: i18n, tema, roteamento por hash (#intro … #sobre), carga dos dados e inicialização das seções.
// Entry point: i18n, theme, hash routing, data loading and section setup.
import * as I18N from './i18n.js';
import { iniciarTema } from './tema.js';
import { $, $$ } from './util.js';
import { carregarDados } from '../core/dados.js';
import { Catalogo } from './catalogo.js';
import { iniciarIntro } from './intro.js';
import { iniciarAmbiente } from './ambiente.js';
import { iniciarPoliticas } from './politicas.js';
import { iniciarTreinar } from './treinar.js';
import { iniciarResultados } from './resultados.js';
import { iniciarSobre } from './sobre.js';

const SECOES = ['intro', 'ambiente', 'politicas', 'treinar', 'resultados', 'sobre'];
const modulos = {};
let atual = null;

function rota(primeira) {
  const h = decodeURIComponent(location.hash.slice(1));
  const id = SECOES.includes(h) ? h : 'intro';
  SECOES.forEach((s) => document.getElementById(s).classList.toggle('ativa', s === id));
  $$('.nav a').forEach((a) => { if (a.getAttribute('href') === '#' + id) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
  if (!primeira) window.scrollTo({ top: 0 });
  if (atual && atual !== id) modulos[atual]?.desativar?.();
  atual = id;
  window.dispatchEvent(new Event('resize'));
  modulos[id]?.ativar?.();
}

async function main() {
  I18N.iniciar();
  iniciarTema();
  rota(true);
  window.addEventListener('hashchange', () => rota(false));
  try {
    const ler = (nome) => fetch(`data/${nome}`).then((r) => { if (!r.ok) throw new Error(`data/${nome}: HTTP ${r.status}`); return r.text(); });
    const dados = await carregarDados(ler);
    const S = { dados, catalogo: new Catalogo(dados) };
    window.__lab = S;
    const ind = dados.indice;
    const av = $('#aviso-prov');
    const avisos = () => {
      av.hidden = ind.completo !== false;
      av.textContent = I18N.t('prov.incompleto');
    };
    avisos(); I18N.onLang(avisos);
    Object.assign(modulos, { intro: iniciarIntro(S), ambiente: iniciarAmbiente(), sobre: iniciarSobre(S) });
    S.politicas = modulos.politicas = iniciarPoliticas(S);
    modulos.treinar = iniciarTreinar(S);
    modulos.resultados = iniciarResultados(S);
    rota(true);
    document.documentElement.dataset.pronto = '1';
  } catch (e) {
    const er = $('#erro-carga');
    er.hidden = false;
    er.textContent = I18N.t('erro.carga', { m: e.message });
    console.error(e);
  }
}
main();
