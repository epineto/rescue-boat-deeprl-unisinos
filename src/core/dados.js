// Carga e agregação dos dados exportados (data/*.csv|json). Módulo puro: recebe uma função `ler(caminho) → texto`.
// Loading and aggregation of the exported data. Pure module: receives a `ler(path) → text` function (fetch in the browser, fs in tests).
import { media, desvio, ic95, minimo, maximo } from './estat.js';

export const ALGOS = ['DQN', 'PPO', 'A2C', 'SARSA'];

/** Analisa CSV (separador vírgula, aspas duplas com "" para escapar, quebras de linha CRLF/LF). Devolve vetor de objetos. */
export function parseCSV(texto) {
  const linhas = [];
  let campo = '', linha = [], aspas = false;
  const fimCampo = () => { linha.push(campo); campo = ''; };
  const fimLinha = () => { fimCampo(); if (!(linha.length === 1 && linha[0] === '')) linhas.push(linha); linha = []; };
  for (let i = 0; i < texto.length; i++) {
    const c = texto[i];
    if (aspas) {
      if (c === '"') { if (texto[i + 1] === '"') { campo += '"'; i++; } else aspas = false; } else campo += c;
    } else if (c === '"') aspas = true;
    else if (c === ',') fimCampo();
    else if (c === '\n') fimLinha();
    else if (c === '\r') { /* ignora; o \n seguinte fecha a linha */ }
    else campo += c;
  }
  if (campo !== '' || linha.length) fimLinha();
  if (!linhas.length) return [];
  const [cab, ...resto] = linhas;
  return resto.map((l) => Object.fromEntries(cab.map((k, i) => [k, converte(l[i])])));
}
function converte(v) {
  if (v === undefined || v === '') return null;
  if (/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(v)) return Number(v);
  return v;
}

/** Colunas `hp` (JSON em texto) viram objeto. */
function comHp(linhas) {
  return linhas.map((l) => { if (typeof l.hp === 'string') { try { return { ...l, hp: JSON.parse(l.hp) }; } catch (_) { /* mantém texto */ } } return l; });
}

/**
 * Carrega o índice e todas as tabelas. Arquivos opcionais ausentes viram `null` (a interface mostra "ainda não disponível").
 * `ler(caminho)` deve devolver uma Promise com o texto do arquivo (relativo a `data/`) ou lançar erro se não existir.
 */
export async function carregarDados(ler) {
  const indice = JSON.parse(await ler('indice.json'));
  const presentes = new Set(indice.arquivos ?? []);
  // só busca os arquivos listados no índice: evita requisições 404 (e erros no console) para arquivos opcionais ausentes
  const tenta = async (nome, csv = true) => {
    if (!presentes.has(nome)) return null;
    try { const t = await ler(nome); return csv ? parseCSV(t) : JSON.parse(t); } catch (_) { return null; }
  };
  const [buscaRuns, buscaResumo, finalRuns, finalResumo, curvasBusca, curvasFinais, base, hipoteses, auc] = await Promise.all(
    ['busca_runs.csv', 'busca_resumo.csv', 'final_runs.csv', 'final_resumo.csv', 'curvas_busca.csv', 'curvas_finais.csv', 'linhas_de_base.csv', 'hipoteses.csv', 'auc_curvas.csv'].map((n) => tenta(n)));
  if (!finalResumo || !base) throw new Error('data/final_resumo.csv ou data/linhas_de_base.csv ausente: executar tools/exportar_para_lab.py');
  const modelos = await Promise.all(indice.modelos.map(async (m) => ({ ...m, dados: JSON.parse(await ler(m.arquivo)) })));
  const sarsa = indice.sarsa ? { ...indice.sarsa, dados: JSON.parse(await ler(indice.sarsa.arquivo)) } : null;
  const paridade = await tenta('paridade.json', false);
  const extra = await carregarExtra(tenta, indice);
  return {
    indice, modelos, sarsa, paridade, hipoteses, auc, extra,
    buscaRuns: buscaRuns && comHp(buscaRuns), buscaResumo: buscaResumo && comHp(buscaResumo), finalRuns: finalRuns && comHp(finalRuns),
    finalResumo, curvasBusca, curvasFinais, base: Object.fromEntries(base.map((l) => [l.agente, l])),
  };
}

/**
 * Experimento suplementar (orçamento maior e sanidade no CartPole): `{runs, resumo, curvas, cartpoleRuns, cartpoleResumo, info}`
 * ou `null` quando o índice não traz o bloco `extra` ou falta algum arquivo obrigatório (a aba mostra "ainda não disponível").
 */
export async function carregarExtra(tenta, indice) {
  if (!indice.extra) return null;
  const [runs, resumo, curvas, cartpoleRuns, cartpoleResumo] = await Promise.all(
    ['extra_runs.csv', 'extra_resumo.csv', 'extra_curvas.csv', 'cartpole_runs.csv', 'cartpole_resumo.csv'].map((n) => tenta(n)));
  if (!runs || !resumo || !curvas) return null;
  return { runs: comHp(runs), resumo, curvas, cartpoleRuns, cartpoleResumo, info: indice.extra };
}

/** Quantas runs aprenderam a tarefa (taxa de sucesso final ≥ `limiar`): `{aprenderam, n}` para as linhas dadas. */
export function contarAprendizes(linhas, limiar = 0.5) {
  return { aprenderam: linhas.filter((l) => l.sucesso >= limiar).length, n: linhas.length };
}

/** Resumo `{n, media, dp, ic, min, max}` de uma lista de linhas para a coluna `col`. */
export function agregarColuna(linhas, col) {
  const v = linhas.map((l) => l[col]).filter((x) => x !== null && Number.isFinite(x));
  return { n: v.length, media: media(v), dp: desvio(v), ic: ic95(v), min: minimo(v), max: maximo(v) };
}

/**
 * Agrega as runs por configuração (média, IC95% entre sementes), com a mesma regra do Python.
 * `filtro(linha)` permite excluir sementes. Devolve linhas `{algo, config, n, ret_desc, ret_desc_dp, ret_desc_ic95, sucesso, ...}`.
 */
export function agregarBusca(runs, filtro = () => true) {
  const grupos = new Map();
  for (const r of runs.filter(filtro)) {
    const k = r.algo + '\u0000' + r.config;
    if (!grupos.has(k)) grupos.set(k, []);
    grupos.get(k).push(r);
  }
  const out = [];
  for (const ls of grupos.values()) {
    const rd = agregarColuna(ls, 'ret_desc');
    out.push({ algo: ls[0].algo, config: ls[0].config, n: ls.length, ret_desc: rd.media, ret_desc_dp: rd.dp, ret_desc_ic95: rd.ic,
      ret: agregarColuna(ls, 'ret').media, sucesso: agregarColuna(ls, 'sucesso').media, passos_ep: agregarColuna(ls, 'passos_ep').media,
      toca_destrocos: agregarColuna(ls, 'toca_destrocos').media, tempo_s: agregarColuna(ls, 'tempo_s').media, hp: ls[0].hp, passos: ls[0].passos });
  }
  return out.sort((a, b) => ALGOS.indexOf(a.algo) - ALGOS.indexOf(b.algo) || a.config.localeCompare(b.config));
}

/** Curva média ± IC95% por passo: de linhas longas (algo/config/semente/passos/métrica) para `{passos[], media[], ic[], min[], max[], n[]}`. */
export function curvaMedia(linhas, metrica) {
  const por = new Map();
  for (const l of linhas) { if (l[metrica] === null) continue; if (!por.has(l.passos)) por.set(l.passos, []); por.get(l.passos).push(l[metrica]); }
  const passos = [...por.keys()].sort((a, b) => a - b);
  return {
    passos, media: passos.map((p) => media(por.get(p))), ic: passos.map((p) => (por.get(p).length > 1 ? ic95(por.get(p)) : 0)),
    min: passos.map((p) => minimo(por.get(p))), max: passos.map((p) => maximo(por.get(p))), n: passos.map((p) => por.get(p).length),
  };
}

/** Curvas individuais por semente: Map semente → {passos[], valores[]}. */
export function curvasPorSemente(linhas, metrica) {
  const m = new Map();
  for (const l of linhas) {
    if (l[metrica] === null) continue;
    if (!m.has(l.semente)) m.set(l.semente, { passos: [], valores: [] });
    m.get(l.semente).passos.push(l.passos); m.get(l.semente).valores.push(l[metrica]);
  }
  return m;
}
