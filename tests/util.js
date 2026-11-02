import { readFileSync, existsSync } from 'node:fs';
export const url = (p) => new URL('../' + p, import.meta.url);
export const lerTexto = (p) => readFileSync(url(p), 'utf8');
export const lerJson = (p) => JSON.parse(lerTexto(p));
/** Função `ler` compatível com carregarDados, lendo de data/. */
export const lerDados = async (nome) => lerTexto('data/' + nome);
export const temDados = () => existsSync(url('data/indice.json'));
