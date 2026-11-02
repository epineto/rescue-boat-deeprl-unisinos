import test from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync } from 'node:fs';
import pt from '../src/i18n/pt.js';
import en from '../src/i18n/en.js';
import { lerTexto, url } from './util.js';

test('dicionários PT e EN têm exatamente as mesmas chaves e nenhum valor vazio', () => {
  const a = Object.keys(pt).sort(), b = Object.keys(en).sort();
  assert.deepEqual(a.filter((k) => !(k in en)), [], 'faltam em EN');
  assert.deepEqual(b.filter((k) => !(k in pt)), [], 'faltam em PT');
  for (const k of a) { assert.ok(String(pt[k]).trim() !== '', 'PT vazio: ' + k); assert.ok(String(en[k]).trim() !== '', 'EN vazio: ' + k); }
});

test('os marcadores {var} de cada chave são os mesmos em PT e EN', () => {
  const vars = (s) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort().join(',');
  for (const k of Object.keys(pt)) assert.equal(vars(pt[k]), vars(en[k]), `marcadores de ${k}`);
});

const html = lerTexto('index.html');

test('todas as chaves data-i18n do HTML existem nos dicionários', () => {
  const chaves = new Set();
  for (const m of html.matchAll(/data-i18n(?:-html)?="([^"]+)"/g)) chaves.add(m[1]);
  for (const m of html.matchAll(/data-i18n-attr="([^"]+)"/g)) m[1].split(';').forEach((p) => chaves.add(p.split(':')[1]));
  assert.deepEqual([...chaves].filter((k) => !(k in pt)), []);
});

test('o texto padrão do HTML é igual ao dicionário PT (sem JavaScript a página mostra o mesmo texto)', () => {
  const re = /<(\w+)\b[^>]*\bdata-i18n(-html)?="([^"]+)"[^>]*>([\s\S]*?)<\/\1>/g;
  let n = 0;
  for (const m of html.matchAll(re)) {
    const esperado = m[2] ? pt[m[3]] : pt[m[3]].replace(/&/g, '&amp;').replace(/</g, '&lt;');
    assert.equal(m[4], esperado, `HTML diverge do dicionário em ${m[3]}`);
    n++;
  }
  assert.ok(n > 100);
});

test('o HTML tem as âncoras de seção exigidas e a navegação aponta para elas', () => {
  for (const id of ['intro', 'ambiente', 'politicas', 'treinar', 'resultados', 'sobre']) {
    assert.match(html, new RegExp(`<section id="${id}"`));
    assert.match(html, new RegExp(`href="#${id}"`));
  }
});

test('toda chave literal usada na interface (src/ui) existe nos dois dicionários', () => {
  const usadas = new Set();
  const prefixos = 'amb|pol|tr|res|sobre|intro|erro|prov|ag|m|estado|acao|tema|nav|cab|lang|rodape|doc';
  for (const f of readdirSync(url('src/ui/'))) {
    if (!f.endsWith('.js')) continue;
    const src = lerTexto('src/ui/' + f);
    for (const m of src.matchAll(new RegExp(`'((?:${prefixos})\\.[a-z0-9_.]*[a-z0-9_])'`, 'g'))) usadas.add(m[1]);
  }
  assert.ok(usadas.size > 80);
  const prefixosDinamicos = ['pol.cmp.c_', 'sobre.lim'];   // completados em tempo de execução (testados no teste seguinte)
  assert.deepEqual([...usadas].filter((k) => !prefixosDinamicos.includes(k)).filter((k) => !(k in pt) || !(k in en)), []);
});

test('chaves construídas dinamicamente (prefixo + sufixo) existem nos dois dicionários', () => {
  const dinamicas = {
    'amb.hud.': ['passo', 'pos', 'cel', 'bordo', 'ret', 'retd', 'rec', 'toques'],
    'estado.': ['0', '1', '2'],
    'm.': ['ret_desc', 'ret', 'sucesso', 'passos_ep', 'toca'],
    'ag.': ['aleatoria', 'tabular', 'heuristica', 'sarsa_pre', 'sarsa_user', 'aleatoria_d', 'tabular_d', 'heuristica_d', 'sarsa_pre_d', 'sarsa_user_d', 'DQN_d', 'PPO_d', 'A2C_d'],
    'intro.ag.': ['DQN', 'PPO', 'A2C', 'sarsa_pre', 'tabular', 'heuristica', 'aleatoria'],
    'intro.par.': ['passo', 'sigma', 're', 'ra', 'rd', 'vmax', 'mu', 'theta', 'sig_i', 'hor'],
    'pol.cmp.c_': ['ret_desc', 'ret', 'sucesso', 'passos', 'toca'],
    'pol.m.': ['passos', 'ret', 'entregues', 'toques'],
    'pol.vivo.': ['agente', 'ep', 'ret', 'retd', 'suc', 'passos', 'toca'],
    'res.hip.': ['h1', 'h2', 'h3'],
    'sobre.lim': ['1', '2', '3', '4', '5', '6', '7'],
    'tr.': ['concluido', 'interrompido'],
  };
  for (const [p, ss] of Object.entries(dinamicas)) for (const s of ss) { assert.ok(p + s in pt, 'PT: ' + p + s); assert.ok(p + s in en, 'EN: ' + p + s); }
});

test('voz impessoal: nenhum texto em PT usa primeira pessoa (eu, nós, meu, nosso, fiz, fizemos...)', () => {
  const proibido = /\b(eu|nós|meu|minha|meus|minhas|nosso|nossa|nossos|nossas|fiz|fizemos|escolhi|escolhemos|implementei|implementamos|criei|criamos|testei|testamos|usei|usamos|vimos|veremos|vamos)\b/i;
  for (const [k, v] of Object.entries(pt)) {
    const texto = String(v).replace(/<[^>]+>/g, ' ');
    assert.ok(!proibido.test(texto), `PT ${k}: primeira pessoa em "${texto}"`);
  }
  const proibidoEn = /\b(we|our|ours|my|mine|myself|ourselves)\b/i;
  for (const [k, v] of Object.entries(en)) {
    const texto = String(v).replace(/<[^>]+>/g, ' ');
    assert.ok(!proibidoEn.test(texto), `EN ${k}: first person in "${texto}"`);
  }
});

test('o HTML não usa primeira pessoa nos textos fixos (README e HTML seguem a voz impessoal)', () => {
  const lista = html.match(/>[^<>]{12,}</g) ?? [];
  const proibido = /\b(nós|fizemos|meu|minha|nosso|nossa)\b/i;
  for (const t of lista) assert.ok(!proibido.test(t), t);
});
