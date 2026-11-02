import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { BarcoContinuo } from '../src/core/env.js';
import { politicaAleatoria, politicaHeuristica, politicaTabular, valorOtimoT2, politicaOtimaT2 } from '../src/core/politicas.js';
import { avaliar, resumo, SEMENTE_AVAL } from '../src/core/avaliar.js';
import { criarMDP } from '../src/core/mdp_t2.js';

// linhas de base do Python: 500 episódios, sementes 10000+e, γ = 0,99 (resultados/linhas_de_base.csv)
const csv = readFileSync(new URL('./linhas_de_base_referencia.csv', import.meta.url), 'utf8').trim().split('\n');
const cab = csv[0].split(',');
const BASE = {};
for (const l of csv.slice(1)) {
  const c = l.split(',');
  BASE[c[0]] = Object.fromEntries(cab.map((k, i) => [k, i ? Number(c[i]) : c[i]]));
}

test('MDP do T2: 832 estados e V*(s0) = 161,93 (γ = 0,99)', () => {
  const mdp = criarMDP();
  assert.equal(mdp.nS, 832);
  assert.equal(mdp.celulas.length, 32);
  assert.equal(mdp.situacoes.length, 26);
  assert.ok(Math.abs(valorOtimoT2() - 161.93) < 0.01, String(valorOtimoT2()));
  const ref = JSON.parse(readFileSync(new URL('./reference.json', import.meta.url), 'utf8'));
  assert.ok(Math.abs(valorOtimoT2() - ref.mdp_t2.v0) < 1e-8);
  assert.deepEqual([...politicaOtimaT2()], ref.mdp_t2.pi, 'política ótima idêntica à do Python em todos os estados');
});

for (const nome of ['tabular', 'heuristica']) {
  test(`linha de base "${nome}": 500 episódios reproduzem os números do Python`, () => {
    const env = new BarcoContinuo();
    const pol = nome === 'tabular' ? politicaTabular() : politicaHeuristica(env);
    const r = resumo(avaliar(pol, 500, SEMENTE_AVAL, 0.99, {}, env)), b = BASE[nome];
    // mesma dinâmica, mesmas sementes, política determinística ⇒ coincidência (a tolerância cobre só log/cos/sin entre linguagens)
    assert.ok(Math.abs(r.ret - b.ret) < 1e-6, `ret ${r.ret} vs ${b.ret}`);
    assert.ok(Math.abs(r.ret_desc - b.ret_desc) < 1e-6, `ret_desc ${r.ret_desc} vs ${b.ret_desc}`);
    assert.equal(r.sucesso, b.sucesso);
    assert.ok(Math.abs(r.passos - b.passos) < 1e-9);
    assert.ok(Math.abs(r.toca - b.toca) < 1e-9);
    assert.ok(Math.abs(r.ret_desc_ic95 - b.ret_desc_ic95) < 1e-6);
  });
}

test('política aleatória: estatística compatível com a do Python (a sequência de ações difere, o ruído não)', () => {
  const env = new BarcoContinuo();
  const av = avaliar(politicaAleatoria(), 500, SEMENTE_AVAL, 0.99, {}, env), r = resumo(av), b = BASE.aleatoria;
  // diferença das médias dentro de 4 erros-padrão combinados (IC95 ≈ 1,96·EP em cada lado)
  const ep = (ic) => ic / 1.96;
  const lim = 4 * Math.hypot(ep(r.ret_ic95), ep(b.ret_ic95));
  assert.ok(Math.abs(r.ret - b.ret) < lim, `ret ${r.ret} vs ${b.ret} (limite ${lim})`);
  assert.ok(Math.abs(r.toca - b.toca) < 0.12);
  assert.ok(r.sucesso < 0.01);
  assert.ok(r.passos > 95);
});

test('a política aleatória é reprodutível por semente de episódio', () => {
  const a = avaliar(politicaAleatoria(), 5, 77), b = avaliar(politicaAleatoria(), 5, 77);
  assert.deepEqual(a.ret, b.ret);
});
