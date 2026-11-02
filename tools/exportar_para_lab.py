#!/usr/bin/env python
"""Exporta os resultados do Trabalho 3 para o laboratório web: gera TODOS os arquivos de ``data/``.

Uso (execução final, depois que ``experimentos_t3.py`` terminar)::

    cd lab
    python tools/exportar_para_lab.py

Entradas (padrão: ``../resultados``, a pasta irmã de ``lab``):
    runs/busca_*.json, runs/final_*.json   uma run por arquivo (curvas, métricas finais, hiperparâmetros)
    modelos/final_<ALGO>_<config>_s<semente>[.zip]   modelos do stable-baselines3 salvos pelas runs finais (a extensão .zip
                           falta quando o nome da configuração contém ponto, por exemplo "expl0.4": comportamento do SB3)
    linhas_de_base.csv, melhores_configs.json, info.json, hipoteses.csv, auc_curvas.csv   (os dois últimos são opcionais)
    extra/*.csv, extra/info_extra.json   experimento suplementar (orçamento maior e sanidade no CartPole); opcional

Saídas (``data/``, tudo regenerado e determinístico: a mesma entrada produz os mesmos bytes):
    indice.json            índice: ambiente, protocolo, configurações, melhores, lista de modelos, contagens, versões
    busca_runs.csv         uma linha por run da busca (TODAS as configurações e sementes)
    busca_resumo.csv       média, desvio e IC95% entre sementes por configuração, posição e marca da escolhida
    final_runs.csv         uma linha por run final (melhor configuração de cada algoritmo × sementes novas)
    final_resumo.csv       resumo por algoritmo e linhas de base (tabular, heurística, aleatória)
    curvas_busca.csv, curvas_finais.csv   curvas de aprendizado em formato longo (grade nominal de passos)
    linhas_de_base.csv, hipoteses.csv, auc_curvas.csv   cópias das saídas de ``experimentos_t3.py``
    extra_runs.csv, extra_resumo.csv, extra_curvas.csv, cartpole_resumo.csv, cartpole_runs.csv   versões enxutas do experimento
                           suplementar (PPO/A2C com 2 milhões de passos; sanidade no CartPole-v1); ``indice.json`` ganha o bloco ``extra``
    modelos/<ALGO>_s<semente>.json   pesos das MLPs (DQN, PPO, A2C) para inferência em JavaScript
    modelos/sarsa_s<semente>.json    pesos esparsos do Sarsa com tile coding (retreinado de forma determinística, ver abaixo)
    paridade.json          observações fixas e saídas do PyTorch/Python (os testes ``npm test`` comparam o JS com elas)

O Sarsa não salva pesos em ``experimentos_t3.py``; por isso o script retreina, com o mesmo código, a mesma semente e o mesmo
orçamento da melhor run final, o que reproduz exatamente o resultado (conferido contra o JSON da run; veja o aviso na saída).

Opções:
    --resultados DIR        pasta de resultados (padrão ../resultados)
    --codigo DIR            pasta com barco_continuo.py (padrão reference/python; se ../codigo existir, compara os hashes)
    --saida DIR             pasta de saída (padrão data)
    --permitir-incompleto   aceita runs faltando (a saída fica marcada como incompleta em indice.json)
    --so-extra              só (re)gera os arquivos do experimento suplementar e atualiza indice.json (sem retreinar o Sarsa nem
                            tocar nos demais arquivos); o resultado é idêntico ao da exportação completa
    --sem-sarsa             não retreina o Sarsa (sem modelo pré-treinado do Sarsa)
    --semente-sarsa N       força a semente do Sarsa exportado (padrão: a de maior retorno descontado nas runs finais)
    --passos-sarsa N        (só desenvolvimento) orçamento do retreino do Sarsa quando não há run final do Sarsa; o padrão é 400000
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
RAIZ = Path(__file__).resolve().parent.parent
ALGOS = ["DQN", "PPO", "A2C", "SARSA"]
HP_SARSA_PADRAO = {"n_tiles": [8, 8, 3], "n_til": 8, "alpha": 0.3}      # usado só se não houver run final do Sarsa (desenvolvimento)


# ---------------------------------------------------------------------------------------------------------------------
def arred(x, n=6):
    """Número para CSV/JSON: ``None`` vira vazio; ``nan`` vira vazio; float arredondado."""
    if x is None:
        return ""
    if isinstance(x, float):
        if x != x:
            return ""
        return round(x, n)
    return x


def f32(x):
    """float32 → float com 9 dígitos significativos (ida e volta exata em float32)."""
    return float(f"{float(x):.9g}")


def escreve_csv(caminho, colunas, linhas):
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(colunas)
        for l in linhas:
            w.writerow([arred(l.get(c)) for c in colunas])


def escreve_json(caminho, obj, indent=None):
    Path(caminho).write_text(json.dumps(obj, ensure_ascii=False, indent=indent, separators=None if indent else (",", ":"), sort_keys=False) + "\n",
                             encoding="utf-8")


def sha(caminho):
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()[:12]


# ---------------------------------------------------------------------------------------------------------------------
def carrega_runs(res: Path):
    runs = {"busca": [], "final": []}
    for f in sorted((res / "runs").glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("erro") or "final" not in d:
            print(f"  aviso: run ignorada (com erro ou incompleta): {f.name}")
            continue
        runs[d["fase"]].append(d)
    return runs


def linha_run(r):
    f = r["final"]
    return {"algo": r["algo"], "config": r["config"], "semente": r["semente"], "passos": r["passos"], "ret": f["ret"], "ret_desc": f["ret_desc"],
            "sucesso": f["sucesso"], "passos_ep": f["passos"], "toca_destrocos": f["toca"], "tempo_s": round(r["tempo"], 1),
            "hp": json.dumps(r["hp"], ensure_ascii=False, separators=(", ", ": "))}


def curvas_longas(runs):
    """Curvas em formato longo, na grade nominal de passos (orçamento / n_pontos), como ``tabela_curvas`` de experimentos_t3.py."""
    linhas = []
    for r in runs:
        freq = r["passos"] / r["n_pontos"]
        pontos = {}
        for c in r["curva"]:
            pontos[int(round(c["passos"] / freq))] = c
        for k, c in sorted(pontos.items()):
            linhas.append({"algo": r["algo"], "config": r["config"], "semente": r["semente"], "passos": int(round(k * freq)),
                           "passos_reais": c["passos"], "ret": c["ret"], "ret_desc": c["ret_desc"], "sucesso": c["sucesso"], "passos_ep": c["passos_ep"]})
    return sorted(linhas, key=lambda l: (l["algo"], l["config"], l["semente"], l["passos"]))


def agrega(linhas, B, chaves):
    """Média, desvio e IC95% (t, entre sementes) por grupo."""
    import numpy as np
    grupos = {}
    for l in linhas:
        grupos.setdefault(tuple(l[c] for c in chaves), []).append(l)
    out = []
    for k, ls in sorted(grupos.items()):
        v = np.array([l["ret_desc"] for l in ls])
        o = dict(zip(chaves, k))
        o.update(n=len(ls), ret_desc=float(v.mean()), ret_desc_dp=float(v.std(ddof=1)) if len(ls) > 1 else float("nan"), ret_desc_ic95=B.ic95(v))
        for m in ("ret", "sucesso", "passos_ep", "toca_destrocos", "tempo_s"):
            o[m] = float(np.mean([l[m] for l in ls]))
        out.append(o)
    return out


# ---------------------------------------------------------------------------------------------------------------------
def exporta_mlp(zip_path: Path):
    """Zip do SB3 → (JSON da MLP, objeto SB3). Camadas lidas dos módulos Linear; ativação do atributo ``activation_fn``."""
    import torch
    from stable_baselines3 import A2C, DQN, PPO
    m = re.match(r"final_(DQN|PPO|A2C)_(.+)_s(\d+)(?:\.zip)?$", zip_path.name)
    if not m:
        raise ValueError(f"nome de modelo inesperado: {zip_path.name}")
    algo, config, semente = m.group(1), m.group(2), int(m.group(3))
    modelo = {"DQN": DQN, "PPO": PPO, "A2C": A2C}[algo].load(str(zip_path), device="cpu")
    pol = modelo.policy
    if algo == "DQN":
        lin = [l for l in pol.q_net.q_net if isinstance(l, torch.nn.Linear)]
        saida = "q"
    else:
        lin = [l for l in pol.mlp_extractor.policy_net if isinstance(l, torch.nn.Linear)] + [pol.action_net]
        saida = "logits"
    ativ = pol.activation_fn.__name__.lower()
    if ativ not in ("relu", "tanh"):
        raise ValueError(f"ativação não suportada no laboratório: {ativ}")
    camadas = [{"entrada": l.in_features, "saida": l.out_features, "W": [[f32(v) for v in r] for r in l.weight.detach().numpy()],
                "b": [f32(v) for v in l.bias.detach().numpy()]} for l in lin]
    return {"algo": algo, "config": config, "semente": semente, "saida": saida, "ativacao": ativ, "camadas": camadas}, modelo


def saidas_torch(modelo, algo, obs):
    """Saídas brutas da última camada (Q ou logits) e ações determinísticas do ``predict`` do SB3."""
    import torch
    t = torch.as_tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        if algo == "DQN":
            y = modelo.policy.q_net(t)
        else:
            feats = modelo.policy.extract_features(t)
            y = modelo.policy.action_net(modelo.policy.mlp_extractor.forward_actor(feats))
    acoes = [int(modelo.predict(o, deterministic=True)[0]) for o in obs]
    return y.numpy(), acoes


def observacoes_fixas(B):
    """Observações de episódios do ambiente (aleatória e heurística, sementes fixas) mais casos-limite."""
    import numpy as np
    env = B.RescueBoatContEnv()
    obs = []
    heur = B.PoliticaHeuristica(env)
    rng = np.random.default_rng(0)
    for semente, usar_heur in ((1, False), (2, True), (3, True), (4, False)):
        o, _ = env.reset(seed=semente)
        for t in range(100):
            if t % 3 == 0:
                obs.append(o.copy())
            a = heur(env, o) if usar_heur else int(rng.integers(4))
            o, _, te, tr, _ = env.step(a)
            if te or tr:
                obs.append(o.copy())
                break
    obs += [np.zeros(13, np.float32), np.ones(13, np.float32), np.full(13, 0.5, np.float32)]
    return np.array(obs, dtype=np.float32)


# ---------------------------------------------------------------------------------------------------------------------
def exporta_sarsa(B, destino: Path, runs_final, melhores, info, semente_forcada, obs_fixas, passos_padrao=400_000):
    import numpy as np
    runs_s = [r for r in runs_final if r["algo"] == "SARSA"]
    if runs_s:
        melhor = max(runs_s, key=lambda r: (r["final"]["ret_desc"], r["final"]["sucesso"])) if semente_forcada is None else \
            next(r for r in runs_s if r["semente"] == semente_forcada)
        hp, semente, passos, n_pontos, config = melhor["hp"], melhor["semente"], melhor["passos"], melhor["n_pontos"], melhor["config"]
        n_ep_final = len(melhor["final_ep"]["ret"])
        n_ep_aval = 5 if info.get("rapido") else 20          # N_EP_AVAL de experimentos_t3.py (modo rápido x completo)
    else:
        melhor, hp, semente, passos, n_pontos, config = None, HP_SARSA_PADRAO, semente_forcada or 100, passos_padrao, 25, "sarsa_8x8x3_a0.3"
        n_ep_final, n_ep_aval = 100, 20
    print(f"Sarsa: retreinando {config}, semente {semente}, {passos:,} passos (determinístico)...".replace(",", "."))
    ag = B.SarsaSemiGradiente(semente=semente, **{"gamma": 0.99, **hp})
    ag.treinar(passos, {}, freq_aval=max(1, passos // n_pontos), n_ep_aval=n_ep_aval, gamma_aval=0.99)
    av = B.avaliar(ag.politica(), n_ep_final, gamma=0.99)
    r = B.resumo(av)
    if melhor is not None:
        if abs(r["ret_desc"] - melhor["final"]["ret_desc"]) > 1e-6:
            print(f"  AVISO: o retreino não reproduziu a run (ret_desc {r['ret_desc']:.4f} vs {melhor['final']['ret_desc']:.4f}). "
                  "Conferir versões do numpy/código. O modelo exportado é o retreinado.")
        else:
            print(f"  conferido: retorno descontado {r['ret_desc']:.4f} idêntico ao da run final.")
    w = ag.w.ravel()
    nz = np.flatnonzero(w)
    val32 = w[nz].astype(np.float32)
    w32 = np.zeros_like(w)
    w32[nz] = val32.astype(np.float64)
    ag32 = B.SarsaSemiGradiente(semente=semente, **{"gamma": 0.99, **hp})
    ag32.w = w32.reshape(ag.w.shape)
    q = [[float(v) for v in ag32.q(*ag32._feat(o))] for o in obs_fixas]       # Q com os pesos já em float32: o JS deve reproduzir
    b64 = lambda a: base64.b64encode(a.astype("<" + a.dtype.str[1:]).tobytes()).decode("ascii")
    nome = f"sarsa_s{semente}"
    cfg = {"nTiles": [int(v) for v in ag.tc.n_tiles], "nTil": int(ag.tc.n_til), "alpha": float(ag.alpha * ag.tc.n_til), "gamma": float(ag.gamma),
           "eps0": float(ag.eps0), "eps1": float(ag.eps1), "semente": int(semente)}
    escreve_json(destino / "modelos" / f"{nome}.json", {
        "tipo": "sarsa", "algo": "SARSA", "id": nome, "config_nome": config, "semente": int(semente), "passos_treino": int(passos), "config": cfg,
        "n_pesos": int(w.size), "n_nao_nulos": int(nz.size),
        "indices_b64": b64(nz.astype("<i4")), "valores_b64": b64(val32), "metricas": {k: float(v) for k, v in r.items()}})
    return nome, config, int(semente), cfg, q, r


# ---------------------------------------------------------------------------------------------------------------------
LIMIAR_APRENDEU = 0.5       # uma run "aprendeu" a tarefa quando a taxa de sucesso final (200 episódios) é pelo menos esta
PASSOS_COMPARACAO = 400_000  # orçamento do experimento principal: a mesma run é lida neste ponto da curva para comparação


def le_csv(caminho):
    return list(csv.DictReader(Path(caminho).open(encoding="utf-8")))


def num(v):
    return float(v) if v not in ("", None) else float("nan")


def exporta_extra(B, res: Path, saida: Path):
    """Exporta o experimento suplementar (``res/extra``) em CSV enxutos; devolve o bloco ``extra`` do índice (ou ``None`` se ausente).

    Saídas: ``extra_runs.csv`` (uma linha por run de 2 milhões de passos, com o retorno/sucesso da mesma run aos 400 mil passos e o
    diagnóstico do treino), ``extra_resumo.csv`` (média, DP, IC95% entre sementes e nº de runs que aprenderam), ``extra_curvas.csv``,
    ``cartpole_runs.csv`` e ``cartpole_resumo.csv`` (sanidade da fábrica de agentes).
    """
    ed = res / "extra"
    if not (ed / "extra_orcamento_por_semente.csv").exists():
        return None
    runs = le_csv(ed / "extra_orcamento_por_semente.csv")
    curvas = le_csv(ed / "extra_curvas.csv")
    diag = {(d["config"], int(d["semente"])): d for d in le_csv(ed / "extra_diagnostico_treino.csv")}
    c400 = {(c["config"], int(c["semente"])): c for c in curvas if int(c["passos"]) == PASSOS_COMPARACAO}
    linhas = []
    for r in runs:
        k = (r["config"], int(r["semente"]))
        d, c = diag[k], c400[k]
        linhas.append({"algo": r["algo"], "config": r["config"], "semente": int(r["semente"]), "passos": int(r["passos"]), "ret": num(r["ret"]),
                       "ret_desc": num(r["ret_desc"]), "sucesso": num(r["sucesso"]), "passos_ep": num(r["passos_ep"]),
                       "toca_destrocos": num(r["toca_destrocos"]), "tempo_s": num(r["tempo_s"]),
                       "ret_desc_400mil": num(c["ret_desc"]), "sucesso_400mil": num(c["sucesso"]),
                       "entropia": num(d["entropia_politica_nats"]), "frac_patamar": num(d["frac_patamar_1_pessoa"]),
                       "sucesso_treino": num(d["sucesso_treino_final"]), "hp": json.dumps(json.loads(r["hp"]), ensure_ascii=False, separators=(", ", ": "))})
    linhas.sort(key=lambda l: (l["algo"], l["config"], l["semente"]))
    cols = ["algo", "config", "semente", "passos", "ret", "ret_desc", "sucesso", "passos_ep", "toca_destrocos", "tempo_s", "ret_desc_400mil",
            "sucesso_400mil", "entropia", "frac_patamar", "sucesso_treino", "hp"]
    escreve_csv(saida / "extra_runs.csv", cols, linhas)

    resumo = agrega(linhas, B, ["algo", "config"])
    for o in resumo:
        sub = [l for l in linhas if (l["algo"], l["config"]) == (o["algo"], o["config"])]
        o["passos"] = sub[0]["passos"]
        o["n_aprenderam"] = sum(l["sucesso"] >= LIMIAR_APRENDEU for l in sub)
    escreve_csv(saida / "extra_resumo.csv", ["algo", "config", "n", "passos", "n_aprenderam", "ret_desc", "ret_desc_dp", "ret_desc_ic95", "ret", "sucesso",
                                              "passos_ep", "toca_destrocos", "tempo_s"], resumo)
    oficial = {(o["algo"], o["config"]): num(o["ret_desc"]) for o in le_csv(ed / "extra_orcamento_resumo.csv")}
    dif = max(abs(oficial[(o["algo"], o["config"])] - o["ret_desc"]) for o in resumo)
    print(f"  extra: {len(linhas)} runs; conferência com extra_orcamento_resumo.csv: diferença máxima em ret_desc = {dif:.2e}" + ("" if dif < 1e-6 else "  <-- AVISO"))

    cur = [{"algo": c["algo"], "config": c["config"], "semente": int(c["semente"]), "passos": int(c["passos"]), "passos_reais": int(c["passos_reais"]),
            "ret": num(c["ret"]), "ret_desc": num(c["ret_desc"]), "sucesso": num(c["sucesso"]), "passos_ep": num(c["passos_ep"])} for c in curvas]
    cur.sort(key=lambda l: (l["algo"], l["config"], l["semente"], l["passos"]))
    escreve_csv(saida / "extra_curvas.csv", ["algo", "config", "semente", "passos", "passos_reais", "ret", "ret_desc", "sucesso", "passos_ep"], cur)

    cp = [{"algo": r["algo"], "config": r["config"], "semente": int(r["semente"]), "passos": int(r["passos"]), "ret": num(r["ret"]),
           "ret_min": num(r["ret_min"]), "frac_ep_475": num(r["frac_ep_475"]), "tempo_s": num(r["tempo_s"])} for r in le_csv(ed / "cartpole_por_semente.csv")]
    cp.sort(key=lambda l: (l["algo"], l["config"], l["semente"]))
    escreve_csv(saida / "cartpole_runs.csv", ["algo", "config", "semente", "passos", "ret", "ret_min", "frac_ep_475", "tempo_s"], cp)
    cpr = [{"algo": r["algo"], "config": r["config"], "n": int(r["n"]), "ret": num(r["ret"]), "ret_min_sem": num(r["ret_min_sem"]), "ret_max_sem": num(r["ret_max_sem"]),
            "frac_ep_475": num(r["frac_ep_475"]), "tempo_s": num(r["tempo_s"])} for r in le_csv(ed / "cartpole_resumo.csv")]
    cpr.sort(key=lambda l: (l["algo"], l["config"]))
    escreve_csv(saida / "cartpole_resumo.csv", ["algo", "config", "n", "ret", "ret_min_sem", "ret_max_sem", "frac_ep_475", "tempo_s"], cpr)
    return {"passos": linhas[0]["passos"], "passos_comparacao": PASSOS_COMPARACAO, "limiar_aprendeu": LIMIAR_APRENDEU,
            "sementes": {c: sorted(l["semente"] for l in linhas if l["config"] == c) for c in sorted({l["config"] for l in linhas})},
            "cartpole_passos": cp[0]["passos"], "cartpole_sementes": sorted({l["semente"] for l in cp}),
            "cartpole_limiar": 475, "erros": json.loads((ed / "info_extra.json").read_text()).get("erros", []) if (ed / "info_extra.json").exists() else []}


def lista_arquivos(saida: Path):
    return sorted(f.name for f in saida.glob("*.csv")) + ["paridade.json"]


# ---------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resultados", type=Path, default=RAIZ.parent / "resultados")
    ap.add_argument("--codigo", type=Path, default=RAIZ / "reference" / "python")
    ap.add_argument("--saida", type=Path, default=RAIZ / "data")
    ap.add_argument("--permitir-incompleto", action="store_true")
    ap.add_argument("--so-extra", action="store_true")
    ap.add_argument("--sem-sarsa", action="store_true")
    ap.add_argument("--semente-sarsa", type=int, default=None)
    ap.add_argument("--passos-sarsa", type=int, default=400_000)
    args = ap.parse_args()
    res, saida = args.resultados.resolve(), args.saida.resolve()

    sys.path.insert(0, str(args.codigo.resolve()))
    import barco_continuo as B
    import numpy as np
    orig = RAIZ.parent / "codigo" / "barco_continuo.py"
    if orig.exists() and sha(orig) != sha(Path(B.__file__)):
        print(f"AVISO: {Path(B.__file__)} difere de {orig} (sha {sha(Path(B.__file__))} vs {sha(orig)}). "
              f"Use --codigo {orig.parent} ou atualize reference/python/barco_continuo.py.")

    print(f"Resultados: {res}\nSaída:      {saida}")
    if args.so_extra:
        indice_p = saida / "indice.json"
        if not indice_p.exists():
            sys.exit("ERRO: --so-extra exige data/indice.json (executar antes a exportação completa)")
        for f in list(saida.glob("extra_*.csv")) + list(saida.glob("cartpole_*.csv")):
            f.unlink()
        extra = exporta_extra(B, res, saida)
        indice = json.loads(indice_p.read_text(encoding="utf-8"))
        indice["arquivos"] = lista_arquivos(saida)
        indice.pop("extra", None)
        if extra:
            indice["extra"] = extra
        escreve_json(indice_p, indice, indent=1)
        print(f"Concluído (--so-extra): extra {'exportado' if extra else 'ausente em ' + str(res / 'extra')}.")
        return
    info = json.loads((res / "info.json").read_text()) if (res / "info.json").exists() else {}
    runs = carrega_runs(res)
    rapido = bool(info.get("rapido", False))
    sem_b = info.get("sementes_busca", [0, 1] if rapido else [0, 1, 2])
    sem_f = info.get("sementes_final", [100, 101] if rapido else [100, 101, 102, 103, 104])
    n_cfg = 2 if rapido else 6
    esperado_b, esperado_f = len(ALGOS) * n_cfg * len(sem_b), len(ALGOS) * len(sem_f)
    n_b, n_f = len(runs["busca"]), len(runs["final"])
    completo = n_b == esperado_b and n_f == esperado_f
    print(f"Runs: busca {n_b}/{esperado_b}, finais {n_f}/{esperado_f} ({'completo' if completo else 'INCOMPLETO'})")
    if not completo and not args.permitir_incompleto:
        sys.exit("ERRO: execução incompleta. Aguardar o fim de experimentos_t3.py ou usar --permitir-incompleto (a interface avisa que os dados estão incompletos).")
    if not (res / "linhas_de_base.csv").exists():
        sys.exit("ERRO: linhas_de_base.csv ausente em " + str(res))

    # limpeza: só o que este script gera
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "modelos").mkdir(exist_ok=True)
    for f in list((saida / "modelos").glob("*.json")) + [saida / n for n in (
            "indice.json", "busca_runs.csv", "busca_resumo.csv", "final_runs.csv", "final_resumo.csv", "curvas_busca.csv", "curvas_finais.csv",
            "linhas_de_base.csv", "hipoteses.csv", "auc_curvas.csv", "paridade.json")] + list(saida.glob("extra_*.csv")) + list(saida.glob("cartpole_*.csv")):
        if f.exists():
            f.unlink()

    # ---- tabelas -------------------------------------------------------------------------------------------------------
    busca = sorted((linha_run(r) for r in runs["busca"]), key=lambda l: (l["algo"], l["config"], l["semente"]))
    final = sorted((linha_run(r) for r in runs["final"]), key=lambda l: (l["algo"], l["semente"]))
    cols_run = ["algo", "config", "semente", "passos", "ret", "ret_desc", "sucesso", "passos_ep", "toca_destrocos", "tempo_s", "hp"]
    escreve_csv(saida / "busca_runs.csv", cols_run, busca)
    escreve_csv(saida / "final_runs.csv", cols_run, final)

    melhores = json.loads((res / "melhores_configs.json").read_text()) if (res / "melhores_configs.json").exists() else {}
    res_busca = agrega(busca, B, ["algo", "config"])
    hp_de = {(r["algo"], r["config"]): r["hp"] for r in runs["busca"] + runs["final"]}
    for a in ALGOS:                                        # critério declarado: maior retorno descontado médio; desempate pelo sucesso
        sub = sorted((l for l in res_busca if l["algo"] == a), key=lambda l: (-l["ret_desc"], -l["sucesso"], l["config"]))
        if a not in melhores and sub:
            melhores[a] = {"config": sub[0]["config"], "hp": hp_de[(a, sub[0]["config"])], "derivada": True}
        for pos, l in enumerate(sub, 1):
            l["posicao"] = pos
    for l in res_busca:
        l["escolhida"] = int(melhores.get(l["algo"], {}).get("config") == l["config"])
        l["hp"] = json.dumps(hp_de[(l["algo"], l["config"])], ensure_ascii=False, separators=(", ", ": "))
    escreve_csv(saida / "busca_resumo.csv", ["algo", "config", "n", "ret_desc", "ret_desc_dp", "ret_desc_ic95", "ret", "sucesso", "passos_ep",
                                             "toca_destrocos", "tempo_s", "posicao", "escolhida", "hp"], res_busca)

    base_linhas = list(csv.DictReader((res / "linhas_de_base.csv").open(encoding="utf-8")))
    base = {l["agente"]: {k: float(v) for k, v in l.items() if k != "agente"} for l in base_linhas}
    shutil.copyfile(res / "linhas_de_base.csv", saida / "linhas_de_base.csv")
    res_final = []
    for a in ALGOS:
        ls = [l for l in final if l["algo"] == a]
        if not ls:
            continue
        o = {"agente": a, "config": ls[0]["config"], "n_sementes": len(ls), "passos": ls[0]["passos"]}
        for m in ("ret_desc", "ret", "sucesso", "passos_ep", "toca_destrocos"):
            v = np.array([l[m] for l in ls])
            o[m] = float(v.mean())
            o[m + "_dp"] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
            o[m + "_ic95"] = B.ic95(v)
        res_final.append(o)
    for nome in ("tabular", "heuristica", "aleatoria"):
        b = base[nome]
        res_final.append({"agente": nome, "config": "-", "n_sementes": 0, "passos": 0, "ret_desc": b["ret_desc"], "ret_desc_ic95": b["ret_desc_ic95"],
                          "ret": b["ret"], "ret_ic95": b["ret_ic95"], "sucesso": b["sucesso"], "sucesso_ic95": b["sucesso_ic95"],
                          "passos_ep": b["passos"], "passos_ep_ic95": b["passos_ic95"], "toca_destrocos": b["toca"], "toca_destrocos_ic95": b["toca_ic95"]})
    cols_f = ["agente", "config", "n_sementes", "passos"] + [m + s for m in ("ret_desc", "ret", "sucesso", "passos_ep", "toca_destrocos") for s in ("", "_dp", "_ic95")]
    escreve_csv(saida / "final_resumo.csv", cols_f, res_final)
    escreve_csv(saida / "curvas_busca.csv", ["algo", "config", "semente", "passos", "passos_reais", "ret", "ret_desc", "sucesso", "passos_ep"], curvas_longas(runs["busca"]))
    escreve_csv(saida / "curvas_finais.csv", ["algo", "config", "semente", "passos", "passos_reais", "ret", "ret_desc", "sucesso", "passos_ep"], curvas_longas(runs["final"]))
    for nome in ("hipoteses.csv", "auc_curvas.csv"):
        if (res / nome).exists():
            shutil.copyfile(res / nome, saida / nome)

    extra = exporta_extra(B, res, saida)

    # conferência com os CSVs oficiais de experimentos_t3.py, quando existirem
    oficial = res / "busca_resumo.csv"
    if oficial.exists():
        off = {(l["algo"], l["config"]): float(l["ret_desc"]) for l in csv.DictReader(oficial.open(encoding="utf-8"))}
        dif = max((abs(off[(l["algo"], l["config"])] - l["ret_desc"]) for l in res_busca if (l["algo"], l["config"]) in off), default=0)
        print(f"  conferência com busca_resumo.csv oficial: diferença máxima em ret_desc = {dif:.2e}" + ("" if dif < 1e-6 else "  <-- AVISO"))

    # ---- modelos MLP e paridade ----------------------------------------------------------------------------------------
    obs_fixas = observacoes_fixas(B)
    paridade = {"observacoes": [[f32(v) for v in o] for o in obs_fixas], "modelos": {}}
    modelos_idx = []
    nomes_run = {(r["algo"], r["semente"]): r for r in runs["final"]}
    # o SB3 só acrescenta ".zip" quando o caminho não tem extensão; nomes como "..._expl0.4_s100" ficam SEM ".zip" (o load aceita os dois)
    for zp in sorted(f for f in (res / "modelos").glob("final_*") if f.is_file() and f.suffix != ".tmp"):
        mj, modelo = exporta_mlp(zp)
        mid = f"{mj['algo']}_s{mj['semente']}"
        y, acoes = saidas_torch(modelo, mj["algo"], obs_fixas)
        y_arg = [int(a) for a in y.argmax(1)]
        if y_arg != acoes:      # só pode divergir em empate exato; avisa para investigar
            print(f"  AVISO: {mid}: argmax das saídas difere do predict() do SB3 em {sum(a != b for a, b in zip(y_arg, acoes))} observações")
        mj["id"] = mid
        run = nomes_run.get((mj["algo"], mj["semente"]))
        mj["metricas"] = {k: run["final"][k] for k in ("ret", "ret_desc", "sucesso", "passos", "toca")} if run else {}
        mj["n_parametros"] = sum(c["entrada"] * c["saida"] + c["saida"] for c in mj["camadas"])
        escreve_json(saida / "modelos" / f"{mid}.json", mj)
        paridade["modelos"][mid] = {"saidas": [[f32(v) for v in r] for r in y], "acoes": acoes}
        modelos_idx.append({"id": mid, "algo": mj["algo"], "config": mj["config"], "semente": mj["semente"], "arquivo": f"modelos/{mid}.json",
                            "n_parametros": mj["n_parametros"], "ativacao": mj["ativacao"], "metricas": mj["metricas"]})
        print(f"  modelo {mid}: {mj['n_parametros']} parâmetros, {mj['ativacao']}")
    if not modelos_idx:
        print("  AVISO: nenhum modelo final_* encontrado em " + str(res / "modelos"))

    sarsa_idx = None
    if not args.sem_sarsa:
        nome, config, semente, cfg, q, r = exporta_sarsa(B, saida, runs["final"], melhores, info, args.semente_sarsa, obs_fixas, args.passos_sarsa)
        paridade["sarsa"] = {"id": nome, "q": q}
        sarsa_idx = {"id": nome, "algo": "SARSA", "config": config, "semente": semente, "arquivo": f"modelos/{nome}.json", "cfg": cfg,
                     "metricas": {k: r[k] for k in ("ret", "ret_desc", "sucesso", "passos", "toca")}}
    escreve_json(saida / "paridade.json", paridade)

    # ---- índice --------------------------------------------------------------------------------------------------------
    configs = {}
    for r in sorted(runs["busca"] + runs["final"], key=lambda r: (r["algo"], r["config"])):
        configs.setdefault(r["algo"], {})[r["config"]] = r["hp"]
    for a in melhores:
        configs.setdefault(a, {}).setdefault(melhores[a]["config"], melhores[a]["hp"])
    tab = B.PoliticaTabularDiscretizada()
    indice = {
        "versao": 1, "completo": bool(completo), "rapido": rapido,
        "ambiente": B.PARAMS_PADRAO, "mapa": B.MAPA_PADRAO, "v_estrela_t2": float(tab.v0),
        "protocolo": {k: info[k] for k in ("orcamento_busca", "orcamento_final", "sementes_busca", "sementes_final", "criterio") if k in info},
        "versoes": {k: info[k] for k in ("python", "numpy", "torch", "stable_baselines3", "gymnasium") if k in info},
        "algos": ALGOS, "configs": configs, "melhores": {a: {"config": m["config"], "hp": m["hp"]} for a, m in melhores.items()},
        "contagem": {"busca": n_b, "busca_esperado": esperado_b, "final": n_f, "final_esperado": esperado_f},
        "arquivos": lista_arquivos(saida),
        "modelos": modelos_idx, "sarsa": sarsa_idx,
    }
    if extra:
        indice["extra"] = extra
    escreve_json(saida / "indice.json", indice, indent=1)
    tam = sum(f.stat().st_size for f in saida.rglob("*") if f.is_file())
    print(f"Concluído: {len(list(saida.rglob('*')))} itens em {saida} ({tam / 1024:.0f} KiB). completo={completo}")


if __name__ == "__main__":
    main()
