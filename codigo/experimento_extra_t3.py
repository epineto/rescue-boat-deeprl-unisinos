# %% [markdown]
# # Trabalho 03 — Experimento complementar: orçamento de passos e sanidade da fábrica de agentes
#
# No experimento principal (`experimentos_t3.py`, 400 mil passos nas execuções finais), DQN e Sarsa com *tile coding* aprendem o
# problema; o PPO aprende em 1 de 5 sementes e o A2C não aprende (fica preso em uma pessoa entregue, retorno −50). Este
# complemento procura separar duas explicações: **(a) falta de orçamento** de passos ou **(b) limite do algoritmo/configuração**
# (ou defeito na fábrica de agentes).
#
# **(a) Orçamento maior.** PPO (`ppo_8envs` e `ppo_clip0.3` com 8 ambientes) e A2C (`a2c_16envs`) com **2.000.000 de passos**,
# sementes novas (200, 201, 202), avaliação periódica (curvas) e avaliação final idêntica à do principal (200 episódios, sementes
# de avaliação 10000, 10001, …, política gulosa, γ = 0,99).
#
# **(b) Sanidade da fábrica.** O mesmo código de criação e treino de PPO e A2C (`barco_continuo.criar_modelo`, com os mesmos
# hiperparâmetros, escala de recompensa e `n_envs` do principal) é aplicado ao CartPole-v1 (Gymnasium), 100 mil passos,
# 2 sementes. Se a fábrica estiver correta, o PPO deve atingir retorno próximo de 400 ou mais e o A2C um valor razoável.
#
# **Reuso.** O tema gráfico (com a logo), as funções de execução paralela retomável (`executa_run`, `roda_lote`, `carrega_runs`),
# de curvas (`tabela_curvas`) e de agregação (`agrega`) são extraídos do próprio `experimentos_t3.py`, sem cópia de código; os
# hiperparâmetros vêm de `resultados/busca_resumo_com_hp.csv`. Nada em `resultados/` fora de `resultados/extra/` é alterado.
#
# **Execução.** `python experimento_extra_t3.py` (retomável: runs prontas em `resultados/extra/runs/` são puladas);
# `RAPIDO=1` roda uma versão de poucos minutos em `resultados/extra/_rapido/`.

# %%
import ast, json, os, sys, time, types, warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import gymnasium as gym

AQUI = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(AQUI))
import barco_continuo as B

RAPIDO = os.environ.get("RAPIDO", "0") == "1"
N_PROC = int(os.environ.get("N_PROC", 4))
RAIZ = AQUI.parent
EXTRA = RAIZ / "resultados" / "extra" / ("_rapido" if RAPIDO else "")
FIG = RAIZ / "figuras" / ("_rapido" if RAPIDO else "")
RUNS_X, MODELOS_X = EXTRA / "runs", EXTRA / "modelos"
for d in (EXTRA, FIG, RUNS_X, MODELOS_X):
    d.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Reuso do experimento principal
# O `experimentos_t3.py` é um *script* de células (executa tudo ao ser importado). Por isso, apenas os trechos necessários são
# extraídos da sua árvore sintática e executados em um módulo próprio, registrado em `sys.modules` (necessário para que as funções
# possam ser enviadas aos processos de trabalho): o bloco do tema gráfico e as funções listadas abaixo.

# %%
GAMMA_AVAL = 0.99
N_EP_AVAL_PERIODICA = 20
FONTE_PRINCIPAL = (AQUI / "experimentos_t3.py").read_text()
FUNCOES_REUSADAS = ("executa_run", "roda_lote", "carrega_runs", "tabela_curvas", "agrega")

P = types.ModuleType("principal_t3_extraido")
sys.modules[P.__name__] = P
P.__dict__.update(dict(B=B, json=json, time=time, np=np, pd=pd, os=os, mp=mp, Path=Path, ProcessPoolExecutor=ProcessPoolExecutor,
                       as_completed=as_completed,
                       FIG=FIG, RUNS=RUNS_X, MODELOS=MODELOS_X, PARAMS_ENV={}, GAMMA_AVAL=GAMMA_AVAL, N_EP_AVAL=N_EP_AVAL_PERIODICA,
                       N_PROC=N_PROC, INFO={"tempos_s": {}}, RAPIDO=RAPIDO))


def _salvar_info():
    (EXTRA / "info_extra.json").write_text(json.dumps(P.INFO, ensure_ascii=False, indent=1))


P.salvar_info = _salvar_info
T_INICIO = time.time()
_arq_info = EXTRA / "info_extra.json"
TEMPOS_ANT = json.loads(_arq_info.read_text()).get("tempos_s", {}) if _arq_info.exists() else {}     # tempos de uma execução anterior


def _preserva_tempo(rotulo):
    """Se o lote não executou nada (retomada com tudo pronto), mantém o tempo de relógio da execução original."""
    if P.INFO["tempos_s"].get(rotulo, 0) < 5 and rotulo in TEMPOS_ANT:
        P.INFO["tempos_s"][rotulo] = TEMPOS_ANT[rotulo]

ini = FONTE_PRINCIPAL.index("# ---- tema gráfico")
fim = FONTE_PRINCIPAL.index("# ---- orçamento e protocolo")
exec(compile(FONTE_PRINCIPAL[ini:fim], "experimentos_t3.py[tema]", "exec"), P.__dict__)          # tema, logo, salvar_fig, fmt_passos
arvore = ast.parse(FONTE_PRINCIPAL)
achadas = set()
for no in arvore.body:
    if isinstance(no, ast.FunctionDef) and no.name in FUNCOES_REUSADAS:
        exec(compile(ast.Module([no], []), "experimentos_t3.py[" + no.name + "]", "exec"), P.__dict__)
        achadas.add(no.name)
assert achadas == set(FUNCOES_REUSADAS), f"funções não encontradas no principal: {set(FUNCOES_REUSADAS) - achadas}"
plt, COR, COR_ALGO, COR_REF, salvar_fig, br, fmt_passos, eixo_passos = (P.plt, P.COR, P.COR_ALGO, P.COR_REF, P.salvar_fig, P.br,
                                                                      P.fmt_passos, P.eixo_passos)

# hiperparâmetros das configurações do principal (fonte única: busca_resumo_com_hp.csv)
_hp_csv = pd.read_csv(RAIZ / "resultados" / "busca_resumo_com_hp.csv")
HP = {r.config: json.loads(r.hiperparametros) for r in _hp_csv.itertuples()}
print("Reuso do principal: tema +", sorted(achadas), "| configurações lidas:", len(HP))

# %% [markdown]
# ## 2. Plano de execução e orçamento de tempo
# Velocidades medidas com 4 processos simultâneos (teste curto antes da execução): PPO com 8 ambientes ≈ 1.000 passos/s
# (16 ambientes não acelera: o custo é dominado pelas épocas de otimização), A2C com 16 ambientes ≈ 1.700 a 3.600 passos/s,
# PPO com 1 ambiente ≈ 590 passos/s. Logo, 2 milhões de passos custam ≈ 30 min (PPO, 8 ambientes), ≈ 10 min (A2C, 16 ambientes)
# e ≈ 56 min (PPO, 1 ambiente, inviável). Com ≈ 180 núcleo-minutos disponíveis (45 min × 4 processos), **cortes declarados**:
# `ppo_clip0.3` roda **com 8 ambientes** (não com 1, como no principal) e **somente com 1 semente (200)**.

# %%
ORC_X = 20_000 if RAPIDO else 2_000_000
SEM_X = [200, 201, 202]
SEM_CLIP = [200]                                            # corte declarado: 1 semente para ppo_clip0.3 (8 ambientes)
N_EP_FINAL = 200                                            # igual ao principal (N_EP_FINAL)
N_PONTOS_X = 2 if RAPIDO else 20
CFG_X = {
    "PPO": {"ppo_8envs": (dict(HP["ppo_8envs"]), SEM_X),
            "ppo_clip0.3_8envs": (dict(HP["ppo_clip0.3"], n_envs=8, n_steps=64), SEM_CLIP)},
    "A2C": {"a2c_16envs": (dict(HP["a2c_16envs"]), SEM_X)},
}
specs_extra = [dict(id=f"extra_{a}_{c}_s{s}", fase="extra", algo=a, config=c, hp=hp, semente=s, passos=ORC_X, n_ep_final=N_EP_FINAL,
                    n_pontos=N_PONTOS_X, salvar_modelo=True, salvar_treino=True)
               for a, cs in CFG_X.items() for c, (hp, sems) in cs.items() for s in sems]

ORC_CP = 4_000 if RAPIDO else 100_000
SEM_CP = [200, 201]
CFG_CP = {"ppo_padrao": ("PPO", {}), "ppo_clip0.3": ("PPO", HP["ppo_clip0.3"]), "ppo_8envs": ("PPO", HP["ppo_8envs"]),
          "a2c_padrao": ("A2C", {}), "a2c_16envs": ("A2C", HP["a2c_16envs"])}
specs_cp = [dict(id=f"cartpole_{a}_{c}_s{s}", fase="cartpole", algo=a, config=c, hp=hp, semente=s, passos=ORC_CP,
                 n_ep_final=20, n_pontos=2 if RAPIDO else 5)
            for c, (a, hp) in CFG_CP.items() for s in SEM_CP]
pt = lambda n: f"{n:,}".replace(",", ".")                    # milhar com ponto
print(f"{len(specs_extra)} runs de orçamento maior ({pt(ORC_X)} passos) + {len(specs_cp)} runs de sanidade (CartPole, {pt(ORC_CP)} passos)")

# %% [markdown]
# ## 3. Sanidade da fábrica: CartPole-v1
# `executa_cartpole` usa a mesma fábrica (`B.criar_modelo`) e a mesma convenção de `n_envs` e de escala de recompensa
# (`escala_recompensa` vira um `TransformReward` no ambiente de treino; as métricas usam a recompensa original). A política
# é avaliada de forma determinística (gulosa) em episódios com sementes fixas.

# %%
def avalia_cartpole(modelo, n_ep, semente0):
    env = gym.make("CartPole-v1")
    ret = []
    for e in range(n_ep):
        obs, _ = env.reset(seed=semente0 + e)
        g, fim = 0.0, False
        while not fim:
            obs, r, te, tr, _ = env.step(int(modelo.predict(obs, deterministic=True)[0]))
            g += r
            fim = te or tr
        ret.append(g)
    return np.array(ret)


def executa_cartpole(spec):
    """Treina e avalia uma run no CartPole-v1 com a fábrica do principal; grava ``runs/<id>.json`` (retomável)."""
    arq = RUNS_X / f"{spec['id']}.json"
    if arq.exists():
        return json.loads(arq.read_text())
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    torch.set_num_threads(1)
    t0 = time.time()
    try:
        hp, sem, passos = spec["hp"], spec["semente"], spec["passos"]
        escala, n_envs = float(hp.get("escala_recompensa", 1.0)), int(hp.get("n_envs", 1))
        fabrica = lambda: gym.wrappers.TransformReward(gym.make("CartPole-v1"), lambda r: r * escala)
        env = fabrica() if n_envs == 1 else DummyVecEnv([fabrica for _ in range(n_envs)])
        modelo = B.criar_modelo(spec["algo"], hp, env, sem)
        curva, freq = [], max(1, passos // spec["n_pontos"])

        class Aval(BaseCallback):
            prox = freq

            def _on_step(self):
                if self.num_timesteps >= self.prox:
                    curva.append({"passos": int(self.num_timesteps), "ret": float(avalia_cartpole(self.model, 10, 20_000).mean())})
                    self.prox += freq
                return True

        modelo.learn(total_timesteps=passos, callback=Aval())
        ep = avalia_cartpole(modelo, spec["n_ep_final"], 10_000)
        # chaves ``ret_desc`` e ``sucesso`` mantêm o formato lido por ``roda_lote``: aqui ret_desc = retorno (sem desconto) e
        # sucesso = fração de episódios com retorno >= 475 (critério usual de "resolvido" do CartPole-v1)
        final = {"ret": float(ep.mean()), "ret_ic95": B.ic95(ep), "ret_desc": float(ep.mean()), "sucesso": float((ep >= 475).mean()),
                 "ret_min": float(ep.min())}
        out = {**{k: spec[k] for k in ("id", "fase", "algo", "config", "semente", "passos", "hp", "n_pontos")},
               "final": final, "final_ep": {"ret": ep.tolist()}, "curva": curva, "tempo": time.time() - t0, "erro": None}
    except Exception as e:
        return {"id": spec["id"], "erro": repr(e)}
    tmp = arq.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, default=float))
    tmp.replace(arq)
    return out


_executa_principal = P.executa_run


def executa_run_extra(spec):
    """Despacha: CartPole (sanidade) ou o ``executa_run`` do principal (problema do barco)."""
    return executa_cartpole(spec) if spec["fase"] == "cartpole" else _executa_principal(spec)


P.executa_run = executa_run_extra                           # ``roda_lote`` do principal passa a usar o despachante

# %% [markdown]
# ## 4. Execução (paralela, retomável)
# A sanidade vai primeiro (rápida e informativa: se a fábrica tivesse um defeito, apareceria aqui).

# %%
P.roda_lote(specs_cp, "cartpole")
_preserva_tempo("cartpole")

# %%
P.roda_lote(specs_extra, "extra")
_preserva_tempo("extra")

# %% [markdown]
# ## 5. Tabelas
# Retorno descontado e sucesso finais por semente e média por configuração; curvas; diagnóstico dos episódios de treino
# (retorno bruto e fração de episódios no patamar de "uma pessoa entregue", retorno entre −60 e −40) e entropia da política.

# %%
ids_x = {s["id"] for s in specs_extra}
runs_x = [r for r in P.carrega_runs("extra") if r["id"] in ids_x]
df_x = pd.DataFrame([{"algo": r["algo"], "config": r["config"], "semente": r["semente"], "passos": r["passos"],
                      "ret": r["final"]["ret"], "ret_desc": r["final"]["ret_desc"], "ret_desc_ic95_ep": r["final"]["ret_desc_ic95"],
                      "sucesso": r["final"]["sucesso"], "passos_ep": r["final"]["passos"], "toca_destrocos": r["final"]["toca"],
                      "tempo_s": round(r["tempo"], 1), "hp": json.dumps(r["hp"], ensure_ascii=False)} for r in runs_x])
df_x = df_x.sort_values(["algo", "config", "semente"]).reset_index(drop=True)
df_x.to_csv(EXTRA / "extra_orcamento_por_semente.csv", index=False)
res_x = P.agrega(df_x, ["algo", "config"])
res_x.to_csv(EXTRA / "extra_orcamento_resumo.csv", index=False)
curv_x = P.tabela_curvas(runs_x)
curv_x.to_csv(EXTRA / "extra_curvas.csv", index=False)


def diagnostico(r):
    """Episódios de treino na última décima parte dos passos + entropia (nats) da política treinada em estados da avaliação."""
    out = {"config": r["config"], "semente": r["semente"]}
    arq = RUNS_X / f"{r['id']}_treino.npz"
    if arq.exists():
        t = np.load(arq)["treino"]
        u = t[t[:, 0] >= 0.9 * r["passos"]]
        out.update(ret_treino_final=float(u[:, 1].mean()), sucesso_treino_final=float(u[:, 3].mean()),
                   frac_patamar_1_pessoa=float(((u[:, 1] > -60) & (u[:, 1] < -40)).mean()), n_ep_treino_final=int(len(u)))
    try:
        import torch
        from stable_baselines3 import A2C, PPO
        m = {"PPO": PPO, "A2C": A2C}[r["algo"]].load(str(MODELOS_X / r["id"]), device="cpu")
        env, obss = B.RescueBoatContEnv(), []
        for e in range(5):
            o, _ = env.reset(seed=10_000 + e)
            fim = False
            while not fim:
                obss.append(o)
                o, _, te, tr, _ = env.step(int(m.predict(o, deterministic=True)[0]))
                fim = te or tr
        with torch.no_grad():
            ent = m.policy.get_distribution(torch.as_tensor(np.array(obss), dtype=torch.float32)).entropy().mean().item()
        out["entropia_politica_nats"] = float(ent)               # máximo para 4 ações: ln 4 = 1,386
    except Exception as e:                                       # modelo indisponível: segue sem a entropia
        out["entropia_politica_nats"] = float("nan")
    return out


df_diag = pd.DataFrame([diagnostico(r) for r in runs_x]).sort_values(["config", "semente"])
df_diag.to_csv(EXTRA / "extra_diagnostico_treino.csv", index=False)

ids_c = {s["id"] for s in specs_cp}
runs_c = [r for r in P.carrega_runs("cartpole") if r["id"] in ids_c]
df_c = pd.DataFrame([{"algo": r["algo"], "config": r["config"], "semente": r["semente"], "passos": r["passos"], "ret": r["final"]["ret"],
                      "ret_ic95_ep": r["final"]["ret_ic95"], "ret_min": r["final"]["ret_min"], "frac_ep_475": r["final"]["sucesso"],
                      "tempo_s": round(r["tempo"], 1), "hp": json.dumps(r["hp"], ensure_ascii=False)} for r in runs_c])
df_c = df_c.sort_values(["algo", "config", "semente"]).reset_index(drop=True)
df_c.to_csv(EXTRA / "cartpole_por_semente.csv", index=False)
res_c = df_c.groupby(["algo", "config"]).agg(n=("semente", "count"), ret=("ret", "mean"), ret_min_sem=("ret", "min"), ret_max_sem=("ret", "max"),
                                             frac_ep_475=("frac_ep_475", "mean"), tempo_s=("tempo_s", "mean")).reset_index()
res_c.to_csv(EXTRA / "cartpole_resumo.csv", index=False)
pd.DataFrame([{"algo": r["algo"], "config": r["config"], "semente": r["semente"], **c} for r in runs_c for c in r["curva"]]).to_csv(
    EXTRA / "cartpole_curvas.csv", index=False)
print(res_x.round(2).to_string(index=False))
print(res_c.round(1).to_string(index=False))

# %% [markdown]
# ## 6. Figura

# %%
ref_tab = float(pd.read_csv(RAIZ / "resultados" / "linhas_de_base.csv").set_index("agente").loc["tabular", "ret_desc"])
COR_CFG = {"ppo_8envs": COR["vermelho"], "ppo_clip0.3_8envs": "#E67E22", "a2c_16envs": COR["verde"]}
ROT_CFG = {"ppo_8envs": "PPO 8 amb. (3 sementes)", "ppo_clip0.3_8envs": "PPO clip 0,3, 8 amb. (1 semente)", "a2c_16envs": "A2C 16 amb. (3 sementes)"}
fig, axs = plt.subplots(1, 3, figsize=(11.6, 3.5), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1]))
for ax, (m, tit, ylab) in zip(axs[:2], (("ret_desc", "Retorno da política gulosa", "retorno descontado (γ = 0,99)"),
                                        ("sucesso", "Taxa de sucesso", "taxa de sucesso"))):
    for c in CFG_X["PPO"].keys() | CFG_X["A2C"].keys():
        d = curv_x[curv_x.config == c]
        if d.empty:
            continue
        for s, g in d.groupby("semente"):
            ax.plot(g.passos, g[m], color=COR_CFG[c], lw=0.8, alpha=0.35)
        ax.plot(*zip(*d.groupby("passos")[m].mean().items()), color=COR_CFG[c], lw=2.0)
    ax.axvline(400_000, color="#999999", ls="--", lw=1)
    if m == "ret_desc":
        ax.axhline(ref_tab, color=COR_REF["tabular"], ls="--", lw=1.3)
    else:
        ax.axhline(1.0, color=COR_REF["tabular"], ls="--", lw=1.3)
        ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("passos de interação"); ax.set_ylabel(ylab); ax.set_title(tit)
    eixo_passos(ax)
    ax.set_xlim(0, ORC_X)
ax = axs[2]
cfgs = list(CFG_CP)
for i, c in enumerate(cfgs[::-1]):
    v = df_c[df_c.config == c].ret.values
    desl = np.linspace(-0.14, 0.14, len(v)) if len(v) > 1 else [0]       # desloca verticalmente as sementes (valores iguais não se sobrepõem)
    ax.scatter(v, i + np.asarray(desl), s=34, color=COR_ALGO[CFG_CP[c][0]], edgecolor="white", lw=0.8, zorder=5)
ax.set_yticks(range(len(cfgs))); ax.set_yticklabels([("padrão SB3" if c.endswith("_padrao") else c.split("_", 1)[1]) + " (" + CFG_CP[c][0] + ")" for c in cfgs[::-1]],
                                                    fontsize=7.5)
ax.axvline(400, color="#999999", ls="--", lw=1)
ax.set_xlim(0, 520); ax.set_ylim(-0.6, len(cfgs) - 0.4)
ax.set_xlabel("retorno final (máx. 500)"); ax.set_title("Sanidade: CartPole-v1, 100 mil passos")
from matplotlib.lines import Line2D
leg = [Line2D([0], [0], color=COR_CFG[c], lw=2) for c in COR_CFG] + [
    Line2D([0], [0], color=COR_REF["tabular"], ls="--", lw=1.3), Line2D([0], [0], color="#999999", ls="--", lw=1)]
rot = list(ROT_CFG.values()) + ["tabular discretizada (taxa 1,0 / retorno 138,5)", "400 mil passos (orçamento do principal) / retorno 400 no CartPole"]
fig.legend(leg, rot, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=7.5, handlelength=2.2, columnspacing=1.6)
salvar_fig(fig, "fig_extra_orcamento.png")

# %% [markdown]
# ## 7. Resumo em Markdown
# As tabelas vêm dos CSVs; a interpretação (texto) é lida de `resultados/extra/interpretacao.md`, se existir.

# %%
def md(df, fmt):
    cab = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return cab + "\n".join("| " + " | ".join(fmt.get(c, str)(v) for c, v in r.items()) + " |" for r in df.to_dict("records"))


f1 = lambda v: br(v, 1)
f2 = lambda v: br(v, 2)
tempo_lote = P.INFO["tempos_s"]
sec = [f"# Resumo do experimento complementar (orçamento de passos e sanidade da fábrica)\n",
       f"Gerado por `codigo/experimento_extra_t3.py`. Orçamento: **{pt(ORC_X)} passos** por run do barco; avaliação final igual à do principal "
       f"(200 episódios, sementes de avaliação 10000+, política gulosa, γ = 0,99); {N_PROC} processos. Tempo de relógio: sanidade "
       f"{br(tempo_lote.get('cartpole', float('nan')) / 60, 1)} min, orçamento maior {br(tempo_lote.get('extra', float('nan')) / 60, 1)} min "
       f"(se a execução foi retomada, apenas o trecho da última execução).\n"]
sec.append("## O que foi cortado\n- `ppo_clip0.3` (a configuração escolhida no principal) foi executada **com 8 ambientes** (`n_envs = 8`, `n_steps = 64`; "
           "no principal usou 1 ambiente, ≈ 590 passos/s, o que custaria ≈ 56 min por run) e **com 1 semente (200)**, para caber no orçamento de ≈ 45 min.\n"
           "- Todas as demais combinações pedidas foram executadas integralmente (PPO `ppo_8envs` e A2C `a2c_16envs`, sementes 200 a 202).\n")
t = df_x[["config", "semente", "ret_desc", "sucesso", "passos_ep", "toca_destrocos", "tempo_s"]].copy()
sec.append("## Orçamento maior: resultado final por semente (" + pt(ORC_X) + " passos)\n" + md(t, {"ret_desc": f2, "sucesso": f2, "passos_ep": f1, "toca_destrocos": f2, "tempo_s": lambda v: f"{v:.0f}"}))
t = res_x[["config", "n", "ret_desc", "ret_desc_dp", "ret_desc_ic95", "sucesso", "tempo_s"]]
sec.append("\n## Orçamento maior: médias por configuração\n" + md(t, {"ret_desc": f2, "ret_desc_dp": f2, "ret_desc_ic95": f2, "sucesso": f2, "tempo_s": lambda v: f"{v:.0f}"}) +
           "\n\nDesvio-padrão e IC95% entre sementes (t de Student); com n = 1 não há dispersão (célula vazia ou `nan`).\n")
c4 = curv_x[curv_x.passos == 400_000][["config", "semente", "ret_desc", "sucesso"]]
sec.append("## O mesmo run aos 400 mil passos (avaliação periódica, 20 episódios) contra o final ao fim do orçamento\n" +
           md(c4.merge(df_x[["config", "semente", "ret_desc", "sucesso"]], on=["config", "semente"], suffixes=("_400mil", "_2mi")),
              {c: f2 for c in ("ret_desc_400mil", "sucesso_400mil", "ret_desc_2mi", "sucesso_2mi")}))
sec.append("\n\n## Diagnóstico do treino (última décima parte dos passos)\n"
           "`frac_patamar_1_pessoa`: fração dos episódios de treino (política estocástica) com retorno bruto entre −60 e −40; "
           "`entropia_politica_nats`: entropia média da política nos estados da avaliação (máximo ln 4 = 1,386).\n\n" +
           md(df_diag, {c: f2 for c in df_diag.columns if c not in ("config", "semente", "n_ep_treino_final")}))
cp = df_c[["config", "semente", "ret", "frac_ep_475", "ret_min", "tempo_s"]]
sec.append("\n\n## Sanidade: CartPole-v1 (100 mil passos, 20 episódios de avaliação, política gulosa)\n" +
           md(cp, {"ret": f1, "frac_ep_475": f2, "ret_min": f1, "tempo_s": lambda v: f"{v:.0f}"}) +
           "\n\n" + md(res_c, {"ret": f1, "ret_min_sem": f1, "ret_max_sem": f1, "frac_ep_475": f2, "tempo_s": lambda v: f"{v:.0f}"}) +
           "\n\n`frac_ep_475`: fração dos episódios de avaliação com retorno ≥ 475 (critério usual de problema resolvido).\n")
arq_int = EXTRA / "interpretacao.md"
if arq_int.exists():
    sec.append("\n" + arq_int.read_text())
sec.append("\n![Figura](../../figuras/fig_extra_orcamento.png)\n")
(EXTRA / "RESUMO.md").write_text("\n".join(sec))
P.INFO["tempos_s"]["total"] = round(P.INFO["tempos_s"].get("cartpole", 0) + P.INFO["tempos_s"].get("extra", 0), 1)   # soma dos dois lotes
_salvar_info()
print(f"Concluído em {time.time() - T_INICIO:.0f} s | {EXTRA / 'RESUMO.md'}")
