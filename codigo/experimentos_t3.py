# %% [markdown]
# # Trabalho 03 — Aprendizado por Reforço (PPGCA/Unisinos, 2026/2)
# **Barco de resgate com estados contínuos: DQN, PPO e A2C (stable-baselines3) e Sarsa semi-gradiente com *tile coding***
#
# Integrantes do grupo: Epitácio Vicente do Nascimento Neto e Antonio Clerton Santana de Araujo · Professor: Gabriel de Oliveira Ramos
#
# **Problema.** O barco de resgate 6×6 do Trabalho 2 (abrigo H, três pessoas ilhadas, capacidade 2, correnteza na linha 2, destroços
# logo abaixo, quatro blocos) é reformulado como um problema de **aproximação de função**: a posição do barco passa a ser **contínua**,
# o movimento tem **ruído gaussiano**, e a correnteza é um **elemento dinâmico** (arrasto aleatório com intensidade variável no
# tempo, observável). O MDP discreto do Trabalho 2 permanece como **referência**: sua política ótima, executada no ambiente
# contínuo por consulta à célula que contém a posição, é a "linha de base tabular discretizada".
#
# **Algoritmos.** DQN (valor, *off-policy*, repetição de experiência), PPO (gradiente de política, *on-policy*, recorte) e A2C
# (ator-crítico síncrono), todos do stable-baselines3, mais o Sarsa semi-gradiente linear com *tile coding* implementado à mão
# (Sutton e Barto, caps. 9 e 10; Aula 13).
#
# **Protocolo.** (1) hiperparâmetros padrão do SB3 como configuração de partida; (2) busca com 6 configurações por algoritmo
# (a padrão inclusa), ≥ 3 sementes e orçamento de passos igual entre algoritmos; (3) a melhor configuração de cada algoritmo é
# reexecutada com ≥ 5 sementes novas e orçamento maior; (4) avaliação da política determinística em episódios com sementes fixas,
# com intervalo de confiança de 95%. **Critério de escolha (declarado antes de ver os resultados finais):** maior retorno descontado
# médio (γ = 0,99) na busca, com desempate pela taxa de sucesso. **Hipóteses:** H1, DQN é mais eficiente em amostras que PPO e A2C;
# H2, PPO é mais estável entre sementes que A2C; H3, a aproximação de função não atinge o desempenho da linha de base tabular
# discretizada e/ou da heurística.
#
# **Como executar.** `RAPIDO = True` roda uma versão curta (poucos minutos) para teste; `RAPIDO = False` executa o protocolo completo
# (estimativa em `resultados/piloto_calibracao.md`). A execução é retomável: runs já gravadas em `resultados/runs/` não são refeitas.
# Em linha de comando: `python experimentos_t3.py` (variável de ambiente `RAPIDO=1` para a versão curta).

# %%
import json, math, os, platform, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")          # 1 thread por processo: o paralelismo é entre runs
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:                                                    # no Colab o SB3 pode faltar; instala se necessário
    import stable_baselines3
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "stable-baselines3"], check=True)
    import stable_baselines3
import gymnasium, torch, scipy
from scipy import stats

AQUI = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(AQUI))
import barco_continuo as B                              # no notebook, este módulo é gravado por uma célula anterior

RAPIDO = os.environ.get("RAPIDO", "0") == "1"           # True: versão reduzida (poucos minutos)
N_PROC = int(os.environ.get("N_PROC", 4))
RAIZ = AQUI.parent if (AQUI.parent / "resultados").exists() or (AQUI.parent / "BRIEF_T3.md").exists() else AQUI
RES = RAIZ / "resultados" / ("_rapido" if RAPIDO else "")
FIG = RAIZ / "figuras" / ("_rapido" if RAPIDO else "")
RUNS = RES / "runs"
MODELOS = RES / "modelos"
for d in (RES, FIG, RUNS, MODELOS):
    d.mkdir(parents=True, exist_ok=True)

# ---- tema gráfico (cópia embutida do tema Unisinos; fonte com fallback para evitar avisos "findfont") ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
COR = {"azul": "#1D2369", "azul2": "#4A5AB8", "vermelho": "#C0392B", "cinza": "#666666", "claro": "#E9ECF3", "verde": "#2E7D5B"}
_disp = {f.name for f in _fm.fontManager.ttflist}
_fonte = [f for f in ["Arial", "Liberation Sans", "DejaVu Sans"] if f in _disp][:1] or ["sans-serif"]
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": _fonte, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": COR["azul"],
    "axes.labelcolor": "#333333", "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#999999",
    "axes.grid": True, "grid.color": "#E3E6EE", "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.8,
    "legend.frameon": False, "xtick.color": "#555555", "ytick.color": "#555555"})
COR_ALGO = {"DQN": COR["azul"], "PPO": COR["vermelho"], "A2C": COR["verde"], "SARSA": COR["azul2"]}
COR_REF = {"tabular": "#B8860B", "heuristica": COR["cinza"], "aleatoria": "#999999"}
NOME_ALGO = {"DQN": "DQN", "PPO": "PPO", "A2C": "A2C", "SARSA": "Sarsa (tile coding)"}
_LOGO = AQUI.parent / "assets" / "logo_unisinos.png"      # logo do repositório (opcional: sem o arquivo, as figuras saem sem logo)


def fmt_passos(x, _pos=None):
    """Rótulo de eixo de passos de interação: ``0``, ``100 mil``, ``1 mi``, ``1,5 mi`` (evita rótulos colados)."""
    if abs(x) < 1:
        return "0"
    if abs(x) >= 1e6:
        return f"{x / 1e6:g}".replace(".", ",") + " mi"
    return f"{x / 1e3:g} mil".replace(".", ",")


def eixo_passos(ax, n=5):
    """Formata o eixo x em milhares/milhões, com no máximo ``n`` marcas."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=n, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_passos))


def salvar_fig(fig, nome, topo=0.87):
    """Grava a figura em ``FIG`` (com a logo da Unisinos no canto superior direito, se o arquivo existir).

    ``topo``: fração da altura reservada às áreas de plotagem (o restante abriga a logo; reduzir se os títulos tiverem 2 linhas)."""
    try:
        fig.tight_layout(rect=[0, 0, 1, topo] if _LOGO.exists() else None)
        if _LOGO.exists():
            ax = fig.add_axes([0.87, 0.915, 0.12, 0.075])
            ax.imshow(plt.imread(str(_LOGO)))
            ax.axis("off")
    except Exception:
        fig.tight_layout()
    fig.savefig(FIG / nome, bbox_inches="tight")
    plt.close(fig)


def br(x, n=2):
    """Número com vírgula decimal (padrão do relatório)."""
    return f"{x:.{n}f}".replace(".", ",")


# ---- orçamento e protocolo --------------------------------------------------------------------
GAMMA_AVAL = 0.99
if RAPIDO:
    ORC_BUSCA, ORC_FINAL = 3_000, 5_000
    SEM_BUSCA, SEM_FINAL = [0, 1], [100, 101]
    N_CFG, N_EP_BUSCA, N_EP_FINAL, N_EP_AVAL, N_EP_BASE = 2, 10, 20, 5, 30
    N_PONTOS_CURVA = 5
else:
    ORC_BUSCA, ORC_FINAL = 200_000, 400_000            # passos de interação (ver piloto_calibracao.md)
    SEM_BUSCA, SEM_FINAL = [0, 1, 2], [100, 101, 102, 103, 104]
    N_CFG, N_EP_BUSCA, N_EP_FINAL, N_EP_AVAL, N_EP_BASE = 6, 100, 200, 20, 500
    N_PONTOS_CURVA = 10
CRITERIO = "maior retorno descontado médio (γ = 0,99) na busca; desempate pela taxa de sucesso"
INFO = {"rapido": RAPIDO, "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "torch": torch.__version__, "stable_baselines3": stable_baselines3.__version__, "gymnasium": gymnasium.__version__,
        "so": platform.platform(), "cpus": os.cpu_count(), "processos": N_PROC, "orcamento_busca": ORC_BUSCA, "orcamento_final": ORC_FINAL,
        "sementes_busca": SEM_BUSCA, "sementes_final": SEM_FINAL, "criterio": CRITERIO, "params_ambiente": B.PARAMS_PADRAO, "tempos_s": {}}
T_INICIO = time.time()


def salvar_info():
    INFO["tempo_total_s"] = round(time.time() - T_INICIO, 1)
    (RES / "info.json").write_text(json.dumps(INFO, ensure_ascii=False, indent=1))


# %% [markdown]
# ## 1. O ambiente contínuo e as referências do Trabalho 2
# `RescueBoatContEnv` (módulo `barco_continuo.py`): posição `(x, y) ∈ [0, 6)²`, ações discretas {N, S, L, O} com passo de 0,7 célula
# e ruído gaussiano, truncamento em 100 passos, blocos como retângulos (colisão ⇒ o barco não se move), destroços como discos (recompensa −10 ao entrar),
# embarque por raio, desembarque no abrigo por raio e correnteza na faixa `y ∈ [2, 3)` com arrasto `v_max · I + ruído` para o sul,
# em que a intensidade `I` segue um processo AR(1) recortado e é **observável**. Recompensas: −1 por passo, −10 ao tocar nos destroços
# (no lugar do −1), +50 por pessoa entregue e +100 ao entregar a última. A observação (`Box` de 13 componentes) traz a posição
# normalizada, a situação de cada pessoa (*one-hot*), a ocupação do barco e `I`; **não há** atributos de direção ao alvo.

# %%
PARAMS_ENV = {}                                          # vazio = parâmetros calibrados (B.PARAMS_PADRAO)
env_demo = B.RescueBoatContEnv(**PARAMS_ENV)
print("Parâmetros do ambiente:", json.dumps(B.PARAMS_PADRAO))
print("Espaço de observação:", env_demo.observation_space, "| ações:", env_demo.action_space)
print(f"Velocidade do simulador (sem SB3): {B.velocidade(20000):,.0f} passos/s".replace(",", "."))
obs, _ = env_demo.reset(seed=0)
print(env_demo.render() if env_demo.render_mode == "ansi" else B.RescueBoatContEnv(render_mode="ansi").render())

# %% [markdown]
# ### 1.1 Linhas de base (sem treino)
# **Tabular discretizada**: Iteração de Valor sobre o MDP discreto do Trabalho 2 (γ = 0,99), com a ação lida na célula que contém a
# posição contínua (ignora a posição dentro da célula, o ruído e a intensidade). **Heurística**: caminho mais curto (BFS) que evita
# blocos e destroços, sem treino. **Aleatória**: ações uniformes. Todas usam os mesmos episódios de avaliação (sementes fixas).

# %%
t0 = time.time()
tab = B.PoliticaTabularDiscretizada()
print(f"V*(s0) do MDP discreto do T2 (γ = 0,99): {tab.v0:.2f}  [esperado 161,93]")
INFO["v_estrela_t2"] = tab.v0
env_base = B.RescueBoatContEnv(**PARAMS_ENV)
bases = {"tabular": tab, "heuristica": B.PoliticaHeuristica(env_base), "aleatoria": B.politica_aleatoria(0)}
linhas = []
for nome, pol in bases.items():
    av = B.avaliar(pol, N_EP_BASE, gamma=GAMMA_AVAL, env=env_base)
    r = B.resumo(av)
    linhas.append({"agente": nome, "n_episodios": N_EP_BASE, **r})
df_base = pd.DataFrame(linhas)
df_base.to_csv(RES / "linhas_de_base.csv", index=False)
BASE = {l["agente"]: l for l in linhas}
print(df_base[["agente", "ret", "ret_desc", "ret_desc_ic95", "sucesso", "passos", "toca"]].round(2).to_string(index=False))
INFO["tempos_s"]["linhas_de_base"] = round(time.time() - t0, 1)

# %% [markdown]
# ## 2. Configurações dos algoritmos
# Cada algoritmo tem 6 configurações (a primeira é sempre o **padrão do SB3**). Variam-se a taxa de aprendizado, γ e os
# hiperparâmetros próprios: DQN (fração e valor final da exploração ε, intervalo de atualização da rede-alvo, largura da rede,
# tamanho do *buffer* e do lote nas configurações-base), PPO (`n_steps`, `n_envs`, `ent_coef`, `clip_range`), A2C (`n_steps`,
# `n_envs`, `ent_coef`, `vf_coef`) e Sarsa (resolução e número de grades do *tile coding*, α, ε). `escala_recompensa` multiplica a
# recompensa vista pelo agente (as métricas usam sempre a recompensa original); `n_envs` é o número de ambientes síncronos
# (PPO/A2C). As configurações-base `_DQN`, `_PPO` e `_A2C` foram definidas no piloto de calibração (`resultados/piloto_calibracao.md`).

# %%
_DQN = dict(learning_rate=1e-3, buffer_size=100_000, learning_starts=1000, exploration_fraction=0.4, target_update_interval=500,
            batch_size=64, escala_recompensa=0.1)
_PPO = dict(learning_rate=1e-3, n_steps=512, batch_size=128, ent_coef=0.05, escala_recompensa=0.1)
_A2C = dict(n_envs=8, n_steps=16, learning_rate=1e-3, ent_coef=0.05, gae_lambda=0.95, escala_recompensa=0.1)
CONFIGS = {
    "DQN": {
        "dqn_padrao": {},
        "dqn_lr1e-3_expl0.4": dict(_DQN),
        "dqn_expl0.2_eps0.1": dict(_DQN, exploration_fraction=0.2, exploration_final_eps=0.1),
        "dqn_lr5e-4_rede128": dict(_DQN, learning_rate=5e-4, target_update_interval=1000, net_arch=[128, 128]),
        "dqn_gamma0.95": dict(_DQN, gamma=0.95),
        "dqn_sem_escala": dict(_DQN, escala_recompensa=1.0),
    },
    "PPO": {
        "ppo_padrao": {},
        "ppo_lr1e-3_ent0.05": dict(_PPO),
        "ppo_lr3e-4_ent0.01": dict(_PPO, learning_rate=3e-4, ent_coef=0.01, batch_size=64),
        "ppo_8envs": dict(_PPO, n_envs=8, n_steps=64),
        "ppo_clip0.3": dict(_PPO, clip_range=0.3),
        "ppo_gamma0.95": dict(_PPO, gamma=0.95),
    },
    "A2C": {
        "a2c_padrao": {},
        "a2c_8envs_n16": dict(_A2C),
        "a2c_8envs_ent0.1": dict(_A2C, ent_coef=0.1),
        "a2c_8envs_n32": dict(_A2C, n_steps=32),
        "a2c_16envs": dict(_A2C, n_envs=16),
        "a2c_vf1.0_lr5e-4": dict(_A2C, vf_coef=1.0, learning_rate=5e-4),
    },
    "SARSA": {
        "sarsa_8x8x3_a0.1": dict(n_tiles=(8, 8, 3), n_til=8, alpha=0.1),
        "sarsa_8x8x3_a0.3": dict(n_tiles=(8, 8, 3), n_til=8, alpha=0.3),
        "sarsa_12x12x4_a0.1": dict(n_tiles=(12, 12, 4), n_til=8, alpha=0.1),
        "sarsa_12x12x4_a0.3": dict(n_tiles=(12, 12, 4), n_til=8, alpha=0.3),
        "sarsa_6x6x3_a0.3": dict(n_tiles=(6, 6, 3), n_til=8, alpha=0.3),
        "sarsa_8x8x3_eps0.5": dict(n_tiles=(8, 8, 3), n_til=8, alpha=0.3, eps0=0.5, eps1=0.05),
    },
}
ALGOS = ["DQN", "PPO", "A2C", "SARSA"]
CONFIGS = {a: dict(list(c.items())[:N_CFG]) for a, c in CONFIGS.items()}
print({a: list(c) for a, c in CONFIGS.items()})


# %% [markdown]
# ## 3. Execução das runs (paralela e retomável)
# Cada run (algoritmo × configuração × semente) grava um JSON em `resultados/runs/`; se o arquivo existe, a run não é refeita.
# Os processos usam 1 thread do torch. Avaliação final: política determinística (gulosa), episódios com sementes fixas
# (10000, 10001, …), as mesmas para todos os agentes. A curva de aprendizado vem de avaliações periódicas (20 episódios).

# %%
def executa_run(spec):
    """Treina e avalia uma run. Grava ``runs/<id>.json`` (e o modelo, se pedido); retoma se o arquivo já existe."""
    arq = RUNS / f"{spec['id']}.json"
    if arq.exists():
        return json.loads(arq.read_text())
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    algo, hp, sem, passos = spec["algo"], spec["hp"], spec["semente"], spec["passos"]
    freq = max(1, passos // spec["n_pontos"])
    try:
        if algo == "SARSA":
            hp_s = dict(hp)
            ag = B.SarsaSemiGradiente(semente=sem, **{"gamma": GAMMA_AVAL, **hp_s})
            curva, log, _ = ag.treinar(passos, PARAMS_ENV, freq_aval=freq, n_ep_aval=N_EP_AVAL, gamma_aval=GAMMA_AVAL)
            av = B.avaliar(ag.politica(), spec["n_ep_final"], gamma=GAMMA_AVAL, **PARAMS_ENV)
        else:
            r = B.treinar(algo, hp, sem, passos, PARAMS_ENV, gamma_aval=GAMMA_AVAL, freq_aval=freq, n_ep_aval=N_EP_AVAL,
                          n_ep_final=spec["n_ep_final"])
            curva, log, av = r["curva"], r["treino"], r["final_ep"]
            if spec.get("salvar_modelo"):
                r["modelo"].save(str(MODELOS / spec["id"]))
        if spec.get("salvar_treino"):
            np.savez_compressed(RUNS / f"{spec['id']}_treino.npz", treino=log)
        out = {**{k: spec[k] for k in ("id", "fase", "algo", "config", "semente", "passos", "hp", "n_pontos")},
               "final": B.resumo(av), "final_ep": {k: np.asarray(v).tolist() for k, v in av.items()},
               "curva": curva, "tempo": time.time() - t0, "erro": None}
    except Exception as e:                                # a run com erro não grava arquivo (será refeita na retomada)
        return {"id": spec["id"], "erro": repr(e)}
    tmp = arq.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, default=float))
    tmp.replace(arq)
    return out


def roda_lote(specs, rotulo):
    """Executa as runs em ``N_PROC`` processos (mais caras primeiro) e mostra o progresso."""
    t0 = time.time()
    pend = [s for s in specs if not (RUNS / f"{s['id']}.json").exists()]
    sps = {"PPO": 400, "A2C": 600, "DQN": 520, "SARSA": 4000}          # passos/s por processo (piloto)
    custo = lambda s: s["passos"] / (sps[s["algo"]] * (3.5 if s["hp"].get("n_envs", 1) > 1 else 1))
    pend.sort(key=lambda s: -custo(s))                                  # mais caras primeiro: melhor balanceamento
    print(f"[{rotulo}] {len(specs)} runs, {len(specs) - len(pend)} já prontas, {len(pend)} a executar com {N_PROC} processos")
    erros = []
    if pend:
        with ProcessPoolExecutor(max_workers=N_PROC, mp_context=mp.get_context("fork")) as ex:
            fut = {ex.submit(executa_run, s): s for s in pend}
            for k, f in enumerate(as_completed(fut), 1):
                r = f.result()
                if r.get("erro"):
                    erros.append(r)
                    print(f"  ERRO {r['id']}: {r['erro']}")
                else:
                    print(f"  [{k}/{len(pend)}] {r['id']}: retorno desc. {r['final']['ret_desc']:.1f}, sucesso {r['final']['sucesso']:.2f}, {r['tempo']:.0f} s", flush=True)
    INFO["tempos_s"][rotulo] = round(time.time() - t0, 1)
    INFO.setdefault("erros", []).extend(erros)
    salvar_info()
    print(f"[{rotulo}] concluído em {time.time() - t0:.0f} s")


def carrega_runs(fase):
    out = []
    for f in sorted(RUNS.glob(f"{fase}_*.json")):
        out.append(json.loads(f.read_text()))
    return out


# %% [markdown]
# ## 4. Busca de hiperparâmetros
# 4 algoritmos × 6 configurações × 3 sementes (0, 1, 2), mesmo orçamento de passos para todos. **Todas** as configurações e sementes
# vão para `resultados/busca_hiperparametros.csv`.

# %%
specs_busca = [dict(id=f"busca_{a}_{c}_s{s}", fase="busca", algo=a, config=c, hp=hp, semente=s, passos=ORC_BUSCA,
                    n_ep_final=N_EP_BUSCA, n_pontos=N_PONTOS_CURVA)
               for a in ALGOS for c, hp in CONFIGS[a].items() for s in SEM_BUSCA]
roda_lote(specs_busca, "busca")
runs_b = [r for r in carrega_runs("busca") if r["id"] in {s["id"] for s in specs_busca}]
df_busca = pd.DataFrame([{"algo": r["algo"], "config": r["config"], "semente": r["semente"], "passos": r["passos"],
                          "ret": r["final"]["ret"], "ret_desc": r["final"]["ret_desc"], "ret_desc_ic95_ep": r["final"]["ret_desc_ic95"],
                          "sucesso": r["final"]["sucesso"], "passos_ep": r["final"]["passos"], "toca_destrocos": r["final"]["toca"],
                          "tempo_s": round(r["tempo"], 1), "hp": json.dumps(r["hp"], ensure_ascii=False)} for r in runs_b])
df_busca = df_busca.sort_values(["algo", "config", "semente"]).reset_index(drop=True)
df_busca.to_csv(RES / "busca_hiperparametros.csv", index=False)


def agrega(df, chaves):
    """Média, desvio e IC95% (entre sementes) por grupo."""
    g = df.groupby(chaves)
    r = g.agg(n=("semente", "count"), ret_desc=("ret_desc", "mean"), ret_desc_dp=("ret_desc", "std"), ret=("ret", "mean"),
              sucesso=("sucesso", "mean"), passos_ep=("passos_ep", "mean"), toca_destrocos=("toca_destrocos", "mean"),
              tempo_s=("tempo_s", "mean")).reset_index()
    ic = g["ret_desc"].apply(B.ic95).reset_index(name="ret_desc_ic95")
    return r.merge(ic, on=chaves)


res_busca = agrega(df_busca, ["algo", "config"])
# critério declarado a priori: maior retorno descontado médio; desempate (diferença < 1e-9) pela taxa de sucesso
melhores = {}
for a in ALGOS:
    sub = res_busca[res_busca.algo == a].sort_values(["ret_desc", "sucesso"], ascending=False)
    melhores[a] = sub.iloc[0]["config"]
res_busca["escolhida"] = [melhores[r.algo] == r.config for r in res_busca.itertuples()]
res_busca["posicao"] = res_busca.groupby("algo")["ret_desc"].rank(ascending=False, method="first").astype(int)
res_busca.to_csv(RES / "busca_resumo.csv", index=False)
(RES / "melhores_configs.json").write_text(json.dumps({a: {"config": c, "hp": CONFIGS[a][c]} for a, c in melhores.items()}, ensure_ascii=False, indent=1))
print(res_busca[["algo", "config", "ret_desc", "ret_desc_dp", "sucesso", "toca_destrocos", "posicao", "escolhida"]].round(2).to_string(index=False))
print("Melhores configurações:", melhores)

# %% [markdown]
# ## 5. Melhores configurações com mais sementes e orçamento maior
# Cada algoritmo usa a configuração escolhida na busca, 5 sementes novas (100 a 104) e orçamento maior; os modelos são salvos.
# Avaliação com 200 episódios por run; o intervalo de confiança de 95% é calculado **entre as sementes** (t de Student).

# %%
specs_final = [dict(id=f"final_{a}_{melhores[a]}_s{s}", fase="final", algo=a, config=melhores[a], hp=CONFIGS[a][melhores[a]], semente=s,
                    passos=ORC_FINAL, n_ep_final=N_EP_FINAL, n_pontos=N_PONTOS_CURVA * 2 + 5, salvar_modelo=a != "SARSA", salvar_treino=True)
               for a in ALGOS for s in SEM_FINAL]
roda_lote(specs_final, "final")
runs_f = [r for r in carrega_runs("final") if r["id"] in {s["id"] for s in specs_final}]
df_final = pd.DataFrame([{"algo": r["algo"], "config": r["config"], "semente": r["semente"], "passos": r["passos"],
                          "ret": r["final"]["ret"], "ret_desc": r["final"]["ret_desc"], "ret_desc_ic95_ep": r["final"]["ret_desc_ic95"],
                          "sucesso": r["final"]["sucesso"], "passos_ep": r["final"]["passos"], "toca_destrocos": r["final"]["toca"],
                          "tempo_s": round(r["tempo"], 1), "hp": json.dumps(r["hp"], ensure_ascii=False)} for r in runs_f])
df_final = df_final.sort_values(["algo", "semente"]).reset_index(drop=True)
df_final.to_csv(RES / "resultado_final.csv", index=False)

linhas = []
for a in ALGOS:
    d = df_final[df_final.algo == a]
    l = {"agente": a, "config": melhores[a], "n_sementes": len(d), "passos": ORC_FINAL}
    for m in ("ret_desc", "ret", "sucesso", "passos_ep", "toca_destrocos"):
        l[m] = d[m].mean()
        l[m + "_dp"] = d[m].std(ddof=1)
        l[m + "_ic95"] = B.ic95(d[m])
    linhas.append(l)
for nome in ("tabular", "heuristica", "aleatoria"):
    b = BASE[nome]
    linhas.append({"agente": nome, "config": "-", "n_sementes": 0, "passos": 0, "ret_desc": b["ret_desc"], "ret_desc_ic95": b["ret_desc_ic95"],
                   "ret": b["ret"], "ret_ic95": b["ret_ic95"], "sucesso": b["sucesso"], "sucesso_ic95": b["sucesso_ic95"],
                   "passos_ep": b["passos"], "passos_ep_ic95": b["passos_ic95"], "toca_destrocos": b["toca"], "toca_destrocos_ic95": b["toca_ic95"]})
df_res = pd.DataFrame(linhas)
df_res.to_csv(RES / "resultado_final_resumo.csv", index=False)
print(df_res[["agente", "config", "ret_desc", "ret_desc_ic95", "sucesso", "passos_ep", "toca_destrocos"]].round(2).to_string(index=False))

# %% [markdown]
# ## 6. Curvas de aprendizado
# Retorno descontado e taxa de sucesso da política gulosa em função dos passos de interação (média ± IC95% entre as sementes das runs finais).

# %%
def tabela_curvas(runs):
    """Curvas em formato longo. Os passos reais (múltiplos do tamanho do lote de cada algoritmo) são arredondados ao ponto
    nominal da grade de avaliação (orçamento / n_pontos), para que as sementes e os algoritmos fiquem alinhados."""
    linhas = []
    for r in runs:
        freq = r["passos"] / r["n_pontos"]
        pontos = {}
        for c in r["curva"]:
            pontos[int(round(c["passos"] / freq))] = c                    # o último ponto do mesmo k (fim do treino) prevalece
        for k, c in sorted(pontos.items()):
            linhas.append({"algo": r["algo"], "config": r["config"], "semente": r["semente"], "passos": int(round(k * freq)),
                           "passos_reais": c["passos"], "ret": c["ret"], "ret_desc": c["ret_desc"], "sucesso": c["sucesso"],
                           "passos_ep": c["passos_ep"]})
    return pd.DataFrame(linhas)


curv_f = tabela_curvas(runs_f)
curv_f.to_csv(RES / "curvas_finais.csv", index=False)
tabela_curvas(runs_b).to_csv(RES / "curvas_busca.csv", index=False)
# registros dos episódios de treino das runs finais (passo, retorno, comprimento, sucesso, toques)
np.savez_compressed(RES / "curvas_treino_finais.npz", **{p.stem.replace("_treino", ""): np.load(p)["treino"] for p in sorted(RUNS.glob("final_*_treino.npz"))
                                                         if p.stem.replace("_treino", "") in {s["id"] for s in specs_final}})


def banda(ax, df, metrica, algo, **kw):
    g = df[df.algo == algo].groupby("passos")[metrica]
    m, ic = g.mean(), g.apply(lambda v: B.ic95(v) if len(v) > 1 else 0.0)
    ax.plot(m.index, m.values, color=COR_ALGO[algo], label=NOME_ALGO[algo], **kw)
    ax.fill_between(m.index, m - ic.fillna(0), m + ic.fillna(0), color=COR_ALGO[algo], alpha=0.15, lw=0)


fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.3))
for a in ALGOS:
    banda(axs[0], curv_f, "ret_desc", a)
    banda(axs[1], curv_f, "sucesso", a)
for nome, ls in (("tabular", "--"), ("heuristica", ":")):
    axs[0].axhline(BASE[nome]["ret_desc"], color=COR_REF[nome], ls=ls, lw=1.3, label={"tabular": "tabular discretizada", "heuristica": "heurística"}[nome])
    axs[1].axhline(BASE[nome]["sucesso"], color=COR_REF[nome], ls=ls, lw=1.3)
axs[0].axhline(BASE["aleatoria"]["ret_desc"], color=COR_REF["aleatoria"], ls="-.", lw=1.0, label="aleatória")
axs[0].set_xlabel("passos de interação"); axs[0].set_ylabel("retorno descontado (γ = 0,99)"); axs[0].set_title("Retorno da política gulosa")
axs[1].set_xlabel("passos de interação"); axs[1].set_ylabel("taxa de sucesso"); axs[1].set_ylim(-0.03, 1.05); axs[1].set_title("Taxa de sucesso")
for ax in axs:
    eixo_passos(ax)
# legenda única fora das áreas de plotagem (abaixo), para não cobrir as curvas
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(l), fontsize=7.5, handlelength=2.2, columnspacing=1.4)
salvar_fig(fig, "fig_curvas_aprendizado.png")

# %% [markdown]
# ## 7. Resultado final, busca completa e trajetórias

# %%
# (a) resultado final: barras com IC95% entre sementes + pontos por semente
fig, axs = plt.subplots(1, 3, figsize=(9.2, 3.2))
ordem = ALGOS + ["tabular", "heuristica", "aleatoria"]
rotulos = ["DQN", "PPO", "A2C", "Sarsa", "tabular\ndiscret.", "heurística", "aleatória"]
cores = [COR_ALGO[a] for a in ALGOS] + [COR_REF["tabular"], COR_REF["heuristica"], COR_REF["aleatoria"]]
for ax, (m, tit) in zip(axs, (("ret_desc", "Retorno descontado"), ("sucesso", "Taxa de sucesso"), ("toca_destrocos", "Fração de episódios que toca destroços"))):
    sub = df_res.set_index("agente").loc[ordem]
    ic = sub[m + "_ic95"].fillna(0).values
    erro = np.vstack([np.minimum(ic, sub[m].values), np.minimum(ic, 1 - sub[m].values)]) if m != "ret_desc" else ic   # frações ficam em [0, 1]
    ax.bar(range(len(ordem)), sub[m], yerr=erro, color=cores, alpha=0.85, capsize=3, error_kw=dict(lw=1, ecolor="#333333"))
    if m != "ret_desc":
        ax.set_ylim(0, 1.05)
    for i, a in enumerate(ALGOS):
        ax.scatter([i] * len(SEM_FINAL), df_final[df_final.algo == a][m], s=9, color="white", edgecolor="#222222", lw=0.6, zorder=5)
    ax.set_xticks(range(len(ordem))); ax.set_xticklabels(rotulos, fontsize=7, rotation=35, ha="right"); ax.set_title(tit, fontsize=9)
salvar_fig(fig, "fig_resultado_final.png")

# (b) busca completa: todas as configurações (média ± IC95% entre sementes)
fig, axs = plt.subplots(1, 4, figsize=(9.6, 3.0), sharex=False)
for ax, a in zip(axs, ALGOS):
    sub = res_busca[res_busca.algo == a].reset_index(drop=True)
    y = np.arange(len(sub))[::-1]
    ax.barh(y, sub.ret_desc, xerr=sub.ret_desc_ic95.fillna(0), color=[COR_ALGO[a] if e else COR["claro"] for e in sub.escolhida],
            edgecolor=COR_ALGO[a], lw=0.8, capsize=2, error_kw=dict(lw=0.8, ecolor="#333333"))
    ax.set_yticks(y); ax.set_yticklabels(["padrão SB3" if c.endswith("_padrao") else c.split("_", 1)[1] for c in sub.config], fontsize=6.5)
    ax.axvline(BASE["tabular"]["ret_desc"], color=COR_REF["tabular"], ls="--", lw=1)
    ax.set_title({"SARSA": "Sarsa"}.get(a, a), fontsize=9); ax.set_xlabel("retorno descontado", fontsize=8)
salvar_fig(fig, "fig_busca_hiperparametros.png")


# (c) trajetórias de um episódio (mesma semente) das melhores sementes + heurística
def trajetoria(politica, semente=10_001):
    env = B.RescueBoatContEnv(**PARAMS_ENV)
    obs, _ = env.reset(seed=semente)
    pts = [(env.x, env.y)]
    fim = False
    while not fim:
        obs, r, te, tr, info = env.step(politica(env, obs))
        pts.append((env.x, env.y))
        fim = te or tr
    return np.array(pts), info["sucesso"], env.passos, env


painel = [("heuristica", "Heurística (BFS)", bases["heuristica"])]
try:
    from stable_baselines3 import A2C, DQN, PPO
    cls = {"DQN": DQN, "PPO": PPO, "A2C": A2C}
    for a in ("DQN", "PPO", "A2C"):
        melhor = df_final[df_final.algo == a].sort_values("ret_desc", ascending=False).iloc[0]
        m = cls[a].load(str(MODELOS / f"final_{a}_{melhor.config}_s{int(melhor.semente)}"), device="cpu")
        painel.append((a, f"{a} (semente {int(melhor.semente)})", B.politica_sb3(m)))
except Exception as e:
    print("Trajetórias dos modelos treinados indisponíveis:", repr(e))
fig, axs = plt.subplots(1, len(painel), figsize=(2.9 * len(painel), 3.1))
axs = np.atleast_1d(axs)
for ax, (chave, tit, pol) in zip(axs, painel):
    pts, ok, n, env_t = trajetoria(pol)
    env_t.reset(seed=10_001)
    env_t.desenha(ax=ax)
    ax.plot(pts[:, 0], pts[:, 1], color=COR_ALGO.get(chave, COR["cinza"]), lw=1.6, zorder=6, clip_on=False)   # sem recorte na borda do mapa
    ax.scatter(pts[-1:, 0], pts[-1:, 1], color=COR["vermelho"], s=18, zorder=7, clip_on=False)
    ax.set_title(f"{tit}\n{n} passos, {'sucesso' if ok else 'sem sucesso'}", fontsize=8)
salvar_fig(fig, "fig_trajetorias.png", topo=0.80)          # títulos de 2 linhas: deixa espaço para a logo

# (d) o cenário e a dinâmica da correnteza
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.1), gridspec_kw=dict(width_ratios=[1, 1.5]))
e0 = B.RescueBoatContEnv(**PARAMS_ENV)
e0.reset(seed=0)
e0.desenha(ax=axs[0])
axs[0].set_title("Cenário (H = abrigo, P = pessoas)", fontsize=9)
for k in range(5):
    e = B.RescueBoatContEnv(**PARAMS_ENV)
    e.reset(seed=k)
    I = [e.inten]
    for _ in range(100):
        e.step(0)
        I.append(e.inten)
    axs[1].plot(I, lw=1.2, color=COR["azul2"], alpha=0.8)
axs[1].set_xlabel("passo"); axs[1].set_ylabel("intensidade da correnteza I"); axs[1].set_ylim(-0.02, 1.02)
axs[1].set_title("Elemento dinâmico: intensidade da correnteza (5 sementes)", fontsize=9)
salvar_fig(fig, "fig_ambiente.png")

# %% [markdown]
# ## 8. Hipóteses
# **H1** (eficiência de amostras): DQN tem maior área sob a curva de aprendizado (retorno descontado médio ao longo dos passos) que PPO e A2C.
# **H2** (estabilidade): PPO tem menor dispersão entre sementes que A2C no retorno descontado final. **H3**: a aproximação de função
# não alcança a linha de base tabular discretizada nem a heurística. Testes: *t* de Welch (H1), razão de desvios e teste de Levene (H2)
# e *t* de uma amostra contra o valor da linha de base (H3); com poucas sementes o poder é baixo e os veredictos são cautelosos.

# %%
def auc(df, a, s):
    d = df[(df.algo == a) & (df.semente == s)].sort_values("passos")
    return float(np.trapezoid(d.ret_desc.values, d.passos.values) / (d.passos.max() - d.passos.min()))


auc_tab = pd.DataFrame([{"algo": a, "semente": s, "auc_ret_desc": auc(curv_f, a, s)} for a in ALGOS for s in SEM_FINAL])
auc_tab.to_csv(RES / "auc_curvas.csv", index=False)
linhas = []
v = lambda a: auc_tab[auc_tab.algo == a].auc_ret_desc.values
for outro in ("PPO", "A2C"):
    t, p = stats.ttest_ind(v("DQN"), v(outro), equal_var=False)
    dif = v("DQN").mean() - v(outro).mean()
    ver = "confirmada" if dif > 0 and p < 0.05 else ("inconclusiva (direção favorável, p ≥ 0,05)" if dif > 0 else
                                                      "refutada (direção contrária)" if dif < 0 else "inconclusiva (sem diferença)")
    linhas.append({"hipotese": f"H1: DQN vs {outro} (AUC do retorno descontado)", "diferenca": dif, "estatistica": t, "p": p, "veredito": ver})
sd = lambda a: df_final[df_final.algo == a].ret_desc.std(ddof=1)
lev, p_lev = stats.levene(df_final[df_final.algo == "PPO"].ret_desc, df_final[df_final.algo == "A2C"].ret_desc)
ver = ("confirmada" if sd("PPO") < sd("A2C") and p_lev < 0.05 else
       "inconclusiva (direção favorável, p ≥ 0,05)" if sd("PPO") < sd("A2C") else
       "refutada (direção contrária)" if sd("PPO") > sd("A2C") else "inconclusiva (sem diferença)")
linhas.append({"hipotese": "H2: desvio entre sementes, PPO vs A2C (retorno descontado final)", "diferenca": sd("PPO") - sd("A2C"),
               "estatistica": lev, "p": p_lev, "veredito": ver})
for a in ALGOS:
    for ref in ("tabular", "heuristica"):
        x = df_final[df_final.algo == a].ret_desc.values
        t, p = stats.ttest_1samp(x, BASE[ref]["ret_desc"])
        dif = x.mean() - BASE[ref]["ret_desc"]
        ver = ("confirmada (abaixo da referência)" if dif < 0 and p < 0.05 else "refutada (acima da referência)" if dif > 0 and p < 0.05
               else "inconclusiva (sem diferença significativa)")
        linhas.append({"hipotese": f"H3: {a} vs {ref} (retorno descontado final)", "diferenca": dif, "estatistica": t, "p": p, "veredito": ver})
df_hip = pd.DataFrame(linhas)
df_hip.to_csv(RES / "hipoteses.csv", index=False)
print(df_hip.round(4).to_string(index=False))

# %% [markdown]
# ## 9. Apêndice: todas as configurações testadas
# Tabela completa da busca (média e desvio-padrão entre 3 sementes). A coluna `escolhida` marca a configuração reexecutada na Seção 5.

# %%
tab_apendice = res_busca.copy()
tab_apendice["hiperparametros"] = [json.dumps(CONFIGS[r.algo][r.config], ensure_ascii=False) for r in tab_apendice.itertuples()]
print(tab_apendice[["algo", "config", "ret_desc", "ret_desc_dp", "sucesso", "toca_destrocos", "escolhida", "hiperparametros"]].round(2).to_string(index=False))
tab_apendice.to_csv(RES / "busca_resumo_com_hp.csv", index=False)
INFO["configs_melhores"] = melhores
INFO["tempos_s"]["total"] = round(time.time() - T_INICIO, 1)
salvar_info()
print(f"Tempo total: {time.time() - T_INICIO:.0f} s | resultados em {RES} | figuras em {FIG}")
