"""Barco de resgate com estados contínuos: ambiente, referências do Trabalho 2, avaliação e fábrica de modelos.

Trabalho 3 de Aprendizado por Reforço (PPGCA/Unisinos, 2026/2). O módulo reúne:

(a) ``RescueBoatContEnv``: ambiente Gymnasium com posição contínua, correnteza dinâmica e ruído de movimento;
(b) ``RescueBoatEnv`` e ``iteracao_valor``: o MDP discreto do Trabalho 2, recriado de forma compacta, usado como referência;
(c) políticas de referência: tabular discretizada (política ótima do T2 aplicada à célula da posição), aleatória e heurística;
(d) funções de avaliação (retorno descontado e não descontado, sucesso, passos, fração que toca destroços);
(e) fábrica de modelos do stable-baselines3 (DQN, PPO, A2C) e rotina de treino com avaliação periódica;
(f) Sarsa semi-gradiente linear com *tile coding* implementado à mão (Sutton e Barto, caps. 9 e 10; Aula 13).

Convenções geométricas
----------------------
A posição é ``(x, y)`` em ``[0, 6)²``. ``x`` cresce para leste (coluna) e ``y`` cresce para o sul (linha do mapa, de cima para
baixo), de modo que a célula ``(linha i, coluna j)`` do mapa do Trabalho 2 ocupa ``x∈[j, j+1)`` e ``y∈[i, i+1)``.
Ações: 0 = N (``y`` diminui), 1 = S (``y`` aumenta), 2 = L (``x`` aumenta), 3 = O (``x`` diminui).

Dinâmica de um passo (ordem fixa; a mesma no laboratório em JavaScript)
------------------------------------------------------------------------
1. Sorteiam-se quatro normais padrão ``(zx, zy)`` e ``(zc, zi)`` (dois pares de Box-Muller, ver abaixo).
2. Movimento: ``x1 = x + passo·dx + σ_mov·zx`` e ``y1 = y + passo·dy + σ_mov·zy``, com recorte em ``[0, 6-1e-9]``.
   Se ``(x1, y1)`` cai dentro de um bloco (célula ``#``), o barco não se move nesta fase (``x1, y1 = x, y``).
3. Correnteza: se ``y1`` está na faixa ``[2, 3)``, ``y2 = y1 + v_max·I + σ_corr·zc`` (arrasto para o sul, com ``I`` a
   intensidade atual da correnteza); recorte e colisão com blocos como na fase anterior. Fora da faixa, ``y2 = y1``.
4. A intensidade evolui como um processo AR(1) recortado em [0, 1]: ``I ← clip(I + θ·(μ − I) + σ_I·zi, 0, 1)``.
   A intensidade é observável, de modo que o processo é markoviano na observação.
5. Eventos: embarque (pessoa ilhada a distância ≤ ``r_embarque`` e vaga a bordo, na ordem P1, P2, P3), desembarque no
   abrigo (distância ≤ ``r_abrigo`` do centro de H e pelo menos uma pessoa a bordo: todas descem, +50 cada), conclusão (+100).
6. Recompensa: −1 por passo; ao *entrar* num disco de destroços (fora no início do passo, dentro no fim) a recompensa do
   passo é −10 no lugar de −1 (mesma convenção do Trabalho 2); +50 por pessoa entregue; +100 ao entregar a última.
7. ``terminated`` quando as 3 pessoas foram entregues; ``truncated`` ao atingir ``max_passos`` sem terminar
   (``info["TimeLimit.truncated"]`` sinaliza o truncamento para o SB3 fazer *bootstrap*).

Gerador pseudoaleatório portável
--------------------------------
Todo o ruído vem de um gerador próprio, implementável de forma idêntica em JavaScript:

* ``mulberry32`` (estado de 32 bits)::

      s = (s + 0x6D2B79F5) mod 2^32
      t = s
      t = imul(t ^ (t >>> 15), t | 1)
      t = t ^ (t + imul(t ^ (t >>> 7), t | 61))        (soma e XOR módulo 2^32)
      u = ((t ^ (t >>> 14)) mod 2^32) / 2^32            em [0, 1)

  em que ``imul`` é o produto de inteiros de 32 bits módulo 2^32 (``Math.imul`` em JavaScript).
* Box-Muller (par de normais com 2 uniformes consecutivos ``u1``, ``u2``)::

      r  = sqrt(−2·ln(1 − u1))          (1 − u1 ∈ (0, 1], sem log(0))
      z0 = r·cos(2π·u2);   z1 = r·sin(2π·u2)

* A semente de ``reset(seed=s)`` é ``s mod 2^32``; sem semente, o primeiro ``reset`` sorteia uma do ``np_random`` do
  Gymnasium e os seguintes continuam o mesmo fluxo. No ``reset`` consome-se 1 par (apenas ``z0`` é usado) para a
  intensidade inicial ``I0 = clip(μ + σ_0·z0, 0, 1)``. Cada ``step`` consome exatamente 2 pares (4 uniformes).

As diferenças entre linguagens limitam-se às funções transcendentais (``log``, ``cos``, ``sin``), que podem diferir na
última casa decimal; a comparação com a trajetória de referência deve usar tolerância (~1e-9).
"""
from __future__ import annotations

import itertools
import math
import os
import time
from collections import defaultdict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# =============================================================================
# Mapa e parâmetros
# =============================================================================
MAPA_PADRAO = [
    "P..P..",
    ".#..#P",
    "~~~~~~",
    "..xx..",
    ".#..#.",
    "..H...",
]
NOMES_ACOES = ("N", "S", "L", "O")
DX = (0.0, 0.0, 1.0, -1.0)
DY = (-1.0, 1.0, 0.0, 0.0)
DELTA_CELULA = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}        # (linha, coluna), como no Trabalho 2

#: Parâmetros calibrados no piloto (ver resultados/piloto_calibracao.md).
PARAMS_PADRAO = dict(
    passo=0.7,           # deslocamento por ação (células)
    sigma_mov=0.05,      # desvio-padrão do ruído gaussiano de movimento (células)
    r_embarque=0.8,      # raio de embarque ao redor do centro da célula de cada pessoa
    r_abrigo=0.9,        # raio de desembarque ao redor do centro do abrigo
    r_destrocos=0.5,     # raio dos discos de destroços
    corr_vmax=0.5,       # arrasto máximo da correnteza por passo (células, quando I = 1)
    corr_sigma=0.05,     # ruído do arrasto (células)
    corr_media=0.5,      # média de longo prazo da intensidade μ
    corr_theta=0.1,      # velocidade de reversão à média θ
    corr_sigma_i=0.12,   # ruído da intensidade σ_I
    corr_sigma_0=0.25,   # dispersão da intensidade inicial σ_0
    max_passos=100,      # truncamento
    capacidade=2,
    r_passo=-1.0, r_destrocos_rec=-10.0, r_pessoa=50.0, r_final=100.0,
)
_EPS = 1e-9
_M32 = 0xFFFFFFFF
_DOIS_PI = 2.0 * math.pi


# =============================================================================
# Gerador portável
# =============================================================================
class Mulberry32:
    """Gerador ``mulberry32`` + Box-Muller (ver docstring do módulo). ``par()`` devolve duas normais padrão."""
    __slots__ = ("s",)

    def __init__(self, semente: int = 0):
        self.s = int(semente) & _M32

    def u32(self) -> int:
        self.s = (self.s + 0x6D2B79F5) & _M32
        t = self.s
        t = ((t ^ (t >> 15)) * (t | 1)) & _M32
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & _M32)) & _M32)) & _M32
        return (t ^ (t >> 14)) & _M32

    def uniforme(self) -> float:
        return self.u32() / 4294967296.0

    def par(self):
        u1 = self.uniforme()
        u2 = self.uniforme()
        r = math.sqrt(-2.0 * math.log(1.0 - u1))
        ang = _DOIS_PI * u2
        return r * math.cos(ang), r * math.sin(ang)


# =============================================================================
# Ambiente contínuo
# =============================================================================
class RescueBoatContEnv(gym.Env):
    """Barco de resgate 6×6 com posição contínua, correnteza dinâmica e ruído de movimento.

    Observação (``Box`` float32, 13 componentes em [0, 1]): ``x/6``, ``y/6``, situação de cada pessoa em *one-hot*
    (ilhada, a bordo, salva; 3×3 = 9 componentes), ``a_bordo/capacidade`` e intensidade ``I`` da correnteza.
    Não há atributos de direção ou distância ao alvo. Ações: ``Discrete(4)`` (N, S, L, O).
    """
    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 6}
    DIM_OBS = 13

    def __init__(self, mapa=MAPA_PADRAO, render_mode=None, **kw):
        super().__init__()
        desconhecidos = set(kw) - set(PARAMS_PADRAO)
        if desconhecidos:
            raise TypeError(f"parâmetros desconhecidos: {sorted(desconhecidos)}")
        self.params = {**PARAMS_PADRAO, **kw}
        p = self.params
        self.mapa = [str(l) for l in mapa]
        self.nl, self.nc = len(self.mapa), len(self.mapa[0])
        self.render_mode = render_mode
        cel = [(i, j) for i in range(self.nl) for j in range(self.nc)]
        self.bloqueios = [(i, j) for i, j in cel if self.mapa[i][j] == "#"]
        self._bloq = [[self.mapa[i][j] == "#" for j in range(self.nc)] for i in range(self.nl)]
        linhas_corr = sorted({i for i, j in cel if self.mapa[i][j] == "~"})
        self.faixa = (float(linhas_corr[0]), float(linhas_corr[-1] + 1)) if linhas_corr else (0.0, 0.0)
        self.destrocos = [(j + 0.5, i + 0.5) for i, j in cel if self.mapa[i][j] == "x"]      # centros (x, y)
        ab = next((i, j) for i, j in cel if self.mapa[i][j] == "H")
        self.abrigo = (ab[1] + 0.5, ab[0] + 0.5)
        self.pessoas = [(j + 0.5, i + 0.5) for i, j in cel if self.mapa[i][j] == "P"]       # ordem de leitura do mapa
        self.n_pes = len(self.pessoas)
        self.cap = int(p["capacidade"])
        self._lim_x, self._lim_y = self.nc - _EPS, self.nl - _EPS
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.DIM_OBS,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self._rng = Mulberry32(0)
        self._semeado = False
        self.x = self.abrigo[0]
        self.y = self.abrigo[1]
        self.status = [0] * self.n_pes
        self.aboard = 0
        self.inten = 0.5
        self.passos = 0
        self.toques = 0
        self._entregues = 0

    # ---- utilidades --------------------------------------------------------
    def _obs(self):
        o = [self.x / self.nc, self.y / self.nl]
        for s in self.status:
            o += [1.0 if s == 0 else 0.0, 1.0 if s == 1 else 0.0, 1.0 if s == 2 else 0.0]
        o += [self.aboard / self.cap, self.inten]
        return np.array(o, dtype=np.float32)

    def _info(self, **extra):
        d = {"intensidade": self.inten, "status": tuple(self.status), "toques": self.toques,
             "sucesso": all(s == 2 for s in self.status)}
        d.update(extra)
        return d

    @property
    def celula(self):
        """Célula discreta ``(linha, coluna)`` que contém a posição atual."""
        return int(self.y), int(self.x)

    # ---- API Gymnasium -----------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = Mulberry32(int(seed))
            self._semeado = True
        elif not self._semeado:
            self._rng = Mulberry32(int(self.np_random.integers(0, 2 ** 32)))
            self._semeado = True
        p = self.params
        self.x, self.y = self.abrigo
        self.status = [0] * self.n_pes
        self.aboard = 0
        z0, _ = self._rng.par()
        self.inten = min(1.0, max(0.0, p["corr_media"] + p["corr_sigma_0"] * z0))
        self.passos = 0
        self.toques = 0
        self._entregues = 0
        return self._obs(), self._info()

    def step(self, action):
        p = self.params
        a = int(action)
        rng = self._rng
        zx, zy = rng.par()
        zc, zi = rng.par()
        x0, y0 = self.x, self.y
        lx, ly = self._lim_x, self._lim_y
        bl = self._bloq
        # fase 1: movimento comandado + ruído
        x1 = x0 + p["passo"] * DX[a] + p["sigma_mov"] * zx
        y1 = y0 + p["passo"] * DY[a] + p["sigma_mov"] * zy
        x1 = 0.0 if x1 < 0.0 else (lx if x1 > lx else x1)
        y1 = 0.0 if y1 < 0.0 else (ly if y1 > ly else y1)
        if bl[int(y1)][int(x1)]:
            x1, y1 = x0, y0
        # fase 2: correnteza (arrasto para o sul)
        y2 = y1
        if self.faixa[0] <= y1 < self.faixa[1]:
            y2 = y1 + p["corr_vmax"] * self.inten + p["corr_sigma"] * zc
            y2 = 0.0 if y2 < 0.0 else (ly if y2 > ly else y2)
            if bl[int(y2)][int(x1)]:
                y2 = y1
        # intensidade (processo AR(1) recortado)
        i = self.inten + p["corr_theta"] * (p["corr_media"] - self.inten) + p["corr_sigma_i"] * zi
        self.inten = 0.0 if i < 0.0 else (1.0 if i > 1.0 else i)
        self.x, self.y = x1, y2
        # eventos
        r = p["r_passo"]
        rd2 = p["r_destrocos"] ** 2
        for cx, cy in self.destrocos:
            if (x1 - cx) ** 2 + (y2 - cy) ** 2 <= rd2 and (x0 - cx) ** 2 + (y0 - cy) ** 2 > rd2:
                r = p["r_destrocos_rec"]
                self.toques += 1
                break
        re2 = p["r_embarque"] ** 2
        for k in range(self.n_pes):
            if self.status[k] == 0 and self.aboard < self.cap:
                px, py = self.pessoas[k]
                if (x1 - px) ** 2 + (y2 - py) ** 2 <= re2:
                    self.status[k] = 1
                    self.aboard += 1
        if self.aboard > 0 and (x1 - self.abrigo[0]) ** 2 + (y2 - self.abrigo[1]) ** 2 <= p["r_abrigo"] ** 2:
            for k in range(self.n_pes):
                if self.status[k] == 1:
                    self.status[k] = 2
                    r += p["r_pessoa"]
            self.aboard = 0
        terminated = all(s == 2 for s in self.status)
        if terminated:
            r += p["r_final"]
        self.passos += 1
        truncated = (not terminated) and self.passos >= p["max_passos"]
        info = self._info()
        if truncated:
            info["TimeLimit.truncated"] = True
        return self._obs(), r, terminated, truncated, info

    # ---- renderização ------------------------------------------------------
    def render(self):
        if self.render_mode == "rgb_array":
            return self.desenha()
        g = [list(l) for l in self.mapa]
        for k, (px, py) in enumerate(self.pessoas):
            if self.status[k] != 0:
                g[int(py)][int(px)] = "."
        g[int(self.y)][int(self.x)] = "B"
        nomes = {0: "ilhada", 1: "a bordo", 2: "salva"}
        return ("\n".join("".join(l) for l in g) + f"\n(x={self.x:.2f}, y={self.y:.2f})  correnteza I={self.inten:.2f}\n"
                + " | ".join(f"P{k + 1}: {nomes[s]}" for k, s in enumerate(self.status)))

    def desenha(self, ax=None, tamanho=3.6):
        """Imagem do cenário (matplotlib). Sem ``ax`` devolve um array RGB; com ``ax`` desenha nele."""
        import matplotlib
        from matplotlib.patches import Circle, Polygon, Rectangle
        proprio = ax is None
        if proprio:
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(tamanho, tamanho))
        cores = {".": "#EAF3FB", "~": "#9CC9EC", "x": "#EAF3FB", "#": "#5B5F6B", "H": "#BFE3C9", "P": "#EAF3FB"}
        for i in range(self.nl):
            for j in range(self.nc):
                ax.add_patch(Rectangle((j, i), 1, 1, fc=cores[self.mapa[i][j]], ec="white", lw=1.5))
        for cx, cy in self.destrocos:
            ax.add_patch(Circle((cx, cy), self.params["r_destrocos"], fc="#C9A27E", ec="#7A5535", lw=1))
        ax.text(self.abrigo[0], self.abrigo[1], "H", ha="center", va="center", color="#2E7D5B", fontsize=12, fontweight="bold")
        for k, (px, py) in enumerate(self.pessoas):
            if self.status[k] == 0:
                ax.text(px, py, f"P{k + 1}", ha="center", va="center", color="#C0392B", fontsize=10, fontweight="bold")
        bx, by = self.x, self.y
        ax.add_patch(Polygon([(bx - .28, by + .08), (bx + .28, by + .08), (bx + .17, by + .24), (bx - .17, by + .24)], fc="#7A4B1F", ec="none", zorder=5))
        ax.add_patch(Polygon([(bx - .02, by + .05), (bx - .02, by - .30), (bx + .22, by + .05)], fc="white", ec="#1D2369", lw=.8, zorder=5))
        ax.set_xlim(0, self.nc)
        ax.set_ylim(self.nl, 0)
        ax.set_aspect("equal")
        ax.axis("off")
        if proprio:
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            plt.close(fig)
            return img


# =============================================================================
# MDP discreto do Trabalho 2 (referência)
# =============================================================================
class RescueBoatEnv(gym.Env):
    """MDP tabular do Trabalho 2 (6×6, células discretas) com o modelo exposto em ``env.P[s][a]``.

    O estado é o par (célula do barco, situação das 3 pessoas) com situação ∈ {0 ilhada, 1 a bordo, 2 salva}, no máximo
    ``capacidade`` a bordo. Na correnteza, com probabilidade ``p_corr`` a ação é substituída por um empurrão para o sul.
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, mapa=MAPA_PADRAO, p_corr=0.2, capacidade=2, max_passos=100,
                 r_passo=-1.0, r_destrocos=-10.0, r_pessoa=50.0, r_final=100.0):
        super().__init__()
        self.mapa = [str(l) for l in mapa]
        self.nl, self.nc = len(self.mapa), len(self.mapa[0])
        self.p_corr, self.cap, self.max_passos = p_corr, capacidade, max_passos
        self.r_passo, self.r_destrocos, self.r_pessoa, self.r_final = r_passo, r_destrocos, r_pessoa, r_final
        cel = [(i, j) for i in range(self.nl) for j in range(self.nc)]
        self.bloqueios = {c for c in cel if self.mapa[c[0]][c[1]] == "#"}
        self.corr = {c for c in cel if self.mapa[c[0]][c[1]] == "~"}
        self.destrocos = {c for c in cel if self.mapa[c[0]][c[1]] == "x"}
        self.abrigo = next(c for c in cel if self.mapa[c[0]][c[1]] == "H")
        self.pessoas = [c for c in cel if self.mapa[c[0]][c[1]] == "P"]
        self.n_pes = len(self.pessoas)
        self.celulas = [c for c in cel if c not in self.bloqueios]
        self.idx_cel = {c: k for k, c in enumerate(self.celulas)}
        self.situacoes = [e for e in itertools.product(range(3), repeat=self.n_pes) if sum(x == 1 for x in e) <= self.cap]
        self.idx_sit = {e: k for k, e in enumerate(self.situacoes)}
        self.nS, self.nA = len(self.celulas) * len(self.situacoes), 4
        self.observation_space, self.action_space = spaces.Discrete(self.nS), spaces.Discrete(self.nA)
        self.s0 = self.codifica(self.abrigo, (0,) * self.n_pes)
        self.P = self._constroi_P()
        self.s, self.passos = self.s0, 0

    def codifica(self, cel, sit):
        return self.idx_cel[tuple(cel)] * len(self.situacoes) + self.idx_sit[tuple(sit)]

    def decodifica(self, s):
        k, e = divmod(int(s), len(self.situacoes))
        return self.celulas[k], self.situacoes[e]

    def _mover(self, cel, d):
        n = (cel[0] + d[0], cel[1] + d[1])
        return cel if not (0 <= n[0] < self.nl and 0 <= n[1] < self.nc) or n in self.bloqueios else n

    def _resultado(self, sit, nova):
        r = self.r_destrocos if nova in self.destrocos else self.r_passo
        sit = list(sit)
        for i, pos in enumerate(self.pessoas):
            if pos == nova and sit[i] == 0 and sum(x == 1 for x in sit) < self.cap:
                sit[i] = 1
        if nova == self.abrigo:
            for i in range(self.n_pes):
                if sit[i] == 1:
                    sit[i] = 2
                    r += self.r_pessoa
        fim = all(x == 2 for x in sit)
        if fim:
            r += self.r_final
        return nova, tuple(sit), r, fim

    def _constroi_P(self):
        P = {}
        for s in range(self.nS):
            cel, sit = self.decodifica(s)
            P[s] = {}
            for a in range(self.nA):
                if all(x == 2 for x in sit):
                    P[s][a] = [(1.0, s, 0.0, True)]
                    continue
                saidas = [(1.0, DELTA_CELULA[a])] if cel not in self.corr else \
                    [(1 - self.p_corr, DELTA_CELULA[a]), (self.p_corr, DELTA_CELULA[1])]
                acum = defaultdict(float)
                for pr, d in saidas:
                    nova, nsit, r, fim = self._resultado(sit, self._mover(cel, d))
                    acum[(self.codifica(nova, nsit), r, fim)] += pr
                P[s][a] = [(pr, s2, r, f) for (s2, r, f), pr in acum.items()]
        return P

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.s, self.passos = self.s0, 0
        return self.s, {}

    def step(self, a):
        trans = self.P[self.s][int(a)]
        i = self.np_random.choice(len(trans), p=[t[0] for t in trans])
        _, s2, r, fim = trans[i]
        self.s, self.passos = s2, self.passos + 1
        return s2, r, fim, (not fim) and self.passos >= self.max_passos, {}


def iteracao_valor(env: RescueBoatEnv, gamma=0.99, theta=1e-10, max_it=100000):
    """Iteração de Valor vetorizada (Sutton e Barto, cap. 4). Devolve ``(V, política, Q, n_iterações)``."""
    nS, nA = env.nS, env.nA
    K = max(len(env.P[s][a]) for s in range(nS) for a in range(nA))
    prob = np.zeros((nS, nA, K))
    prox = np.zeros((nS, nA, K), dtype=int)
    rec = np.zeros((nS, nA, K))
    nf = np.ones((nS, nA, K))                                  # 1 - terminou
    for s in range(nS):
        for a in range(nA):
            for k, (pr, s2, r, f) in enumerate(env.P[s][a]):
                prob[s, a, k], prox[s, a, k], rec[s, a, k], nf[s, a, k] = pr, s2, r, 0.0 if f else 1.0
    V = np.zeros(nS)
    for it in range(max_it):
        Q = (prob * (rec + gamma * V[prox] * nf)).sum(2)
        Vn = Q.max(1)
        d = np.abs(Vn - V).max()
        V = Vn
        if d < theta:
            break
    Q = (prob * (rec + gamma * V[prox] * nf)).sum(2)
    return V, Q.argmax(1), Q, it + 1


# =============================================================================
# Políticas de referência: callables ``f(env, obs) -> ação``
# =============================================================================
def politica_aleatoria(semente=0):
    rng = np.random.default_rng(semente)
    return lambda env, obs: int(rng.integers(4))


class PoliticaTabularDiscretizada:
    """Política ótima do T2 (Iteração de Valor) consultada pela célula que contém a posição contínua.

    Serve de linha de base: o agente conhece o mapa e o modelo do T2, mas ignora a posição dentro da célula, o ruído e a
    intensidade da correnteza (que no T2 era um empurrão discreto com probabilidade 0,2).
    """

    def __init__(self, gamma=0.99, p_corr=0.2):
        self.env = RescueBoatEnv(p_corr=p_corr)
        self.V, self.pi, self.Q, _ = iteracao_valor(self.env, gamma=gamma)
        self.v0 = float(self.V[self.env.s0])

    def __call__(self, env, obs):
        s = self.env.codifica(env.celula, env.status)
        return int(self.pi[s])


class PoliticaHeuristica:
    """Heurística de caminho mais curto: nunca usa treino.

    Escolhe o alvo (pessoa ilhada mais próxima enquanto houver vaga; abrigo se o barco está cheio ou se não há ilhadas),
    calcula por BFS a distância em células até o alvo evitando blocos *e* destroços e segue em direção ao centro da
    próxima célula do caminho. Ignora a correnteza (apenas insiste no rumo).
    """

    def __init__(self, env: RescueBoatContEnv):
        self.nl, self.nc = env.nl, env.nc
        livre = {(i, j) for i in range(self.nl) for j in range(self.nc) if env.mapa[i][j] != "#"
                 and env.mapa[i][j] != "x"}
        self.livre = livre
        self.alvos = {("P", k): (int(py), int(px)) for k, (px, py) in enumerate(env.pessoas)}
        self.alvos["H"] = (int(env.abrigo[1]), int(env.abrigo[0]))
        self.dist = {k: self._bfs(c) for k, c in self.alvos.items()}

    def _bfs(self, alvo):
        d = {alvo: 0}
        fila = [alvo]
        for c in fila:
            for dl, dc in DELTA_CELULA.values():
                n = (c[0] + dl, c[1] + dc)
                if n in self.livre and n not in d:
                    d[n] = d[c] + 1
                    fila.append(n)
        return d

    def __call__(self, env, obs):
        cel = env.celula
        espera = [k for k, s in enumerate(env.status) if s == 0]
        if env.aboard >= env.cap or not espera:
            chave = "H"
        else:
            chave = min((("P", k) for k in espera), key=lambda c: self.dist[c].get(cel, 1e9))
        d = self.dist[chave]
        alvo = self.alvos[chave]
        if cel == alvo or cel not in d:
            tc = alvo
        else:
            tc = min((n for n in ((cel[0] + dl, cel[1] + dc) for dl, dc in DELTA_CELULA.values()) if n in d),
                     key=lambda n: d[n])
        ex, ey = tc[1] + 0.5 - env.x, tc[0] + 0.5 - env.y
        if abs(ex) > abs(ey):
            return 2 if ex > 0 else 3
        return 1 if ey > 0 else 0


def politica_sb3(modelo):
    """Adapta um modelo do SB3 (política determinística/gulosa) à interface ``f(env, obs)``."""
    return lambda env, obs: int(modelo.predict(obs, deterministic=True)[0])


# =============================================================================
# Avaliação
# =============================================================================
SEMENTE_AVAL = 10_000


def avaliar(politica, n_ep=100, semente=SEMENTE_AVAL, gamma=0.99, env=None, **params_env):
    """Executa ``n_ep`` episódios com sementes ``semente + e`` (mesmas para todos os agentes: comparação pareada).

    Devolve arrays por episódio: ``ret`` (não descontado), ``ret_desc`` (descontado com ``gamma``), ``sucesso``,
    ``passos`` e ``toca`` (1 se entrou em algum disco de destroços).
    """
    env = env or RescueBoatContEnv(**params_env)
    ret, retd, suc, pas, toc = [], [], [], [], []
    for e in range(n_ep):
        obs, info = env.reset(seed=semente + e)
        G = Gd = 0.0
        fator = 1.0
        fim = False
        while not fim:
            obs, r, term, trunc, info = env.step(politica(env, obs))
            G += r
            Gd += fator * r
            fator *= gamma
            fim = term or trunc
        ret.append(G)
        retd.append(Gd)
        suc.append(float(info["sucesso"]))
        pas.append(env.passos)
        toc.append(float(info["toques"] > 0))
    return {"ret": np.array(ret), "ret_desc": np.array(retd), "sucesso": np.array(suc),
            "passos": np.array(pas, dtype=float), "toca": np.array(toc)}


_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def ic95(v):
    """Meia-largura do intervalo de confiança de 95% da média (t de Student para n ≤ 10; normal acima)."""
    v = np.asarray(v, dtype=float)
    n = len(v)
    if n < 2:
        return float("nan")
    t = _T95.get(n - 1, 1.96 if n > 30 else 2.0)
    return float(t * v.std(ddof=1) / math.sqrt(n))


def resumo(av):
    """Médias (e IC95% sobre episódios) das métricas de ``avaliar``."""
    out = {}
    for k, v in av.items():
        out[k] = float(np.mean(v))
        out[k + "_ic95"] = ic95(v)
    return out


# =============================================================================
# Stable-baselines3
# =============================================================================
HP_PADRAO = {"DQN": {}, "PPO": {}, "A2C": {}}          # {} = hiperparâmetros padrão do SB3 (2.x)


class EnvTreino(gym.Wrapper):
    """Registra os episódios de treino (retorno bruto, comprimento, sucesso, toques) e escala a recompensa ao agente."""

    def __init__(self, env, escala=1.0, fator=1):
        super().__init__(env)
        self.escala = float(escala)
        self.fator = int(fator)                              # nº de ambientes paralelos: converte passos locais em passos totais
        self.passo_global = 0
        self.log = []
        self._g = 0.0

    def reset(self, **kw):
        self._g = 0.0
        return self.env.reset(**kw)

    def step(self, a):
        obs, r, term, trunc, info = self.env.step(a)
        self.passo_global += 1
        self._g += r
        if term or trunc:
            self.log.append((self.passo_global * self.fator, self._g, self.env.passos, float(info["sucesso"]), float(info["toques"] > 0)))
        return obs, r * self.escala, term, trunc, info


def criar_modelo(algo, hp, env, semente, verbose=0):
    """Cria o modelo SB3 (``DQN``, ``PPO`` ou ``A2C``) a partir de um dicionário de hiperparâmetros.

    Chaves: qualquer argumento do construtor do algoritmo (``learning_rate``, ``gamma``, ``n_steps``...), mais
    ``net_arch`` (lista de larguras da MLP; vira ``policy_kwargs``) e, tratadas fora (em ``treinar``), ``escala_recompensa``
    (fator aplicado à recompensa vista pelo agente) e ``n_envs`` (ambientes paralelos síncronos, só para PPO/A2C).
    ``{}`` mantém os valores padrão do SB3. Usa CPU e 1 thread do torch.
    """
    import torch
    from stable_baselines3 import A2C, DQN, PPO
    torch.set_num_threads(1)
    hp = dict(hp)
    hp.pop("escala_recompensa", None)
    hp.pop("n_envs", None)
    classes = {"DQN": DQN, "PPO": PPO, "A2C": A2C}
    if "net_arch" in hp:
        hp["policy_kwargs"] = dict(net_arch=list(hp.pop("net_arch")))
    return classes[algo]("MlpPolicy", env, seed=int(semente), device="cpu", verbose=verbose, **hp)


def _callback_avaliacao(freq, n_ep, params_env, gamma, historico):
    from stable_baselines3.common.callbacks import BaseCallback

    class _Aval(BaseCallback):
        def __init__(self):
            super().__init__()
            self.prox = freq
            self.env_aval = RescueBoatContEnv(**params_env)

        def _on_step(self):
            if self.num_timesteps >= self.prox:
                self._avalia()
                self.prox += freq
            return True

        def _avalia(self):
            r = resumo(avaliar(politica_sb3(self.model), n_ep, semente=20_000, gamma=gamma, env=self.env_aval))
            historico.append({"passos": int(self.num_timesteps), "ret": r["ret"], "ret_desc": r["ret_desc"],
                              "sucesso": r["sucesso"], "passos_ep": r["passos"]})

        def _on_training_end(self):
            if not historico or historico[-1]["passos"] != self.num_timesteps:
                self._avalia()
    return _Aval()


def treinar(algo, hp, semente, passos, params_env=None, gamma_aval=0.99, freq_aval=None, n_ep_aval=20, n_ep_final=100):
    """Treina um modelo e devolve ``dict(curva, treino, final, tempo)``.

    ``curva``: avaliações periódicas (política gulosa, 20 episódios com sementes fixas); ``treino``: array com colunas
    (passo, retorno, comprimento, sucesso, toques) dos episódios de treino; ``final``: ``resumo`` + vetores por episódio de
    ``n_ep_final`` episódios de avaliação determinística; ``tempo`` em segundos de relógio.
    """
    params_env = dict(params_env or {})
    t0 = time.time()
    n_envs = int(hp.get("n_envs", 1))
    envs = [EnvTreino(RescueBoatContEnv(**params_env), escala=hp.get("escala_recompensa", 1.0), fator=n_envs) for _ in range(n_envs)]
    if n_envs == 1:
        env = envs[0]
    else:
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv([(lambda e=e: e) for e in envs])
    modelo = criar_modelo(algo, hp, env, semente)
    hist = []
    cb = _callback_avaliacao(freq_aval or max(1, passos // 10), n_ep_aval, params_env, gamma_aval, hist)
    modelo.learn(total_timesteps=passos, callback=cb)
    av = avaliar(politica_sb3(modelo), n_ep_final, gamma=gamma_aval, **params_env)
    return {"curva": hist, "treino": np.array(sorted(l for e in envs for l in e.log), dtype=float).reshape(-1, 5), "final": resumo(av), "final_ep": av,
            "tempo": time.time() - t0, "modelo": modelo}


# =============================================================================
# Sarsa semi-gradiente linear com tile coding (Sutton e Barto, cap. 9-10; Aula 13)
# =============================================================================
class TileCoder:
    """Tile coding assimétrico sobre ``d`` dimensões em [0, 1]: ``n_til`` grades deslocadas de ``k/n_til`` de tile.

    Cada grade cobre a dimensão ``i`` com ``n_tiles[i]`` tiles (mais um, para a folga do deslocamento). Uma entrada ativa
    exatamente ``n_til`` características (uma por grade); o índice devolvido já inclui o deslocamento de cada grade.
    """

    def __init__(self, n_tiles, n_til=8):
        self.n_tiles = np.asarray(n_tiles, dtype=int)
        self.n_til = n_til
        self.dims = self.n_tiles + 1
        self.por_grade = int(np.prod(self.dims))
        self.tam = self.por_grade * n_til
        # deslocamentos assimétricos (1, 3, 5, ...)/n_til · tile, diferentes por dimensão (Sutton e Barto, §9.5.4)
        k = np.arange(n_til)[:, None]
        passo_d = 2 * np.arange(len(n_tiles))[None, :] + 1
        self.desl = ((k * passo_d) % n_til) / n_til                                    # (n_til, d)
        mult = np.ones(len(n_tiles), dtype=int)
        for i in range(len(n_tiles) - 2, -1, -1):
            mult[i] = mult[i + 1] * self.dims[i + 1]
        self.mult = mult
        self.base = np.arange(n_til) * self.por_grade

    def indices(self, z):
        """``z``: vetor em [0, 1]^d. Devolve os ``n_til`` índices ativos."""
        t = np.asarray(z) * self.n_tiles[None, :] + self.desl
        t = np.minimum(t.astype(int), self.dims - 1)
        return self.base + (t * self.mult).sum(1)


class SarsaSemiGradiente:
    """Sarsa semi-gradiente com ``q̂(s, a, w) = Σ w[situação, a, características(x, y, I)]`` (linear em tile coding).

    As características usam apenas a observação: posição ``(x, y)`` e intensidade ``I`` passam por tile coding; a situação
    das pessoas (≤ 27 combinações) escolhe o bloco de pesos. Atualização: ``w ← w + α/n_til · (R + γ q̂(S', A') − q̂(S, A))``
    nas ``n_til`` características ativas. ε-gulosa com decaimento linear de ``eps0`` a ``eps1`` na primeira metade do treino.
    """

    def __init__(self, n_tiles=(12, 12, 4), n_til=8, alpha=0.1, gamma=0.99, eps0=0.3, eps1=0.02, semente=0):
        self.tc = TileCoder(n_tiles, n_til)
        self.alpha, self.gamma, self.eps0, self.eps1 = alpha / n_til, gamma, eps0, eps1
        self.w = np.zeros((27, 4, self.tc.tam))
        self.rng = np.random.default_rng(semente)
        self.semente = semente

    @staticmethod
    def _sit(obs):
        o = obs[2:11].reshape(3, 3)
        a = o.argmax(1)
        return int(a[0] * 9 + a[1] * 3 + a[2])

    def _feat(self, obs):
        return self._sit(obs), self.tc.indices(np.array([obs[0], obs[1], obs[12]])[None, :])

    def q(self, sit, idx):
        return self.w[sit][:, idx].sum(1)

    def acao(self, obs, eps=0.0):
        sit, idx = self._feat(obs)
        if eps > 0 and self.rng.random() < eps:
            return int(self.rng.integers(4))
        q = self.q(sit, idx)
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    def politica(self):
        return lambda env, obs: self.acao(obs, 0.0)

    def treinar(self, passos, params_env=None, freq_aval=None, n_ep_aval=20, gamma_aval=0.99):
        """Treina por ``passos`` passos de interação. Devolve ``(curva, log_treino, tempo)`` como em ``treinar``."""
        params_env = dict(params_env or {})
        t0 = time.time()
        env = RescueBoatContEnv(**params_env)
        env_aval = RescueBoatContEnv(**params_env)
        freq = freq_aval or max(1, passos // 10)
        curva, log = [], []
        obs, _ = env.reset(seed=self.semente)
        sit, idx = self._feat(obs)
        a = self._escolhe(sit, idx, self.eps0)
        G = 0.0
        prox_aval = freq
        for t in range(1, passos + 1):
            eps = max(self.eps1, self.eps0 - (self.eps0 - self.eps1) * t / (0.5 * passos))
            obs2, r, term, trunc, info = env.step(a)
            G += r
            if term or trunc:
                log.append((t, G, env.passos, float(info["sucesso"]), float(info["toques"] > 0)))
                G = 0.0
            sit2, idx2 = self._feat(obs2)
            a2 = self._escolhe(sit2, idx2, eps)
            alvo = r if term else r + self.gamma * self.w[sit2][a2, idx2].sum()           # truncamento: continua a fazer bootstrap
            delta = alvo - self.w[sit][a, idx].sum()
            self.w[sit][a, idx] += self.alpha * delta
            if term or trunc:
                obs2, _ = env.reset()
                sit2, idx2 = self._feat(obs2)
                a2 = self._escolhe(sit2, idx2, eps)
            sit, idx, a = sit2, idx2, a2
            if t >= prox_aval or t == passos:
                r_ = resumo(avaliar(self.politica(), n_ep_aval, semente=20_000, gamma=gamma_aval, env=env_aval))
                curva.append({"passos": t, "ret": r_["ret"], "ret_desc": r_["ret_desc"], "sucesso": r_["sucesso"],
                              "passos_ep": r_["passos"]})
                prox_aval += freq
        return curva, np.array(log, dtype=float).reshape(-1, 5), time.time() - t0

    def _escolhe(self, sit, idx, eps):
        if self.rng.random() < eps:
            return int(self.rng.integers(4))
        q = self.q(sit, idx)
        return int(self.rng.choice(np.flatnonzero(q == q.max())))


def velocidade(n=20000, semente=0):
    """Passos por segundo do ambiente (ações aleatórias, sem SB3)."""
    env = RescueBoatContEnv()
    env.reset(seed=semente)
    rng = np.random.default_rng(semente)
    acoes = rng.integers(0, 4, size=n).tolist()
    t0 = time.time()
    for a in acoes:
        _, _, te, tr, _ = env.step(a)
        if te or tr:
            env.reset()
    return n / (time.time() - t0)


if __name__ == "__main__":
    tab = PoliticaTabularDiscretizada()
    print(f"V*(s0) do T2 (γ=0,99): {tab.v0:.2f}")
    print(f"velocidade do ambiente: {velocidade():.0f} passos/s")
