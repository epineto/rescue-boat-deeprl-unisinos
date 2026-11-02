# Reprodução e regeneração dos dados

Este documento descreve como os arquivos de `data/` são produzidos e como o laboratório se liga ao experimento do Trabalho 3.

## 1. Experimento (fora deste repositório)
O experimento (`experimentos_t3.py` e `barco_continuo.py`, em `codigo/` do projeto do trabalho) grava, em `resultados/`:

- `runs/<fase>_<ALGO>_<config>_s<semente>.json`: uma execução por arquivo (hiperparâmetros, curva de avaliação, métricas finais por episódio);
- `modelos/final_<ALGO>_<config>_s<semente>.zip`: modelos do stable-baselines3 das execuções finais;
- `linhas_de_base.csv`, `melhores_configs.json`, `info.json`, `hipoteses.csv`, `auc_curvas.csv`;
- `extra/`: experimento suplementar (`codigo/experimento_extra_t3.py`): `extra_orcamento_por_semente.csv`, `extra_orcamento_resumo.csv`, `extra_curvas.csv`, `extra_diagnostico_treino.csv`, `cartpole_por_semente.csv`, `cartpole_resumo.csv`, `info_extra.json`.

`reference/python/barco_continuo.py` é uma **cópia literal** do módulo do experimento (ambiente, MDP do Trabalho 2, Sarsa, tile coding). O script de exportação compara os hashes quando a pasta `../codigo` existe e avisa se divergirem.

## 2. Exportação
```bash
cd lab
python tools/exportar_para_lab.py \
    [--resultados ../resultados] [--codigo reference/python] [--saida data] \
    [--so-extra] [--permitir-incompleto] [--sem-sarsa] [--semente-sarsa N]
```

Sem opções, a execução é a **final**: exige todas as execuções (busca: 4 algoritmos × 6 configurações × sementes; finais: 4 algoritmos × sementes). Com `--permitir-incompleto`, aceita resultados parciais e marca `indice.json` (`completo: false`); a interface exibe então um aviso de dados incompletos. Com `--so-extra`, apenas os arquivos do experimento suplementar (`extra_*.csv`, `cartpole_*.csv`) e o bloco `extra` de `indice.json` são regenerados, sem retreinar o Sarsa. O script é idempotente: apaga somente os arquivos que gera e produz os mesmos bytes para a mesma entrada (conferido: a exportação completa e a sequência exportação completa + `--so-extra` geram arquivos idênticos).

| Arquivo | Conteúdo |
|---|---|
| `indice.json` | Parâmetros do ambiente, protocolo, configurações, melhores, lista de modelos, contagens, versões |
| `busca_runs.csv`, `busca_resumo.csv` | Todas as execuções da busca; média, DP e IC95% por configuração, posição e marca da escolhida |
| `final_runs.csv`, `final_resumo.csv` | Execuções finais; resumo por algoritmo e linhas de base |
| `curvas_busca.csv`, `curvas_finais.csv` | Curvas de aprendizado em formato longo (grade nominal de passos) |
| `linhas_de_base.csv`, `hipoteses.csv`, `auc_curvas.csv` | Cópias das saídas do experimento |
| `extra_runs.csv`, `extra_resumo.csv`, `extra_curvas.csv` | Experimento suplementar (2 milhões de passos): uma linha por run (com o valor da mesma run aos 400 mil passos e o diagnóstico do treino), resumo por configuração (média, DP, IC95% e nº de runs que aprenderam, com sucesso final ≥ 0,5) e curvas |
| `cartpole_runs.csv`, `cartpole_resumo.csv` | Sanidade da fábrica de agentes no CartPole-v1 |
| `modelos/<ALGO>_s<semente>.json` | Pesos das MLPs (camadas, ativação, saída Q ou logits) |
| `modelos/sarsa_s<semente>.json` | Pesos esparsos (índices int32 e valores float32 em base64) |
| `paridade.json` | Observações fixas e saídas do PyTorch/Python para os testes |

O Sarsa não salva pesos no experimento. O script retreina a melhor semente das execuções finais com o mesmo código, os mesmos hiperparâmetros, o mesmo orçamento e a mesma grade de avaliação, o que reproduz a execução; o retorno descontado retreinado é comparado com o do JSON da execução e uma divergência é avisada na saída.

## 3. Testes de paridade
- `tests/reference.json` (gerado por `tools/gen_reference.py`, só necessário se o código Python mudar): MDP do Trabalho 2, tile coding, 1.500 atualizações do Sarsa e MLPs sintéticos.
- `tests/trajetoria_referencia.json` e `tests/linhas_de_base_referencia.csv`: cópias das saídas do experimento usadas como verdade de referência para o ambiente em JavaScript.
- `data/paridade.json`: gerado a cada exportação, compara a inferência de cada modelo exportado com o PyTorch.

## 4. Capturas de tela
```bash
npm run serve                                   # em outro terminal
python tools/capturas.py  # grava docs/img/*.png (Playwright)
python tools/gerar_og.py  # grava assets/og.png com o gráfico final de retorno descontado
```
