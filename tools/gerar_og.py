"""Gera assets/og.png (1200x630) com o gráfico de retorno descontado FINAL do próprio laboratório (servir antes: npm run serve).

Uso:  python tools/gerar_og.py [url_base]   (padrão: http://127.0.0.1:8000/index.html)
Os números exibidos vêm de data/final_resumo.csv (via o gráfico renderizado e a tabela de chamadas abaixo); nada é digitado à mão.
"""
import base64
import csv
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

RAIZ = Path(__file__).resolve().parent.parent
BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/index.html").split("#")[0]
fmt = lambda x, d=1: f"{x:.{d}f}".replace(".", ",").replace("-", "−")
res = {r["agente"]: r for r in csv.DictReader((RAIZ / "data" / "final_resumo.csv").open(encoding="utf-8"))}
chips = [("DQN", "DQN"), ("SARSA", "Sarsa"), ("PPO", "PPO"), ("A2C", "A2C"), ("tabular", "Tabular (T2)")]
chips_html = "".join(f'<div class="chip"><b>{n}</b><span>{fmt(float(res[k]["ret_desc"]))}</span></div>' for k, n in chips)
b64 = lambda p: base64.b64encode(Path(p).read_bytes()).decode()
logo = b64(RAIZ / "assets" / "logo_unisinos.png")

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_context(viewport={"width": 1100, "height": 900}, color_scheme="light", device_scale_factor=2).new_page()
    erros = []
    pg.on("console", lambda m: erros.append(m.text) if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda x: erros.append(str(x)))
    pg.goto(BASE + "#resultados")
    pg.wait_for_selector('html[data-pronto="1"]')
    pg.wait_for_timeout(500)
    graf = pg.locator("#res-g-ret").locator("xpath=..")
    graf.screenshot(path=str(RAIZ / "assets" / "_grafico_tmp.png"))
    assert not erros, erros
    img = b64(RAIZ / "assets" / "_grafico_tmp.png")
    (RAIZ / "assets" / "_grafico_tmp.png").unlink()
    html = f"""<html><body style="margin:0;width:1200px;height:630px;font-family:Arial,Helvetica,sans-serif;color:#fff;
      background:linear-gradient(135deg,#1D2369 0%,#2A3190 100%);display:flex;box-sizing:border-box;padding:56px 60px;gap:44px">
      <style>.chip{{background:rgba(255,255,255,.12);border-radius:8px;padding:8px 11px;display:flex;flex-direction:column;gap:2px;min-width:0}}
      .chip b{{font-size:15px;color:#C9D1FF}}.chip span{{font-size:24px;font-weight:700}}</style>
      <div style="flex:1;display:flex;flex-direction:column;justify-content:center;gap:20px">
        <div style="background:#fff;border-radius:6px;padding:10px 14px;align-self:flex-start"><img style="height:56px;display:block" src="data:image/png;base64,{logo}"></div>
        <div style="font-size:46px;font-weight:700;line-height:1.12">Laboratório de Aprendizado por Reforço</div>
        <div style="font-size:23px;line-height:1.35;color:#D6DBFF">Barco de resgate com estados contínuos: DQN, PPO, A2C e Sarsa com tile coding, no navegador.</div>
        <div style="font-size:16px;color:#C9D1FF;margin-top:2px">Retorno descontado final (média entre 5 sementes)</div>
        <div style="display:flex;gap:8px">{chips_html}</div>
        <div style="font-size:17px;color:#AEB8FF">Trabalho 3 · PPGCA/Unisinos · 2026/2</div>
      </div>
      <div style="width:470px;align-self:center;background:#fff;border-radius:14px;padding:14px;box-shadow:0 10px 30px rgba(0,0,0,.35)">
        <img style="width:100%;display:block" src="data:image/png;base64,{img}"></div></body></html>"""
    pg2 = b.new_context(viewport={"width": 1200, "height": 630}).new_page()
    pg2.set_content(html)
    pg2.wait_for_timeout(300)
    pg2.screenshot(path=str(RAIZ / "assets" / "og.png"))
    b.close()
print("ok", RAIZ / "assets" / "og.png")
