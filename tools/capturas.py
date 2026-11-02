"""Gera as capturas de tela de docs/img com Playwright (servir antes: npm run serve). Falha se houver erro no console.

Uso:  python tools/capturas.py [url_base]   (padrão: http://127.0.0.1:8000/index.html)
"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/index.html").split("#")[0]
OUT = Path(__file__).resolve().parent.parent / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
SECOES = ["intro", "ambiente", "politicas", "treinar", "resultados", "resultados-extra", "sobre"]


def preparar(pg, secao):
    pg.goto(f"{BASE}#{secao.split('-')[0]}")
    pg.wait_for_selector('html[data-pronto="1"]')
    if secao == "ambiente":
        pg.click("#amb-canvas")
        for k in ["ArrowLeft", "ArrowLeft", "ArrowUp", "ArrowUp", "ArrowRight", "ArrowDown"]:
            pg.keyboard.press(k)
    elif secao == "politicas":
        pg.evaluate("() => { const v = document.querySelector('#pol-vel'); v.value = 40; v.dispatchEvent(new Event('input')); }")
        pg.click("#pol-play")
        pg.wait_for_timeout(2600)
        pg.click("#pol-play")
        pg.fill("#cmp-n", "100")
        pg.click("#cmp-go")
        pg.wait_for_selector("#cmp-res:not([hidden])", timeout=120000)
    elif secao == "treinar":
        pg.uncheck("#tr-anima")
        pg.click("#tr-go")
        pg.wait_for_selector("#tr-res-cartao:not([hidden])", timeout=120000)
        pg.click("#tr-play")
        pg.wait_for_timeout(2500)
    elif secao == "resultados":
        pg.click("#aba-curvas")
    elif secao == "resultados-extra":
        pg.click("#aba-extra")
    pg.evaluate("() => window.scrollTo(0, 0)")   # cabeçalho fixo não pode aparecer no meio da captura de página inteira
    pg.wait_for_timeout(400)


with sync_playwright() as p:
    b = p.chromium.launch()
    for lang in ("pt", "en"):
        for nome, vp in (("desktop", {"width": 1280, "height": 900}), ("mobile", {"width": 390, "height": 800})):
            ctx = b.new_context(viewport=vp, color_scheme="light")
            pg = ctx.new_page()
            erros = []
            pg.on("console", lambda m, e=erros: e.append(m.text) if m.type in ("error", "warning") else None)
            pg.on("pageerror", lambda x, e=erros: e.append(str(x)))
            pg.goto(BASE)
            pg.wait_for_selector('html[data-pronto="1"]')
            pg.evaluate("l => localStorage.setItem('rl-barco-lang', l)", lang)
            pg.reload()
            pg.wait_for_selector('html[data-pronto="1"]')
            for s in SECOES:
                if nome == "mobile" and s not in ("intro", "ambiente", "politicas", "treinar", "resultados-extra"):
                    continue
                preparar(pg, s)
                pg.screenshot(path=str(OUT / f"{nome}-{lang}-{s}.png"), full_page=True)
            ctx.close()
            assert not erros, f"erros no console ({nome}, {lang}): {erros}"
            print(nome, lang, "console sem erros")
    b.close()
print("ok", OUT)
