// Seção #sobre: limitações e versões. / About section: limitations and versions.
import { $, el } from './util.js';
import * as I18N from './i18n.js';

const { t } = I18N;

export function iniciarSobre(S) {
  function render() {
    $('#sobre-lim').replaceChildren(...[1, 2, 3, 4, 5, 6, 7].map((k) => el('li', { html: t('sobre.lim' + k) })));
    const v = S.dados.indice.versoes ?? {};
    $('#sobre-versao').textContent = Object.keys(v).length ? t('sobre.versoes', { v: Object.entries(v).map(([k, x]) => `${k} ${x}`).join(', ') }) : '';
  }
  render();
  I18N.onLang(render);
  return { ativar() {}, desativar() {} };
}
