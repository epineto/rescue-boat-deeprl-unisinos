// Gerador portável: mulberry32 + Box-Muller. Idêntico ao de reference/python/barco_continuo.py (classe Mulberry32).
// Portable generator: mulberry32 + Box-Muller, identical to the Python reference.

/** Cria um gerador com semente de 32 bits. `u()` devolve um uniforme em [0, 1); `par()` devolve duas normais padrão. */
export function criarRng(semente = 0) {
  let s = semente >>> 0;
  const u = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const par = () => {
    const u1 = u(), u2 = u();
    const r = Math.sqrt(-2 * Math.log(1 - u1));
    const ang = 2 * Math.PI * u2;
    return [r * Math.cos(ang), r * Math.sin(ang)];
  };
  return { u, par, int: (n) => Math.floor(u() * n), get estado() { return s; } };
}
