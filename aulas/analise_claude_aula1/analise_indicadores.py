"""Análise exploratória dos indicadores do Banco Mundial (2000-2023).

Gera os gráficos em analise/graficos/. Executar com:
    uv run --with pandas --with matplotlib python analise/analise_indicadores.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = Path(__file__).resolve().parent / "graficos"
SAIDA.mkdir(exist_ok=True)

# Paleta categórica (ordem fixa) e tokens de texto
SUPERFICIE, TEXTO, TEXTO_2, GRADE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
AZUL, LARANJA, AQUA, AMARELO, MAGENTA, VERDE, VIOLETA = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")

REGIOES = {  # nome original -> (rótulo, cor)
    "North America": ("América do Norte", AZUL),
    "Europe & Central Asia": ("Europa e Ásia Central", LARANJA),
    "East Asia & Pacific": ("Ásia Oriental e Pacífico", AQUA),
    "Latin America & Caribbean": ("América Latina e Caribe", AMARELO),
    "Middle East, North Africa, Afghanistan & Pakistan": ("Oriente Médio, N. África, AFG e PAK", MAGENTA),
    "South Asia": ("Sul da Ásia", VERDE),
    "Sub-Saharan Africa": ("África Subsaariana", VIOLETA),
}

plt.rcParams.update({
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
    "text.color": TEXTO, "axes.labelcolor": TEXTO_2, "xtick.color": TEXTO_2, "ytick.color": TEXTO_2,
    "axes.edgecolor": GRADE, "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "font.size": 10.5, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "lines.linewidth": 2, "legend.frameon": False,
})

d = pd.read_csv(RAIZ / "dados" / "indicadores_banco_mundial.csv")
d["regiao_pt"] = d["regiao"].map(lambda r: REGIOES[r][0])
ULT = d["ano"].max()


def media_pond(g, col):
    g = g.dropna(subset=[col])
    return np.average(g[col], weights=g["populacao"]) if len(g) else np.nan


def titulo(ax, t, sub):
    ax.set_title(t, pad=26)
    ax.text(0, 1.03, sub, transform=ax.transAxes, color=TEXTO_2, fontsize=10, va="bottom")


def fonte(fig):
    fig.text(0.01, 0.005, "Fonte: Banco Mundial (indicadores_banco_mundial.csv), 2000-2023.",
             color=TEXTO_2, fontsize=8.5, va="bottom")


def salvar(fig, nome):
    fonte(fig)
    fig.savefig(SAIDA / nome, dpi=160, bbox_inches="tight")
    plt.close(fig)


def sem_sobreposicao(ys, minimo):
    """Afasta rótulos diretos para não colidirem (ys: dict nome -> y)."""
    ordem = sorted(ys, key=ys.get)
    pos = {}
    anterior = -np.inf
    for n in ordem:
        pos[n] = max(ys[n], anterior + minimo)
        anterior = pos[n]
    return pos


# ---------------------------------------------------------------- 1. Expectativa de vida
vida = d.groupby(["regiao", "ano"]).apply(lambda g: media_pond(g, "expectativa_vida"), include_groups=False).unstack(0)
mundo_vida = d.groupby("ano").apply(lambda g: media_pond(g, "expectativa_vida"), include_groups=False)

fig, ax = plt.subplots(figsize=(10, 5.8))
rot = sem_sobreposicao({r: vida[r].iloc[-1] for r in REGIOES}, 1.0)
for r, (nome, cor) in REGIOES.items():
    ax.plot(vida.index, vida[r], color=cor)
    ax.text(ULT + 0.3, rot[r], nome, color=TEXTO, va="center", fontsize=9.5)
    ax.plot(ULT, vida[r].iloc[-1], "o", color=cor, ms=6, mec=SUPERFICIE, mew=1.5)
ax.plot(mundo_vida.index, mundo_vida, color=TEXTO_2, ls="--", lw=1.5)
ax.text(ULT + 0.3, mundo_vida.iloc[-1] - 0.9, "Média mundial (tracejado)", color=TEXTO_2, fontsize=9, va="center")
ax.axvspan(2020, 2021, color=GRADE, alpha=0.7, lw=0)
ax.text(2020.5, 50.7, "Covid-19", color=TEXTO_2, fontsize=9, ha="center")
ax.set_xlim(2000, ULT + 7.5)
ax.set_xticks(range(2000, 2024, 5))
ax.set_ylabel("Anos")
titulo(ax, "A África Subsaariana ganhou 11 anos de vida desde 2000, mas ainda está 16 anos atrás",
       "Expectativa de vida ao nascer por região (média ponderada pela população)")
salvar(fig, "01_expectativa_de_vida_por_regiao.png")

# ---------------------------------------------------------------- 2. Renda x longevidade
a = d[(d["ano"] == ULT)].dropna(subset=["pib_per_capita_usd"])
fig, ax = plt.subplots(figsize=(10, 6.2))
for r, (nome, cor) in REGIOES.items():
    s = a[a["regiao"] == r]
    ax.scatter(s["pib_per_capita_usd"], s["expectativa_vida"], s=np.sqrt(s["populacao"]) / 25,
               color=cor, alpha=0.85, edgecolor=SUPERFICIE, linewidth=1, label=nome)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"US$ {v:,.0f}".replace(",", ".")))
ax.set_xlabel("PIB per capita (escala logarítmica)")
ax.set_ylabel("Expectativa de vida (anos)")
for p, dx, dy in [("Brazil", 0, -2.6), ("India", 0, -2.6), ("China", 0, 1.6), ("United States", 0, -2.6),
                  ("Nigeria", 0, -2.6), ("Japan", 0, 1.4)]:
    l = a[a["pais"] == p]
    if len(l):
        ax.annotate(p.replace("Brazil", "Brasil").replace("India", "Índia").replace("United States", "EUA"),
                    (l["pib_per_capita_usd"].iloc[0], l["expectativa_vida"].iloc[0]),
                    xytext=(0, dy * 3.2), textcoords="offset points", ha="center", fontsize=9, color=TEXTO)
leg = ax.legend(loc="lower right", fontsize=8.5, markerscale=0.6)
r_ = np.corrcoef(np.log(a["pib_per_capita_usd"]), a["expectativa_vida"])[0, 1]
titulo(ax, f"Renda e longevidade caminham juntas (correlação de {r_:.2f} com log do PIB)",
       f"Países em {ULT}; tamanho da bolha = população. Retornos decrescentes: acima de ~US$ 20 mil a renda pouco muda a longevidade")
salvar(fig, "02_pib_x_expectativa_de_vida.png")

# ---------------------------------------------------------------- 3. Maiores ganhos de longevidade
p = d.pivot(index="pais", columns="ano", values="expectativa_vida")
ganho = (p[ULT] - p[2000])
# Exclui República Centro-Africana: série com valores implausíveis (14,7 anos em 2009; 18,8 em 2022)
ganho = ganho.drop("Central African Republic", errors="ignore").sort_values()
sel = pd.concat([ganho.head(5), ganho.tail(10)])
fig, ax = plt.subplots(figsize=(10, 6.2))
cores = [LARANJA if v < 0 else AZUL for v in sel]
ax.barh(sel.index, sel.values, color=cores, height=0.65)
for y, v in enumerate(sel.values):
    ax.text(v + (0.3 if v >= 0 else -0.3), y, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right",
            fontsize=9, color=TEXTO)
ax.axvline(0, color=TEXTO_2, lw=1)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Variação 2000-2023 (anos)")
ax.set_xlim(-8, 24)
titulo(ax, "Os maiores ganhos de longevidade estão na África; poucos países recuaram",
       "10 maiores ganhos e 5 menores variações da expectativa de vida, 2000-2023 (exclui Rep. Centro-Africana, dados inconsistentes)")
salvar(fig, "03_ganhos_expectativa_de_vida.png")

# ---------------------------------------------------------------- 4. Crescimento do PIB per capita
q = d.pivot(index="pais", columns="ano", values="pib_per_capita_usd")
ok = q[[2000, ULT]].dropna()
cagr = ((ok[ULT] / ok[2000]) ** (1 / (ULT - 2000)) - 1).mul(100).sort_values()
sel = pd.concat([cagr.head(6), cagr.tail(10)])  # 6 menores + 10 maiores
fig, ax = plt.subplots(figsize=(10, 6.2))
ax.barh(sel.index, sel.values, color=[LARANJA if v < 0 else AZUL for v in sel], height=0.65)
for y, v in enumerate(sel.values):
    ax.text(v + (0.15 if v >= 0 else -0.15), y, f"{v:+.1f}%", va="center",
            ha="left" if v >= 0 else "right", fontsize=9)
ax.axvline(0, color=TEXTO_2, lw=1)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Crescimento médio anual do PIB per capita em US$ correntes, 2000-2023 (%)")
ax.set_xlim(-3, 16)
titulo(ax, "Emergentes (Guiana, Leste Europeu, China) lideram o crescimento; Venezuela e Líbia encolheram",
       "10 maiores e 6 menores crescimentos (países com dados em 2000 e 2023); efeito de câmbio e commodities incluído")
salvar(fig, "04_crescimento_pib_per_capita.png")

# ---------------------------------------------------------------- 5. Desemprego
desemp = d.groupby("ano").apply(lambda g: media_pond(g, "desemprego_pct"), include_groups=False)
br = d[d["pais"] == "Brazil"].set_index("ano")["desemprego_pct"]
la = d[d["regiao"] == "Latin America & Caribbean"].groupby("ano").apply(
    lambda g: media_pond(g, "desemprego_pct"), include_groups=False)
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.plot(br.index, br, color=AZUL)
ax.plot(la.index, la, color=LARANJA)
ax.plot(desemp.index, desemp, color=TEXTO_2, ls="--", lw=1.5)
for serie, nome, cor in [(br, "Brasil", AZUL), (la, "América Latina", LARANJA), (desemp, "Mundo", TEXTO_2)]:
    ax.plot(ULT, serie.iloc[-1], "o", color=cor, ms=6, mec=SUPERFICIE, mew=1.5)
rot = sem_sobreposicao({"Brasil": br.iloc[-1], "América Latina": la.iloc[-1], "Mundo": desemp.iloc[-1]}, 0.7)
for nome, y in rot.items():
    ax.text(ULT + 0.3, y, nome, va="center", fontsize=9.5)
ax.annotate(f"Pico do Brasil: {br.max():.1f}% ({br.idxmax()})", xy=(br.idxmax(), br.max()),
            xytext=(2007.5, 14.4), fontsize=9, color=TEXTO_2,
            arrowprops=dict(arrowstyle="-", color=TEXTO_2, lw=0.8))
ax.set_xlim(2000, ULT + 4)
ax.set_ylim(0, 15.5)
ax.set_ylabel("% da força de trabalho")
titulo(ax, "O desemprego brasileiro dobrou em 2015-17 e voltou a cair, mas ainda supera o mundo",
       "Taxa de desemprego; Mundo e América Latina são médias ponderadas pela população")
salvar(fig, "05_desemprego_brasil_x_mundo.png")

# ---------------------------------------------------------------- 6. Brasil: indexado a 2000
b = d[d["pais"] == "Brazil"].set_index("ano")
fig, ax = plt.subplots(figsize=(10, 5.6))
idx = pd.DataFrame({"PIB per capita (US$)": b["pib_per_capita_usd"], "Expectativa de vida": b["expectativa_vida"],
                    "População": b["populacao"]})
idx = idx / idx.loc[2000] * 100
for (c, cor) in zip(idx.columns, [AZUL, LARANJA, AQUA]):
    ax.plot(idx.index, idx[c], color=cor)
    ax.plot(ULT, idx[c].iloc[-1], "o", color=cor, ms=6, mec=SUPERFICIE, mew=1.5)
rot = sem_sobreposicao({c: idx[c].iloc[-1] for c in idx}, 9)
for c, y in rot.items():
    ax.text(ULT + 0.3, y, f"{c} ({idx[c].iloc[-1]:.0f})", va="center", fontsize=9.5)
ax.axhline(100, color=TEXTO_2, lw=0.8)
ax.set_xlim(2000, ULT + 6.5)
ax.set_ylabel("Índice (2000 = 100)")
titulo(ax, "Brasil: a renda em dólares oscila muito; longevidade e população crescem de forma estável",
       "Indicadores do Brasil indexados a 2000 = 100 (PIB em US$ correntes, sujeito ao câmbio)")
salvar(fig, "06_brasil_indexado.png")

# ---------------------------------------------------------------- Números-chave para o relatório
print("Média mundial (pond.) expectativa de vida:", round(mundo_vida[2000], 1), "->", round(mundo_vida[ULT], 1))
print("Desemprego mundial (pond.):", round(desemp[2000], 2), "->", round(desemp[ULT], 2), "| 2020:", round(desemp[2020], 2))
print("Gap vida N.América x Subsaariana:", round(vida["North America"][2000] - vida["Sub-Saharan Africa"][2000], 1),
      "->", round(vida["North America"][ULT] - vida["Sub-Saharan Africa"][ULT], 1))
print("Subsaariana ganho:", round(vida["Sub-Saharan Africa"][ULT] - vida["Sub-Saharan Africa"][2000], 1))
print("Brasil 2023:", b.loc[ULT, ["pib_per_capita_usd", "expectativa_vida", "desemprego_pct"]].round(1).to_dict())
print("Corr log PIB x vida", round(r_, 2))
