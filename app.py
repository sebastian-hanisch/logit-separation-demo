"""Logit und Trennung – wenn der Schätzer nicht existiert - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein Ja/Nein-Ausgang (Standgeld fällig: Verweilzeit eines Containers über 7 Tage): Logit, IRLS gleich Newton, was geschieht, wenn die Daten trennen, wie oft das vorkommt
und was hilft. Handbeispiele mit gesetzten Zahlen und ein erfundener Containerstrom; schwere Messungen sind vorgerechnet (data/lsp_frozen.json).

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import lsp_constants as C
import lsp_evaluation as ev
import lsp_scenario as S
from lsp_layout import dunkles_design, kopf_css
from lsp_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from lsp_visualization import (
    build_exp1_daten,
    build_exp1_fehler,
    build_exp2_beta,
    build_exp2_deviance,
    build_exp2_tests,
    build_exp2_zuwachs,
    build_exp3,
    build_exp4,
    build_exp5,
)

st.set_page_config(page_title="Logit und Trennung – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _e1(beispiel):
    return ev.exp1(beispiel)


@st.cache_data(show_spinner=False)
def _e2(daten, skala, steps, regel):
    return ev.exp2(daten, skala, steps, regel)


@st.cache_data(show_spinner=False)
def _e3_beispiel(n, noise, seed):
    return ev.exp3_beispiel(n, noise, seed)


@st.cache_data(show_spinner=False)
def _e4_beispiel(n, seed):
    return ev.exp4_beispiel(n, seed)


@st.cache_data(show_spinner=False)
def _raten():
    return ev.wahre_rate_je_art()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in lsp_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📈 Logit und Trennung – wenn der Schätzer nicht existiert")
st.markdown(
    """
Ob ein Container **Standgeld** zahlen muss (Verweilzeit über 7 Tage), ist ein **Ja/Nein-Ausgang**. Eine Gerade durch Nullen und Einsen sagt Werte unter 0 und über 1 voraus; darum passt man ein **Logit**
an, per **IRLS** (iterativ gewichtete kleinste Quadrate), und beim Logit ist jeder IRLS-Schritt ein **Newton-Schritt**. Die Falle: **trennen** die Daten, etwa weil in der Stichprobe kein einziger
Umschlag-Container Standgeld zahlt, **existiert der Schätzer nicht**, die Software meldet trotzdem Zahlen, und sie hängen an der Abbruchregel; der Wald-Test sieht dabei keinen Effekt.

Die Demo rechnet Handbeispiele in Brüchen und einen erfundenen Containerstrom (kein Echtdatenbezug), misst, wie oft Trennung vorkommt und was bei getrennten Stichproben hilft. Weiter unten: die vier
Experimente, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert das Logit mit IRLS", expanded=True):
    st.markdown(
        r"""
1. **Modell:** Standgeld ($y = 1$) oder nicht ($y = 0$); die Merkmale bestimmen $\eta = X\beta$, und $p = 1/(1 + e^{-\eta})$ ist die Wahrscheinlichkeit.
2. **Ein IRLS-Schritt:** aus $\beta$ die Wahrscheinlichkeiten $p_i$ und Gewichte $w_i = p_i(1 - p_i)$ bilden, dann $\beta \leftarrow \beta + (X'WX)^{-1} X'(y - p)$. Beim Logit ist das ein Newton-Schritt auf der Log-Likelihood.
3. **Wiederholen**, bis sich $\beta$ nicht mehr ändert. Im ersten Beispiel unten (neun Container) sind es sechs Schritte, und der Grenzwert hat eine geschlossene Form.
4. **Wann es nicht endet:** gibt es eine Richtung, in der die Log-Likelihood nie sinkt (die Daten **trennen**), wächst $\beta$ mit jedem Schritt weiter, und der Schätzer existiert nicht.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Frisch gezogene Container**")
    seed = st.number_input(
        "Zufalls-Seed der Stichprobe", *bounds("seed_input"), key="seed_input", step=1,
        help="Zufalls-Seed der frisch gezogenen Stichproben in den Experimenten zu Häufigkeit und Hilfe. Gleiche Eingabe, gleiche Container, auf jeder Plattform.",
    )
    st.button("🎲 Neue Stichprobe ziehen", width="stretch", on_click=randomize_seed)
    st.caption("Die übrigen Regler stehen bei ihrem Experiment.")

fz = ev.load_frozen()
br3000 = ev.bruecke_zeile(fz, 3000)

# --- Primäransicht: von Hand ----------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Von Hand: IRLS gleich Newton")
st.caption("Gesetzte Zahlen, kein Generatorlauf. Zollhalt: Zollhalt 0 hat vier Container mit einem Standgeld, Zollhalt 1 fünf Container mit vier Standgeldern. Vorlauf: acht Container, Vorlauf 1 bis 8 Tage.")
c1, c2 = st.columns([1, 2])
beispiel = c1.radio("Beispiel", C.E1_BEISPIELE, key="e1_beispiel")
steps1 = c2.slider("IRLS-Schritte", *bounds("e1_steps"), key="e1_steps", help="Wie viele Schritte vom Start β = 0 aus gerechnet werden.")
r1 = _e1(beispiel)
b1 = r1["pfad"][steps1]
m1, m2, m3, m4 = st.columns(4)
m1.metric("β nach den Schritten", f"({ev.de(b1[0], 3)}; {ev.de(b1[1], 3)})")
m2.metric("Grenzwert", f"({ev.de(r1['grenz'][0], 3)}; {ev.de(r1['grenz'][1], 3)})")
m3.metric("Abstand zum Grenzwert", ev.abstand_text(r1["fehler"][steps1]))
m4.metric("Schritte bis Änderung < 1e-13", str(r1["k_konv"]))
g1, g2 = st.columns(2)
g1.plotly_chart(build_exp1_daten(r1, steps1), width="stretch", key="chart_e1_daten")
g2.plotly_chart(build_exp1_fehler(r1, steps1), width="stretch", key="chart_e1_fehler")
st.caption(
    "Links: die Daten, das Logit nach den gewählten Schritten und die Gerade auf y ∈ {0, 1}; gepunktet 0 und 1. Rechts: der größte Abstand von β zum Grenzwert je Schritt, logarithmisch."
)
zeilen = []
for k in range(steps1 + 1):
    zeilen.append(
        {
            "Schritt": k, "β₀ (IRLS)": ev.de(r1["pfad"][k, 0], 6), "β₁ (IRLS)": ev.de(r1["pfad"][k, 1], 6), "Deviance": ev.de(r1["dev"][k], 4),
            "β₀ (Newton)": ev.de(r1["newton"][k, 0], 6), "β₁ (Newton)": ev.de(r1["newton"][k, 1], 6), "Abstand zum Grenzwert": ev.abstand_text(r1["fehler"][k]),
        }
    )
st.dataframe(pd.DataFrame(zeilen), hide_index=True, width="stretch")
st.caption(
    "Newton: Hesse-Matrix aus der zweiten Ableitung der Log-Likelihood, numerisch als Differenzenquotient des Gradienten gebildet (ohne die Formel w = p (1 − p) zu benutzen). "
    f"Erster Schritt in Brüchen: β = ({ev.bruch_text(r1['bruch'][0])}; {ev.bruch_text(r1['bruch'][1])}) (gleich dem Gleitkommawert der Tabelle)."
)
kind, text = ev.verdict_exp1(r1, steps1, br3000)
getattr(st, kind)(text)
with st.expander("Gerade gegen Logit im Container-Vehikel (vorgerechnet)"):
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Container je Stichprobe": ev.fmt_int(r["n"]), "Stichproben": r["reps"], "Testcontainer mit Vorhersage unter 0 oder über 1 (Mittel ± Streuung)": f"{ev.pct(r['ausserhalb_mittel'])} ± {ev.de(100 * r['ausserhalb_sd'], 1)}",
                    "RMSE gegen die wahre Wahrscheinlichkeit: Gerade (auf [0, 1] gekappt)": ev.de(r["rmse_ols"], 4), "RMSE: Logit mit Firth": ev.de(r["rmse_firth"], 4),
                }
                for r in fz["bruecke"]
            ]
        ),
        hide_index=True, width="stretch",
    )
    st.caption("Vorgerechnet (`tools/lsp_make_frozen.py`), Zähleinheit: der Testcontainer (100 000 Container, nur Merkmale) und die Stichprobe; Standardstreuung, sieben Spalten.")

st.markdown("---")

# --- Kernfragen -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
r2_k = _e2(st.session_state["e2_daten"], st.session_state["e2_skala"], st.session_state["e2_steps"], st.session_state["e2_regel"])
zeile3_k = ev.haeufigkeit_zeile(fz, st.session_state["e3_n"], C.E3_NOISE[st.session_state["e3_streuung"]])
rows4_k = ev.exp4_tabelle(fz, st.session_state["e4_n"], st.session_state["e4_k"])
st.markdown("**1. Warum ein Logit statt einer Geraden, und ist das Newton?**")
kind, text = ev.kernfrage_1(r1, br3000)
getattr(st, kind)(text)
st.markdown("**2. Was geschieht, wenn die Daten trennen?**")
kind, text = ev.kernfrage_2(r2_k)
getattr(st, kind)(text)
st.markdown("**3. Wie oft kommt das vor, und was hilft?**")
kind, text = ev.kernfrage_3(zeile3_k, rows4_k, st.session_state["e4_n"], st.session_state["e4_k"])
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 2: Trennung ------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment: Wenn die Daten trennen")
st.caption(
    "Gesetzte Zahlen, kein Generatorlauf: sechs Punkte x = −3, −2, −1, 1, 2, 3 mit y = 0, 0, 0, 1, 1, 1 (vollständig getrennt) oder acht Punkte x = −3, −2, −1, 0, 0, 1, 2, 3 mit y = 0, 0, 0, 0, 1, 1, 1, 1 "
    "(zwei Punkte bei 0 mit verschiedenem Ausgang: nur quasi getrennt). Die Schritte rechnen höchstens so lange, die Abbruchregel sagt, wo ein Programm anhalten würde."
)
e1, e2c = st.columns(2)
daten = e1.radio("Daten", C.E2_DATEN, key="e2_daten")
regel = e2c.selectbox("Abbruchregel", C.E2_REGELN, key="e2_regel", help="Deviance-Regel: |ΔD| / (|D| + 0,1) unter 1e-8; Koeffizientenänderung: größte Änderung unter 1e-8; feste Schrittzahl: genau so viele Schritte, wie der Regler sagt.")
e3, e4 = st.columns(2)
steps2 = e3.slider("Schritte (höchstens)", *bounds("e2_steps"), key="e2_steps")
skala = e4.select_slider("Skalierung des Merkmals", options=C.E2_SKALEN, key="e2_skala", help="x wird mit diesem Faktor multipliziert; bei den sechs Punkten ändert das den Zuwachs je Schritt.")
r2 = _e2(daten, skala, steps2, regel)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Trennung (Rezessions-LP)", r2["art"], help=f"Optimum des Linearprogramms: {ev.de(r2['lp_opt'], 1)}; über {C.LP_SCHWELLE:g} heißt: kein endlicher Schätzer.")
m2.metric("Abbruch bei Schritt", str(r2["k_stopp"]))
m3.metric("β₁ beim Abbruch", ev.de(r2["pfad"][r2["k_stopp"], 1], 2))
m4.metric("Wald z beim Abbruch", ev.de(r2["z"][r2["k_stopp"]], 4))
a1, a2 = st.columns(2)
a1.plotly_chart(build_exp2_beta(r2), width="stretch", key="chart_e2_beta")
a2.plotly_chart(build_exp2_zuwachs(r2), width="stretch", key="chart_e2_zuwachs")
a3, a4 = st.columns(2)
a3.plotly_chart(build_exp2_deviance(r2), width="stretch", key="chart_e2_dev")
a4.plotly_chart(build_exp2_tests(r2), width="stretch", key="chart_e2_tests")
st.caption(
    "Gerechnet mit stabilen Formeln (w = σ(η) σ(−η), Residuum ohne Auslöschung): kein Wert wird gekappt, kein Schritt halbiert. Der Wald-Test teilt β₁ durch den Standardfehler aus (X'WX)⁻¹; "
    "der Likelihood-Quotient-Test vergleicht die Deviance mit der des Modells nur mit Achsenabschnitt (ein Freiheitsgrad)."
)
tab2 = []
for rg in C.E2_REGELN:
    k, greift = r2["stopp"][rg]
    tab2.append({"Abbruchregel": rg, "hält bei Schritt": f"{k}" + ("" if greift else " (Ende des Pfads)"), "β₁": ev.de(r2["pfad"][k, 1], 2), "Wald z": ev.de(r2["z"][k], 4), "p (Wald)": ev.de(r2["p_wald"][k], 3), "p (Likelihood-Quotient)": ev.de(r2["p_lq"][k], 4)})
st.dataframe(pd.DataFrame(tab2), hide_index=True, width="stretch")
kind, text = ev.verdict_exp2(r2)
getattr(st, kind)(text)
with st.expander("Was Programme daraus machen (Messung dieser README, nicht von der App gerechnet)"):
    st.markdown(
        "scikit-learn 1.9.1 (`LogisticRegression`, `penalty=None`, lbfgs) auf die sechs Punkte, Windows: β₁ = 8,35 bei `max_iter` 100 und `tol` 1e-4, 17,36 bei 1000 und 1e-8, 27,06 bei 100000 und 1e-12; "
        "**keine Konvergenzwarnung** (aufgefangen wurde nur eine Warnung zum veralteten Argument `penalty`). Eine Koeffizientenregel (Änderung unter 1e-8) greift in 700 Schritten nie. "
        "Das gilt für diese Version, diese Einstellungen und diese Punkte; die Regel mit Kappung |η| ≤ 30 ist hier aus dem Gedächtnis nachgebaut (Experiment „Was hilft?“), nicht gegen R oder eine andere Software geprüft."
    )

st.markdown("---")

# --- Experiment 3: Häufigkeit ---------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment: Wie oft trennen die Daten?")
st.caption(
    "Container-Vehikel: Art (Import, Export, Umschlag, Leer), Zollhalt, Wochenende, Vorlauf; Standgeld = Verweilzeit über 7 Tage; Modell mit sieben Spalten (Achsenabschnitt, Export, Umschlag, Leer, Zollhalt, Wochenende, Vorlauf). "
    "Zähleinheit: die Stichprobe. **Vorgerechnet** (`tools/lsp_make_frozen.py`): 100 bis 300 Stichproben je n; unten zusätzlich eine frisch gezogene Stichprobe."
)
f1, f2 = st.columns(2)
n3 = f1.select_slider("Container je Stichprobe n", options=C.E3_N, key="e3_n")
streu = f2.radio("Streuung der Log-Verweilzeit", C.E3_STREUUNG, key="e3_streuung", horizontal=True, help="Halbe Streuung heißt stärkeres Signal: die Merkmale erklären mehr vom Ausgang.")
noise3 = C.E3_NOISE[streu]
z3 = ev.haeufigkeit_zeile(fz, n3, noise3)
lo3, hi3 = ev.wilson(z3["getrennt"], z3["reps"])
m1, m2, m3, m4 = st.columns(4)
m1.metric("Getrennte Stichproben", f"{z3['getrennt']} von {z3['reps']}", help="Zähleinheit: die Stichprobe; Rezessions-LP, vollständig oder quasi.")
m2.metric("Anteil (95-%-Band)", f"{ev.pct(z3['getrennt'] / z3['reps'])}", delta=f"{ev.pct(lo3, 0)} bis {ev.pct(hi3, 0)}", delta_color="off")
m3.metric("davon nur quasi", str(z3["quasi"]))
m4.metric("davon mit Nullzelle", str(z3["nullzelle"]), help="Eine von sechs Einzelgruppen (eine der vier Arten, Zollhalt 1, Wochenende 1; keine Kreuzzelle) mit mindestens einem Container, in der alle gleich ausgehen.")
ns_dicht = list(range(7, 101)) + [200, 500, 1000, 3000]
st.plotly_chart(
    build_exp3(ev.haeufigkeit_je_n(fz, 1.0), ev.haeufigkeit_je_n(fz, 0.5), (ns_dicht, ev.cover_je_n(7, ns_dicht)), (n3, noise3)), width="stretch", key="chart_e3"
)
raten = _raten()
st.caption(
    "Band: Wilson-Intervall (95 %) für den Anteil bei der Zahl der Stichproben. Gestrichelt: ohne jeden Zusammenhang zwischen Merkmalen und Ausgang trennen n Punkte in allgemeiner Lage mit 7 Spalten nur bis n = 14 mit Wahrscheinlichkeit von mindestens 1/2 "
    "(Cover-Formel; bei n = 2p genau 1/2). Im Container-Vehikel " + (f"hatten bei diesem n und dieser Streuung {z3['nullzelle']} von {z3['getrennt']} getrennten Stichproben eine Nullzelle" if z3["getrennt"] else "gab es bei diesem n und dieser Streuung keine getrennte Stichprobe") + "; das passt zu Zellen mit sehr kleiner Standgeld-Rate, vor allem Umschlag "
    f"(wahre Rate im Mittel {ev.pct(raten['Umschlag'])}, Import {ev.pct(raten['Import'])}, Export {ev.pct(raten['Export'])}, Leer {ev.pct(raten['Leer'])}; Testmenge mit 100 000 Containern), ist aber eine Deutung, keine gemessene Ursache."
)
kind, text = ev.verdict_exp3(z3, fz["haeufigkeit_zweiter_strom"], noise3)
getattr(st, kind)(text)
st.markdown(f"**Eine frisch gezogene Stichprobe** (n = {ev.fmt_int(n3)}, Zufalls-Seed {int(seed)}, gleiche Streuung wie oben)")
if n3 >= 1000:
    with st.spinner(f"{ev.fmt_int(n3)} Container werden gezogen (rund {0.07 * n3 / 1000:.1f} s) …".replace(".", ",")):
        bsp3 = _e3_beispiel(n3, noise3, int(seed))
else:
    bsp3 = _e3_beispiel(n3, noise3, int(seed))
tab3 = pd.DataFrame([{"Art": a, "Container": n, "mit Standgeld": s, "Anteil": ev.pct(s / n) if n else "–"} for a, n, s in bsp3["je_art"]])
st.dataframe(tab3, hide_index=True, width="stretch")
if bsp3["art"] == "überlappend":
    st.info(f"Diese Stichprobe überlappt: der Schätzer existiert (Rezessions-LP: Optimum {ev.de(max(bsp3['lp_opt'], 0.0), 2)}).")
else:
    zellen = "; ".join(f"{nm}: {k} Container, {s} mit Standgeld" for nm, k, s in bsp3["nullzellen"]) or "keine"
    st.warning(
        f"Diese Stichprobe ist **{bsp3['art']} getrennt** (Optimum {ev.de(bsp3['lp_opt'], 1)}): der ML-Schätzer existiert nicht. Zellen, in denen alle gleich ausgehen: {zellen}."
        + (f" Spalten ohne Variation (weggelassen): {', '.join(bsp3['spalten_ohne_variation'])}." if bsp3["spalten_ohne_variation"] else "")
    )

st.markdown("---")

# --- Experiment 4: Was hilft? ---------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment: Was hilft bei getrennten Stichproben?")
st.caption(
    "**Vorgerechnet:** je n bis zu 150 getrennte Stichproben (Standardstreuung); Mehrverlust = mittlerer Log-Verlust gegen die wahre Wahrscheinlichkeit auf 100 000 Testcontainern minus die Entropie der wahren Wahrscheinlichkeiten (nat). "
    "Verfahren: ML mit der gängigen Abbruchregel (Kappung |η| ≤ 30, aus dem Gedächtnis nachgebaut), ML mit 25 stabilen Schritten, Firth, Ridge mit zwei festen Stärken (Spalten standardisiert), Streichen der Spalte mit dem größten |β| beim Abbruch."
)
h1, h2, h3 = st.columns(3)
n4 = h1.radio("Container je Stichprobe", C.E4_N, key="e4_n", horizontal=True)
k4 = h2.slider("Getrennte Stichproben (die ersten k)", *bounds("e4_k"), key="e4_k", step=C.E4_K_STEP)
verf = h3.selectbox("Verfahren hervorheben", C.E4_VERFAHREN, key="e4_verfahren")
rows4 = ev.exp4_tabelle(fz, n4, k4)
wh = fz["was_hilft"][str(n4)]
ref = fz["bezug_nicht_getrennt"]
grenze_mv = fz["meta"]["mehrverlust_grenze"]
sel = next(r for r in rows4 if r["verfahren"] == verf)
m1, m2, m3, m4 = st.columns(4)
m1.metric(f"{verf}: Mehrverlust (Median)", ev.de(sel["median"], 3))
m2.metric(f"{verf}: 90. Perzentil", ev.de(sel["p90"], 3))
m3.metric("Besser als Firth in", "–" if sel["besser"] is None else f"{sel['besser']} von {sel['k']}")
m4.metric("Grenze der Logit-Form", ev.de(grenze_mv, 3), help="Kleinster erwarteter Mehrverlust, den irgendein Logit in diesen Merkmalen erreichen kann (die wahre Form ist nicht logistisch).")
st.plotly_chart(build_exp4(rows4, grenze_mv), width="stretch", key="chart_e4")
st.dataframe(
    pd.DataFrame(
        [
            {"Verfahren": r["verfahren"], "Mehrverlust Median": ev.de(r["median"], 3), "Mittel": ev.de_mv(r["mittel"]), "90. Perzentil": ev.de(r["p90"], 3), "RMSE gegen Wahrheit (Median)": ev.de(r["rmse_median"], 3),
             "besser als Firth in": "–" if r["besser"] is None else f"{r['besser']} von {r['k']}"}
            for r in rows4
        ]
    ),
    hide_index=True, width="stretch",
)
st.caption(
    f"Gezogen wurden {wh['gezogen']} Stichproben, davon {wh['getrennt']} getrennt (Anteil {ev.pct(wh['getrennt'] / wh['gezogen'])}); die Tabelle nimmt die ersten {k4}. "
    f"Bezug ohne Trennung: {ref['stichproben']} nicht getrennte Stichproben mit n = {ref['n']}: Mehrverlust im Median ML {ev.de(ref['ml_median'], 3)}, Firth {ev.de(ref['firth_median'], 3)}; Firth besser in {ref['firth_besser']}."
)
kind, text = ev.verdict_exp4(rows4, ref, grenze_mv, n4, k4, verf)
getattr(st, kind)(text)
st.markdown(f"**Eine frisch gezogene getrennte Stichprobe** (n = {n4}, gesucht ab Zufalls-Seed {int(seed)}, höchstens {C.SEARCH_MAX} Versuche)")
with st.spinner("Beispielstichprobe wird gesucht (beim ersten Aufruf etwa 1 s: Testmenge und Logit-Grenze) …"):
    bsp4 = _e4_beispiel(n4, int(seed))
if bsp4 is None:
    st.info(f"In {C.SEARCH_MAX} Stichproben ab diesem Zufalls-Seed war keine getrennt.")
else:
    st.caption(f"Gefunden bei Zufalls-Seed {bsp4['seed']} (Versuch {bsp4['versuche']}). Container und Standgeld je Art: " + "; ".join(f"{a} {n} ({s})" for a, n, s in bsp4["je_art"]) + ".")
    zeilen4 = []
    for j, nm in enumerate(C.SPALTEN):
        zeile = {"Spalte": nm}
        for m in C.E4_VERFAHREN:
            zeile[m] = ev.de(bsp4["koef"][m][j], 2, sign=True)
        zeile["Grenze (pseudowahr)"] = ev.de(bsp4["grenze"][j], 2, sign=True)
        zeilen4.append(zeile)
    zeilen4.append({"Spalte": "Mehrverlust (nat)", **{m: ev.de(bsp4["mehrverlust"][m], 3) for m in C.E4_VERFAHREN}, "Grenze (pseudowahr)": ev.de(S.mehrverlust(bsp4["grenze"]), 3)})
    st.dataframe(pd.DataFrame(zeilen4), hide_index=True, width="stretch")
    zellen = "; ".join(f"{nm}: {k} Container, {s} mit Standgeld" for nm, k, s in bsp4["nullzellen"]) or "keine"
    st.caption(
        f"Zellen, in denen alle gleich ausgehen: {zellen}. Spalte mit dem größten |β| beim Abbruch: {bsp4['spalte']}; Wald z = {ev.de(bsp4['z'], 2)} (p = {ev.de(bsp4['p_wald'], 3)}), "
        f"Likelihood-Quotient beim Weglassen der Spalte {ev.de(bsp4['lq'], 2)} (p = {ev.de(bsp4['p_lq'], 4)})" + ("." if bsp4["lq_konv"] else "; das Modell ohne die Spalte hat selbst keinen endlichen Schätzer, die Zahl ist ein Abbruchwert.")
    )

st.markdown("---")

# --- Experiment 5: Faustregel ---------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment: Gilt „n ≥ 5p“?")
st.caption(
    "**Vorgerechnet:** n = 1000 Container, dazu p_R reine Rauschmerkmale N(0, 1) mit Wirkung null; ML mit höchstens 30 Schritten, 100 bis 150 Stichproben je Zeile. Der Grenzwert des Zollhalt-Koeffizienten ist pseudowahr "
    "(bestes Logit gegen die wahre Wahrscheinlichkeit). Die Quelle der Faustregel (Sur und Candès 2019) misst Gauß-Merkmale; hier Container-Merkmale, andere Anordnung, nicht dieselbe Zahl."
)
pr = st.select_slider("Rauschmerkmale p_R", options=C.E5_PR, key="e5_pr")
z5 = ev.faustregel_zeile(fz, pr)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Spalten p ÷ Container n", ev.de(z5["p_n"], 3))
m2.metric("Streuung ÷ Standardfehler", "–" if z5["sd_ueber_se"] is None else ev.de(z5["sd_ueber_se"], 2))
m3.metric("Wald |z| über 1,96 bei Rauschmerkmalen", "–" if z5["wald_ablehnung"] is None else ev.pct(z5["wald_ablehnung"]), help="Nominal 5 %.")
m4.metric("Zollhalt ÷ Grenzwert (Median)", ev.de(z5["zollhalt_verhaeltnis_median"], 2))
st.plotly_chart(build_exp5(fz["faustregel"], pr), width="stretch", key="chart_e5")
st.dataframe(
    pd.DataFrame(
        [
            {"Rauschmerkmale": r["p_r"], "Spalten p": r["spalten"], "p ÷ n": ev.de(r["p_n"], 3), "Stichproben": r["reps"], "ohne Schätzer": r["nicht_konvergiert"],
             "Streuung ÷ Standardfehler": "–" if r["sd_ueber_se"] is None else ev.de(r["sd_ueber_se"], 2), "Wald-Ablehnung": "–" if r["wald_ablehnung"] is None else ev.pct(r["wald_ablehnung"]),
             "Zollhalt ÷ Grenzwert": ev.de(r["zollhalt_verhaeltnis_median"], 2), "Export ÷ Grenzwert": ev.de(r["export_verhaeltnis_median"], 2)}
            for r in fz["faustregel"]
        ]
    ),
    hide_index=True, width="stretch",
)
kind, text = ev.verdict_exp5(z5)
getattr(st, kind)(text)

sync_query_params({k: st.session_state[k] for k in ("seed_input", "e1_beispiel", "e1_steps", "e2_daten", "e2_steps", "e2_regel", "e2_skala", "e3_n", "e3_streuung", "e4_n", "e4_verfahren", "e4_k", "e5_pr")})

st.markdown("---")

# --- Grenzen --------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
rg = ev.faustregel_zeile(fz, 200)
z100 = ev.haeufigkeit_zeile(fz, 100, 1.0)
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Ausgang ist stetig und sein Mittel linear im Merkmal** (Annahme der Demos davor) | Eine Gerade auf Standgeld ja/nein sagt für {ev.pct(br3000['ausserhalb_mittel'])} der Testcontainer einen Wert außerhalb von [0, 1] voraus (n = 3000, Container-Vehikel). | Das Logit, in dieser Demo. |
| **Der ML-Schätzer existiert** | Bei n = 100 trennten {z100['getrennt']} von {z100['reps']} Stichproben (Standardstreuung), alle nur quasi; dann ist jeder angezeigte Koeffizient ein Abbruchwert und der Wald-Test sagt „kein Effekt“. | Firth (hier gemessen) und Ridge (zwei feste Stärken); die Wahl der Stärke: [tikhonov-demo](https://github.com/sebastian-hanisch/tikhonov-demo), [parameterwahl-demo](https://github.com/sebastian-hanisch/parameterwahl-demo). |
| **Die Standardfehlerformel gilt, wenn n ≥ 5p** | Bei p/n = {ev.de(rg['p_n'], 2)} lehnt der Wald-Test bei Rauschmerkmalen in {ev.pct(rg['wald_ablehnung'])} statt 5 % ab, der Zollhalt-Koeffizient ist {ev.de(rg['zollhalt_verhaeltnis_median'], 2)}-fach überhöht. | Nur gemessen, nicht gelöst (keine Korrektur in dieser Demo). |
| **Die Form ist logistisch** | Die wahre Wahrscheinlichkeit ist hier nicht logistisch; selbst das beste Logit lässt einen Mehrverlust von {ev.de(fz['meta']['mehrverlust_grenze'], 3)} nat. | Kalibrierung und andere Links: Nachbar-Demos der Regressions-Demos. |
| **Standgeld hängt nur von den sieben Spalten ab** | Die zwölf Kunden wirken im Strom auf die Verweilzeit, stehen aber nicht im Modell (nicht getrennt gemessen, wie viel vom Rest daher kommt). | Gemischte Modelle: Nachbar-Demo. |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $n$ Container mit Merkmalszeilen $x_i$ und Ausgang $y_i \in \{0, 1\}$ (Standgeld ja/nein); gesucht ist $\beta$ mit der größten Log-Likelihood. $p$ zählt die Spalten **inklusive**
Achsenabschnitt (in ISLR2 zählt $p$ die Merkmale ohne Achsenabschnitt).

| Symbol | Bedeutung |
|---|---|
| $X$ | Designmatrix, Achsenabschnitt zuerst, $p$ Spalten |
| $\eta = X\beta$ | lineare Vorhersage |
| $p_i = 1/(1 + e^{-\eta_i})$ | Wahrscheinlichkeit für Standgeld |
| $w_i = p_i(1 - p_i)$ | Gewicht |
| $z_i = \eta_i + (y_i - p_i)/w_i$ | Arbeitsantwort |
| $D$ | Deviance |
| $s_i = 2y_i - 1$ | Vorzeichen des Ausgangs |

**Modell.** $y_i$ ist Bernoulli mit Wahrscheinlichkeit $p_i$ (kanonischer Link: Logit). Die Log-Likelihood ist

$$\ell(\beta) = \sum_i \left[ y_i \eta_i - \log\left(1 + e^{\eta_i}\right) \right], \qquad D = -2\,\ell(\beta).$$

**IRLS gleich Newton.** Gradient $X'(y - p)$ und Hesse-Matrix $-X'WX$; die zweite Ableitung enthält $y$ nicht. Darum ist der Newton-Schritt

$$\beta^{+} = \beta + (X'WX)^{-1} X'(y - p) = (X'WX)^{-1} X'W z,$$

also gewichtete kleinste Quadrate auf der Arbeitsantwort $z$. Bei einem 0/1-Merkmal ist der Grenzwert in geschlossener Form $\hat p_g = s_g / n_g$ je Gruppe und $\beta = (\ln \mathrm{Odds}_0,\ \ln \mathrm{OR})$.

**Wald und Likelihood-Quotient.** $z_j = \hat\beta_j / \sqrt{[(X'WX)^{-1}]_{jj}}$; der Likelihood-Quotient ist $D_0 - D$ mit der Deviance $D_0$ des Modells nur mit Achsenabschnitt, bei Gültigkeit der Nullhypothese $\chi^2$ mit $p - 1$ Freiheitsgraden.

**Trennung.** Die Daten heißen *vollständig getrennt*, wenn ein $\beta$ mit $s_i x_i'\beta > 0$ für alle $i$ existiert, *quasi-vollständig*, wenn nur $\ge$ gilt und für mindestens einen Punkt Gleichheit, sonst *überlappend*
(Begriffe nach Kosmidis und Firth 2019). Dann hat der ML-Schätzer mindestens eine unendliche Komponente. Prüfung hier als Linearprogramm

$$\max_d \ \sum_i s_i x_i' d \quad \text{unter} \quad s_i x_i' d \ge 0,\ \ -1 \le d_j \le 1;$$

ein Optimum größer als 0 (Schwelle 1e-9) heißt Trennung. Die Zulässigkeitsprüfung „gibt es $\beta$ mit $s_i x_i'\beta \ge 1$ für alle $i$“ erkennt nur die vollständige Trennung.

**Cover-Formel.** Der Anteil der $2^n$ Beschriftungen, die $n$ Punkte in allgemeiner Lage mit $p$ Spalten vollständig trennen, ist

$$\frac{2 \sum_{k=0}^{p-1} \binom{n-1}{k}}{2^n}.$$

Für $n = p$ ist er 1, für $n = 2p$ genau $1/2$; hier durch Aufzählung für kleine Fälle nachgerechnet, nicht allgemein bewiesen.

**Firth.** Modifizierte Score-Gleichung (Jeffreys-Strafe $\tfrac12 \log |X'WX|$ beim kanonischen Link)

$$U^{*}(\beta) = X'\left(y - p + h\left(\tfrac12 - p\right)\right) = 0,$$

mit den Hebeln $h_i$ der gewichteten Hat-Matrix. Im Zwei-Gruppen-Modell ist die Lösung $p_g = (s_g + \tfrac12)/(n_g + 1)$: ein halbes in jede Zelle. **Ridge:** Strafe $\tfrac12\lambda\sum_{j \ge 1}\beta_j^2$ auf standardisierten Spalten.

**Mehrverlust.** Gegen die wahre Wahrscheinlichkeit $\pi_i = \Phi((\mu_i - \ln 7)/s_i)$ des Stroms: der Mittelwert über Testcontainer von $-[\pi \ln \hat p + (1 - \pi)\ln(1 - \hat p)]$ minus die Entropie der $\pi_i$.

**Literatur** (nur so zitiert, wie gelesen): Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, Abschn. 4.4.1 (IRLS, Newton; Wortlaut gelesen). Kosmidis und Firth 2019, Abschn. 1 (Begriffe der Trennung,
Hinweis auf Wald-Tests; Albert und Anderson 1984 nur über diese Quelle). Sur und Candès 2019 sowie Candès und Sur 2020 (Faustregel, Schwelle bei Merkmalen ohne Zusammenhang; Wortlaut gelesen). Firth 1993 und Cover 1965 nur über Zweitquellen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
