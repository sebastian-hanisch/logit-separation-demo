"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import special

import lsp_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


# --- Experiment 1 ---------------------------------------------------------------------------------------------------------------------------
def build_exp1_daten(res, steps):
    """Daten, Logit nach `steps` Schritten und Gerade auf y in {0, 1}; gestrichelt 0 und 1."""
    x, y = res["x"], res["y"]
    lo, hi = float(x.min()) - 0.5, float(x.max()) + 0.5
    xs = np.linspace(lo, hi, 200)
    b = res["pfad"][steps]
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=C.COLOR_GREY, dash="dot"))
    fig.add_hline(y=1, line=dict(color=C.COLOR_GREY, dash="dot"))
    fig.add_trace(go.Scatter(x=xs, y=res["b_ols"][0] + res["b_ols"][1] * xs, mode="lines", line=dict(color=C.COLOR_LINE, width=2, dash="dash"), name="Gerade (kleinste Quadrate)"))
    fig.add_trace(go.Scatter(x=xs, y=special.expit(b[0] + b[1] * xs), mode="lines", line=dict(color=C.COLOR_LOGIT, width=3), name=f"Logit nach {steps} Schritten"))
    zaehl = {}
    for xx, yy in zip(x, y):
        zaehl[(float(xx), float(yy))] = zaehl.get((float(xx), float(yy)), 0) + 1
    fig.add_trace(go.Scatter(x=[k[0] for k in zaehl], y=[k[1] for k in zaehl], mode="markers+text", text=[str(v) if v > 1 else "" for v in zaehl.values()], textposition="middle right",
                             marker=dict(size=11, color=C.COLOR_BLUE, line=dict(color="white", width=1)), name="Container (Zahl = Anzahl bei gleichen Werten)",
                             hovertemplate="x = %{x}, Standgeld = %{y}<extra></extra>"))
    ymin = min(-0.15, float(res["pred"].min()) - 0.05)
    ymax = max(1.15, float(res["pred"].max()) + 0.05)
    fig.update_xaxes(title_text="Merkmal x (Zollhalt 0/1 bzw. Vorlauf in Tagen)", range=[lo, hi])
    fig.update_yaxes(title_text="Standgeld (1) / Wahrscheinlichkeit", range=[ymin, ymax])
    return _base(fig)


def build_exp1_fehler(res, steps):
    """Abstand zum Grenzwert je Schritt, logarithmisch; der gewählte Schritt ist markiert."""
    k = np.arange(len(res["fehler"]))
    err = np.maximum(res["fehler"], 1e-17)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=err, mode="lines+markers", line=dict(color=C.COLOR_LOGIT, width=3), name="größter Abstand von β zum Grenzwert", hovertemplate="Schritt %{x}: %{y:.2e}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[steps], y=[err[steps]], mode="markers", marker=dict(size=14, color=C.COLOR_RED, symbol="circle-open", line=dict(width=3)), name="gewählter Schritt", hoverinfo="skip"))
    fig.update_xaxes(title_text="IRLS-Schritt", dtick=1)
    fig.update_yaxes(title_text="Abstand zum Grenzwert", type="log", exponentformat="power", dtick=2)
    return _base(fig, 300)


# --- Experiment 2 ---------------------------------------------------------------------------------------------------------------------------
def build_exp2_beta(res):
    k = np.arange(len(res["pfad"]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=res["pfad"][:, 1], mode="lines+markers", line=dict(color=C.COLOR_LOGIT, width=3), name="β₁", hovertemplate="Schritt %{x}: β₁ = %{y:.3f}<extra></extra>"))
    if res["formel_zuwachs"] is not None and len(k) > 7:
        k0 = 6
        fig.add_trace(go.Scatter(x=k[k0:], y=res["pfad"][k0, 1] + res["formel_zuwachs"] * (k[k0:] - k0), mode="lines", line=dict(color=C.COLOR_GREY, dash="dash"), name="Gerade mit Steigung 1/min|x|"))
    ks = res["k_stopp"]
    fig.add_trace(go.Scatter(x=[ks], y=[res["pfad"][ks, 1]], mode="markers", marker=dict(size=14, color=C.COLOR_RED, symbol="circle-open", line=dict(width=3)),
                             name=f"Abbruch ({res['regel']}) bei Schritt {ks}", hoverinfo="skip"))
    fig.update_xaxes(title_text="IRLS-Schritt")
    fig.update_yaxes(title_text="Koeffizient β₁")
    return _base(fig)


def build_exp2_zuwachs(res):
    k = np.arange(1, len(res["pfad"]))
    fig = go.Figure(go.Scatter(x=k, y=res["zuwachs"], mode="lines+markers", line=dict(color=C.COLOR_BLUE, width=3), name="Zuwachs von β₁ je Schritt", hovertemplate="Schritt %{x}: +%{y:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="IRLS-Schritt")
    fig.update_yaxes(title_text="Zuwachs von β₁")
    return _base(fig, 300)


def build_exp2_deviance(res):
    k = np.arange(len(res["dev"]))
    fig = go.Figure(go.Scatter(x=k, y=np.maximum(res["dev"], 1e-300), mode="lines", line=dict(color=C.COLOR_LINE, width=3), name="Deviance", hovertemplate="Schritt %{x}: %{y:.3e}<extra></extra>"))
    fig.update_xaxes(title_text="IRLS-Schritt")
    fig.update_yaxes(title_text="Deviance (logarithmisch)", type="log", exponentformat="power")
    return _base(fig, 300)


def build_exp2_tests(res):
    """p-Wert des Wald-Tests und des Likelihood-Quotient-Tests je Schritt."""
    k = np.arange(len(res["p_wald"]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=res["p_wald"], mode="lines", line=dict(color=C.COLOR_RED, width=3), name="Wald-Test", hovertemplate="Schritt %{x}: p = %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=k, y=res["p_lq"], mode="lines", line=dict(color=C.COLOR_LOGIT, width=3), name="Likelihood-Quotient-Test", hovertemplate="Schritt %{x}: p = %{y:.4f}<extra></extra>"))
    fig.add_hline(y=0.05, line=dict(color=C.COLOR_GREY, dash="dot"))
    fig.update_xaxes(title_text="IRLS-Schritt")
    fig.update_yaxes(title_text="p-Wert (gepunktet: 0,05)", range=[0, 1.02])
    return _base(fig, 300)


# --- Experiment 3 ---------------------------------------------------------------------------------------------------------------------------
def build_exp3(rows1, rows05, cover, gewaehlt):
    """Anteil getrennter Stichproben je n (Wilson-Band) für beide Streuungen; gestrichelt: der Anteil ohne Signal nach der Cover-Formel (p = 7 Spalten)."""
    fig = go.Figure()
    for rows, farbe, name in ((rows1, C.COLOR_RED, "Standardstreuung"), (rows05, C.COLOR_BLUE, "halbe Streuung")):
        xs = [r["n"] for r in rows]
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=[100 * r["hi"] for r in rows] + [100 * r["lo"] for r in rows][::-1], fill="toself", mode="lines", fillcolor="rgba(128,128,128,0.15)",
                                 line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=[100 * r["anteil"] for r in rows], mode="lines+markers", line=dict(color=farbe, width=3), name=name,
                                 customdata=[[r["getrennt"], r["reps"]] for r in rows], hovertemplate="n = %{x}: %{y:.1f} % (%{customdata[0]} von %{customdata[1]})<extra></extra>"))
    fig.add_trace(go.Scatter(x=cover[0], y=[100 * v for v in cover[1]], mode="lines", line=dict(color=C.COLOR_GREY, dash="dash"), name="ohne Signal (Cover, 7 Spalten)",
                             hovertemplate="n = %{x}: %{y:.2f} %<extra></extra>"))
    n, noise = gewaehlt
    sel = next((r for r in (rows1 if noise == 1.0 else rows05) if r["n"] == n), None)
    if sel is not None:
        fig.add_trace(go.Scatter(x=[n], y=[100 * sel["anteil"]], mode="markers", marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(color="black", width=3)), name="gewählt", hoverinfo="skip"))
    fig.update_xaxes(title_text="Container je Stichprobe n", type="log", tickvals=[7, 10, 30, 60, 100, 200, 500, 1000, 3000], ticktext=["7", "10", "30", "60", "100", "200", "500", "1000", "3000"])
    fig.update_yaxes(title_text="getrennte Stichproben (%)", range=[-2, 102])
    return _base(fig, 380)


# --- Experiment 4 ---------------------------------------------------------------------------------------------------------------------------
def build_exp4(rows, grenze_mv):
    """Median des Mehrverlusts je Verfahren (Balken) bis zum 90. Perzentil (Fehlerbalken); gepunktet: Grenze der Logit-Form."""
    fig = go.Figure(go.Bar(x=[r["verfahren"] for r in rows], y=[r["median"] for r in rows], marker_color=[C.METHOD_COLORS[r["verfahren"]] for r in rows],
                           error_y=dict(type="data", symmetric=False, array=[max(r["p90"] - r["median"], 0) for r in rows], arrayminus=[0] * len(rows), color="rgba(80,80,80,0.8)"),
                           hovertemplate="%{x}: Median %{y:.3f}<extra></extra>", name="Median (Fehlerbalken bis 90. Perzentil)"))
    fig.add_hline(y=grenze_mv, line=dict(color="black", dash="dot"), annotation_text="Grenze der Logit-Form", annotation_position="top left")
    fig.update_yaxes(title_text="Mehrverlust gegen die Wahrheit (nat)")
    return _base(fig, 340)


# --- Experiment 5 ---------------------------------------------------------------------------------------------------------------------------
def build_exp5(rows, gewaehlt):
    """Links: Streuung der Rausch-Schätzer geteilt durch den mittleren Standardfehler; rechts: Zollhalt-Koeffizient geteilt durch den Grenzwert; je gegen p/n."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Rausch-Schätzer: Streuung ÷ Standardfehler", "Zollhalt-Koeffizient ÷ Grenzwert"))
    pn = [r["p_n"] for r in rows]
    sdse = [r["sd_ueber_se"] if r["sd_ueber_se"] is not None else None for r in rows]
    fig.add_trace(go.Scatter(x=pn, y=sdse, mode="lines+markers", line=dict(color=C.COLOR_RED, width=3), name="Streuung ÷ Standardfehler", hovertemplate="p/n = %{x:.3f}: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pn, y=[r["zollhalt_verhaeltnis_median"] for r in rows], mode="lines+markers", line=dict(color=C.COLOR_LOGIT, width=3), name="Zollhalt ÷ Grenzwert (Median)",
                             hovertemplate="p/n = %{x:.3f}: %{y:.2f}<extra></extra>"), row=1, col=2)
    for c in (1, 2):
        fig.add_hline(y=1.0, line=dict(color=C.COLOR_GREY, dash="dot"), row=1, col=c)
    sel = next((r for r in rows if r["p_r"] == gewaehlt), None)
    if sel is not None:
        if sel["sd_ueber_se"] is not None:
            fig.add_trace(go.Scatter(x=[sel["p_n"]], y=[sel["sd_ueber_se"]], mode="markers", marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(color="black", width=3)), showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[sel["p_n"]], y=[sel["zollhalt_verhaeltnis_median"]], mode="markers", marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(color="black", width=3)), showlegend=False, hoverinfo="skip"), row=1, col=2)
    fig.update_xaxes(title_text="Spalten p ÷ Container n")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.3), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)
