"""Erzeugt data/lsp_frozen.json: die vorgerechneten Messungen der App (Häufigkeit der Trennung, was bei getrennten Stichproben hilft, die Faustregel n ≥ 5p, die Brücke vom stetigen Ausgang).

Aufruf (aus dem Repo-Ordner):
    python tools/lsp_make_frozen.py            schreibt data/lsp_frozen.json (Laufzeit je nach Rechner einige Minuten; ein BLAS-Faden)
    python tools/lsp_make_frozen.py --check    rechnet Teile neu und vergleicht sie mit der Datei als Band (Zählungen ± 2, Floats relativ 1e-6); Exitcode 1 bei Abweichung
    python tools/lsp_make_frozen.py --check --full   rechnet alles neu (wie ohne Argument, schreibt aber nichts) und vergleicht als Band

Zähleinheit der Messungen: die Stichprobe (ein Datensatz aus n Containern); alle Zufalls-Seeds stehen in `lsp_messungen.py` und in den Einträgen der Datei."""

import os

# Ein BLAS-Faden: bei vielen gleichzeitigen Prozessen kostete ein Fit mit 207 Spalten mit allen Fäden 4,5 s statt 0,02 s.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lsp_messungen as M  # noqa: E402

PFAD = ROOT / "data" / "lsp_frozen.json"


def _bandvergleich(alt: dict, neu: dict, zaehl_tol: int = 2, rel: float = 1e-6, pfad: str = "") -> list[str]:
    """Rekursiver Vergleich: ganze Zahlen (Zählungen) bis ± zaehl_tol, Floats relativ rel (plus 1e-9 absolut), Zeichenketten und Wahrheitswerte gleich."""
    fehler = []
    if isinstance(alt, dict):
        for k in alt:
            if k in neu:
                fehler += _bandvergleich(alt[k], neu[k], zaehl_tol, rel, f"{pfad}/{k}")
    elif isinstance(alt, list):
        if len(alt) != len(neu):
            fehler.append(f"{pfad}: Länge {len(alt)} gegen {len(neu)}")
        else:
            for i, (a, b) in enumerate(zip(alt, neu)):
                fehler += _bandvergleich(a, b, zaehl_tol, rel, f"{pfad}[{i}]")
    elif isinstance(alt, bool) or isinstance(alt, str) or alt is None:
        if alt != neu:
            fehler.append(f"{pfad}: {alt!r} gegen {neu!r}")
    elif isinstance(alt, int):
        if abs(alt - neu) > zaehl_tol:
            fehler.append(f"{pfad}: {alt} gegen {neu}")
    else:
        if abs(alt - neu) > rel * abs(alt) + 1e-9:
            fehler.append(f"{pfad}: {alt} gegen {neu}")
    return fehler


def pruefe_teile() -> list[str]:
    """Rechnet kleine Teile neu (Anfang der Messungen mit denselben Zufalls-Seeds) und vergleicht sie mit der Datei."""
    fz = json.loads(PFAD.read_text(encoding="utf-8"))
    fehler = []
    # Häufigkeit: die ersten 40 Stichproben bei n = 60 und n = 100, Standardstreuung (bitweise je Stichprobe, höchstens 2 abweichende)
    for n in (60, 100):
        eintrag = next(e for e in fz["haeufigkeit"] if e["noise"] == 1.0 and e["n"] == n)
        neu = M.haeufigkeit(1.0, n, 40)
        abw = sum(a != b for a, b in zip(eintrag["bits"][:40], neu["bits"]))
        if abw > 2:
            fehler.append(f"Häufigkeit n = {n}: {abw} von 40 Stichproben weichen ab")
    # Was hilft: die ersten 6 getrennten Stichproben bei n = 100
    alt = fz["was_hilft"]["100"]
    neu = M.was_hilft(100, ziel=6)
    for m, werte in neu["mehrverlust"].items():
        fehler += _bandvergleich(alt["mehrverlust"][m][:6], werte, rel=1e-4, pfad=f"was_hilft/100/{m}")
    # Brücke und Faustregel: kleine Wiederholungszahlen lassen sich nicht mit den vollen Zahlen vergleichen; hier nur die Meta-Zahlen
    import lsp_scenario as S

    fehler += _bandvergleich(fz["meta"]["beta_grenze"], [float(v) for v in S.logit_grenze()], rel=1e-6, pfad="meta/beta_grenze")
    fehler += _bandvergleich(fz["meta"]["mehrverlust_grenze"], S.mehrverlust(S.logit_grenze()), rel=1e-6, pfad="meta/mehrverlust_grenze")
    return fehler


if __name__ == "__main__":
    args = set(sys.argv[1:])
    if "--check" in args and "--full" not in args:
        fehler = pruefe_teile()
        print("OK: die Teile stimmen mit data/lsp_frozen.json überein (Band)" if not fehler else "\n".join(fehler))
        sys.exit(1 if fehler else 0)
    out = M.alles()
    if "--check" in args:
        fehler = _bandvergleich(json.loads(PFAD.read_text(encoding="utf-8")), out)
        print("OK: alles neu gerechnet und gleich (Band)" if not fehler else "\n".join(fehler[:50]))
        sys.exit(1 if fehler else 0)
    out["_hinweis"] = "erzeugt von tools/lsp_make_frozen.py; Zufalls-Seeds und Zahl der Stichproben stehen in den Einträgen und in lsp_messungen.py"
    PFAD.parent.mkdir(exist_ok=True)
    PFAD.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"geschrieben: {PFAD}")
