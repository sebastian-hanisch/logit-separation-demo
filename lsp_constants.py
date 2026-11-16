"""Konstanten: Container-Vehikel, Regler-Grenzen, feste Annahmen, Farben und Presets.

Das Vehikel ist ein erfundener Containerstrom (kein Echtdatenbezug): Art, Kunde, Zollhalt, Wochenende und Vorlauf bestimmen die log-Verweilzeit; Standgeld wird fällig,
wenn die Verweilzeit über STAND_TAGE Tage liegt. Die Parameter des Stroms folgen den Annahmen der `verweilzeit-demo` (dort erfunden, hier neu geschrieben)."""

# --- Vehikel: Merkmale und Verweilzeit -------------------------------------------------------------------------------------------------
ART_LABELS = ("Import", "Export", "Umschlag", "Leer")
ART_P = (0.35, 0.30, 0.25, 0.10)             # Anteile der Containerarten
K_KUNDEN = 12                                 # Kunden mit je einem Effekt auf die log-Verweilzeit (im Modell nicht als Spalte enthalten)
KUNDEN_SEED = 12345
U_SD = 0.30                                   # Streuung der Kundeneffekte (log)
U_SCALE = (1.0, 0.3, 0.3, 1.0)                # Kundeneffekt je Art
S_BASE = (0.55, 0.25, 0.35, 0.90)             # Streuung der log-Verweilzeit je Art
P_ZOLL = 0.25                                 # Zollhalt bei Import
STAND_TAGE = 7.0                              # Standgeld fällig, wenn die Verweilzeit über so viele Tage liegt
SPALTEN = ("Achsenabschnitt", "Export", "Umschlag", "Leer", "Zollhalt", "Wochenende", "Vorlauf")
N_DRAWS = 7                                   # Ziehungen der SplitMix64-Folge je Container: Art, Kunde, Zoll, Wochentag, Vorlauf, zwei für die Normalzahl
TEST_N = 100_000                              # Testcontainer (nur Merkmale) für den Mehrverlust gegen die wahre Wahrscheinlichkeit
TEST_SEED = 999
LIMIT_SEED = 123                              # Merkmale für die Logit-Grenze (pseudowahre Koeffizienten)
LIMIT_N = 400_000

# --- Handbeispiele (gesetzte Zahlen, kein Generatorlauf) -----------------------------------------------------------------------------------
ZOLL9_X = (0, 0, 0, 0, 1, 1, 1, 1, 1)
ZOLL9_Y = (1, 0, 0, 0, 1, 1, 1, 1, 0)         # Zollhalt 0: 1 von 4 mit Standgeld; Zollhalt 1: 4 von 5
VORLAUF8_X = (1, 2, 3, 4, 5, 6, 7, 8)
VORLAUF8_Y = (0, 0, 0, 1, 0, 1, 1, 1)
SEP6_X = (-3, -2, -1, 1, 2, 3)
SEP6_Y = (0, 0, 0, 1, 1, 1)
QUASI8_X = (-3, -2, -1, 0, 0, 1, 2, 3)
QUASI8_Y = (0, 0, 0, 0, 1, 1, 1, 1)
CONV_TOL = 1e-10                              # „konvergiert“: größte Änderung der Koeffizienten eines Schritts unter dieser Schwelle
DEV_RULE_EPS = 1e-8                           # Deviance-Regel: |ΔD| / (|D| + 0,1) unter dieser Schwelle
COEF_RULE_EPS = 1e-8                          # Koeffizientenregel: größte Änderung eines Koeffizienten unter dieser Schwelle
CAP_ETA = 30.0                                # Kappung |η| ≤ 30 im Nachbau der gängigen Regel (nur Exp. 4)
LP_SCHWELLE = 1e-9                            # Optimum des Rezessions-LP über dieser Schwelle: getrennt

# --- Regler -------------------------------------------------------------------------------------------------------------------------------
SEED_MAX, DEFAULT_SEED = 9999, 1
E1_BEISPIELE = ("Zollhalt (9 Container)", "Vorlauf (8 Container)")
E1_STEPS_MIN, E1_STEPS_MAX, E1_STEPS_DEFAULT = 0, 8, 6
E2_DATEN = ("vollständig (6 Punkte)", "quasi (8 Punkte)")
E2_STEPS_MIN, E2_STEPS_MAX, E2_STEPS_DEFAULT = 1, 200, 30
E2_REGELN = ("Deviance-Regel", "Feste Schrittzahl", "Koeffizientenänderung")
E2_SKALEN = (0.5, 1.0, 2.0)
E3_N = (30, 60, 100, 200, 500, 1000, 3000)
E3_STREUUNG = ("Standard (× 1)", "halb (× 0,5)")
E3_NOISE = {"Standard (× 1)": 1.0, "halb (× 0,5)": 0.5}
E4_N = (60, 100)
E4_VERFAHREN = ("ML, Abbruchregel", "ML, 25 Schritte", "Firth", "Ridge 0,1", "Ridge 1", "Streichen")
E4_K_MIN, E4_K_MAX, E4_K_STEP, E4_K_DEFAULT = 50, 150, 10, 150
E5_PR = (0, 50, 100, 200)
SEARCH_MAX = 300                               # so viele Stichproben probiert die Beispielsuche (erste getrennte Stichprobe)

# --- Darstellung ------------------------------------------------------------------------------------------------------------------------
COLOR_LOGIT = "#2F6B65"
COLOR_LINE = "#c77700"
COLOR_NEWTON = "#7a3fb0"
COLOR_BAND = "rgba(76,120,168,0.18)"
COLOR_BLUE = "#4c78a8"
COLOR_GREY = "#8a94a3"
COLOR_RED = "#c0392b"
METHOD_COLORS = {"ML, Abbruchregel": "#c0392b", "ML, 25 Schritte": "#e08a7d", "Firth": "#2F6B65", "Ridge 0,1": "#4c78a8", "Ridge 1": "#7fa8d4", "Streichen": "#c77700"}

PRESETS = {
    "Von Hand (Zollhalt, 9 Container)": {},
    "Trennung (6 Punkte)": {"e2_daten": E2_DATEN[0], "e2_steps": 25},
    "Quasi-Trennung (8 Punkte)": {"e2_daten": E2_DATEN[1], "e2_steps": 40},
    "Container n = 100": {"e3_n": 100, "e3_streuung": E3_STREUUNG[0]},
    "Container n = 1000": {"e3_n": 1000, "e3_streuung": E3_STREUUNG[0]},
    "Halbe Streuung": {"e3_n": 500, "e3_streuung": E3_STREUUNG[1]},
    "Viele Merkmale (p/n = 0,2)": {"e5_pr": 200},
}
PRESET_HELP = {
    "Von Hand (Zollhalt, 9 Container)": "Voreinstellung: neun Container, Standgeld nach Zollhalt; IRLS gleich Newton, sechs Schritte bis zum Grenzwert.",
    "Trennung (6 Punkte)": "Sechs Punkte, die der Wert des Merkmals vollständig trennt: der Koeffizient wächst mit jedem Schritt um 1.",
    "Quasi-Trennung (8 Punkte)": "Acht Punkte, zwei davon beim Merkmalswert 0 mit verschiedenem Ausgang: nur quasi getrennt, die einfache Zulässigkeitsprüfung sieht es nicht.",
    "Container n = 100": "Wie oft trennt eine Stichprobe aus 100 Containern? Vorgerechnete Zählung und eine frisch gezogene Stichprobe.",
    "Container n = 1000": "Dieselbe Frage bei 1000 Containern (Standardstreuung): in den vorgerechneten Stichproben keine getrennte.",
    "Halbe Streuung": "500 Container, aber nur halbe Streuung der Log-Verweilzeit: das stärkere Signal trennt wieder häufiger.",
    "Viele Merkmale (p/n = 0,2)": "1000 Container mit 200 reinen Rauschmerkmalen zusätzlich: die Faustregel n ≥ 5p reicht für die Standardfehler nicht.",
}
