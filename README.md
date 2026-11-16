# Logit und Trennung – wenn der Schätzer nicht existiert – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-logit-separation-demo.streamlit.app/)** -->

Demo der Regressions-Demos (Konzepte) für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Demos davor setzen einen stetigen Ausgang voraus, dessen Mittel linear im Merkmal ist
(zuletzt [quantil-ueberdeckung-demo](https://github.com/sebastian-hanisch/quantil-ueberdeckung-demo)); hier hebt das Logit diese Annahme auf und zeigt, was dabei neu schiefgehen kann: **trennen die Daten, existiert der Schätzer nicht.**
Vehikel ist die Container-Verweilzeit (Generator neu geschrieben nach den erfundenen Annahmen der [verweilzeit-demo](https://github.com/sebastian-hanisch/verweilzeit-demo)); der Ausgang heißt „Standgeld fällig“ (Verweilzeit über 7 Tage),
dazu kommen Handbeispiele mit gesetzten Zahlen, die sich von Hand in Brüchen nachrechnen lassen.

## Warum dieses Problem

Eine Gerade auf Standgeld ja/nein sagt für 15,5 % der Testcontainer einen Wert unter 0 oder über 1 voraus; also passt man ein Logit an, per IRLS, und beim Logit ist das ein Newton-Verfahren. Dann geschieht etwas, das bei stetigen
Ausgängen nicht vorkommt: Gibt es in der Stichprobe eine Containerart ohne ein einziges Standgeld (im Strom dieser Demo kommt das bei Umschlag mit einer wahren Rate von 3,3 % vor: bei n = 100 haben 147 von 151 getrennten Stichproben eine Gruppe (eine der vier Arten, Zollhalt 1 oder Wochenende 1), in der alle gleich ausgehen), so wächst der Koeffizient mit jedem Schritt weiter,
die Software meldet trotzdem Zahlen, und der Wald-Test sagt „kein Effekt“, obwohl die Daten vollkommen trennen. Die Demo zeigt das an sechs Punkten, misst, wie oft es im Container-Strom vorkommt (bei n = 100 in 151 von 300 und in 136 von 300 Stichproben aus zwei Zufallsströmen)
und was bei getrennten Stichproben hilft. Nichts davon ist wissenschaftlich neu: Trennung ist Lehrbuchstoff; neu ist nur die Vorführbarkeit mit Zähleinheiten.

## Modell

- **Ausgang** y ∈ {0, 1}: Standgeld fällig (1), wenn die Verweilzeit über 7 Tage liegt. **Spalten** von X: Achsenabschnitt zuerst, dann Export, Umschlag, Leer (Referenz Import), Zollhalt, Wochenende, Vorlauf in Tagen; **p** zählt die Spalten
  inklusive Achsenabschnitt (in ISLR2 zählt p die Merkmale ohne Achsenabschnitt).
- **Logit:** p_i = 1 / (1 + e^(−η_i)), η = Xβ; Log-Likelihood ℓ(β) = Σ [y_i η_i − log(1 + e^(η_i))]; Gewicht w_i = p_i (1 − p_i); IRLS-Schritt β⁺ = β + (X'WX)⁻¹ X'(y − p). Beim kanonischen Link ist das der Newton-Schritt (die zweite Ableitung enthält y nicht).
  Gerechnet wird mit stabilen Formeln (w = σ(η) σ(−η), Residuum y σ(−η) − (1 − y) σ(η)), ohne Kappung und ohne Schrittweitenhalbierung.
- **Trennung** (Begriffe nach Kosmidis und Firth 2019): *vollständig*, wenn ein β mit s_i x_i'β > 0 für alle i existiert (s_i = 2y_i − 1); *quasi-vollständig*, wenn nur „≥“ gilt und für mindestens einen Punkt Gleichheit; sonst *überlappend*. Prüfung hier als
  Linearprogramm „maximiere Σ s_i x_i'd unter s_i x_i'd ≥ 0, −1 ≤ d_j ≤ 1“ (HiGHS, `scipy.optimize.linprog`); ein Optimum über 1e-9 heißt Trennung, der Schätzer existiert nicht. Die Zulässigkeitsprüfung „gibt es β mit s_i x_i'β ≥ 1“ erkennt nur die vollständige Trennung.
- **Container-Strom:** Art Import 35 % / Export 30 % / Umschlag 25 % / Leer 10 %, zwölf Kunden mit Effekt auf die log-Verweilzeit (nicht im Modell), Zollhalt bei Import (25 %), Wochenende, Vorlauf bei Export und Umschlag; die log-Verweilzeit ist normalverteilt mit Streuung 0,55 / 0,25 / 0,35 / 0,90 je Art.
  Die wahre Wahrscheinlichkeit Φ((μ − ln 7)/s) ist bekannt, aber nicht logistisch in den Merkmalen; darum bleibt jedem Logit ein Rest („Grenze der Logit-Form“). **Alle Parameter des Stroms sind erfunden.**
- **Zufall** nur aus dem ganzzahligen SplitMix64 (`lsp_rng.py`, wörtliche Kopie der geprüften Vorlage plus ein Zusatz); je Container sieben Ziehungen in fester Reihenfolge; die Normalzahl der Verweilzeit aus `SplitMix64.normal()` (Dezimalrechnung, auf Windows und Linux bitgleich).
  Die Rauschmerkmale der Messung zur Faustregel kommen aus `lsp_rngvec.py` (Box-Muller mit einem Logarithmus und Kosinus nur aus Grundrechenarten; gegen `numpy.log` und `numpy.cos` getestet).

## Methodik

1. **Von Hand** (neun Container, Zollhalt; acht Container, Vorlauf): IRLS-Pfad, Newton-Pfad mit numerisch abgeleiteter Hesse-Matrix, Grenzwert (geschlossene Form beim 0/1-Merkmal), Wald und Likelihood-Quotient, Gerade gegen Logit; der erste Schritt zusätzlich in Brüchen.
2. **Trennung** (sechs Punkte, acht Punkte, Skalierung 0,5 bis 2): Pfad bis 200 Schritte, drei Abbruchregeln (Deviance-Regel |ΔD| / (|D| + 0,1) < 1e-8, feste Schrittzahl, Koeffizientenänderung < 1e-8), Wald z und Likelihood-Quotient je Schritt, Rezessions-LP.
3. **Häufigkeit** (vorgerechnet, Zähleinheit die Stichprobe): Anteil getrennter Stichproben je n und Streuung; dazu eine frisch gezogene Stichprobe mit ihrer Nullzelle.
4. **Was hilft** (vorgerechnet, je n bis zu 150 getrennte Stichproben): ML mit einer Regel, die Wahrscheinlichkeiten bei |η| ≤ 30 kappt und nach der Deviance-Regel abbricht (**aus dem Gedächtnis nachgebaut, nicht gegen eine bestimmte Software geprüft**), ML mit 25 stabilen Schritten,
   Firth (Newton mit Halbierung auf ℓ + ½ log|X'WX|), Ridge mit zwei festen Stärken (λ = 0,1 und 1, Spalten standardisiert), Streichen der Spalte mit dem größten |β| beim Abbruch. **Mehrverlust** = mittlerer Log-Verlust gegen die wahre Wahrscheinlichkeit auf 100 000 Testcontainern (nur Merkmale)
   minus die Entropie der wahren Wahrscheinlichkeiten, in nat. **Grenze der Logit-Form** = das Logit mit dem kleinsten erwarteten Log-Verlust gegen die wahre Wahrscheinlichkeit (IRLS mit der wahren Wahrscheinlichkeit als Ausgang auf 400 000 Merkmalssätzen).
5. **Faustregel n ≥ 5p** (vorgerechnet): n = 1000 Container plus 0 bis 200 reine Rauschmerkmale N(0, 1); ML mit höchstens 30 Schritten.

Die schweren Messungen (Hunderte Stichproben mit bis zu 3000 Containern, 100 Stichproben mit 207 Spalten) sind **vorgerechnet** (`tools/lsp_make_frozen.py` schreibt `data/lsp_frozen.json`; Laufzeit einige Minuten, ein BLAS-Faden) und in der App so gekennzeichnet.
Live gerechnet wird, was klein ist: die Handbeispiele, ein Pfad von bis zu 200 Schritten, eine frisch gezogene Stichprobe. Beim Bau gemessen (Windows, nicht im Test): ein Pfad mit 200 Schritten 0,03 s, eine Stichprobe mit 3000 Containern 0,2 s, die Beispielsuche beim ersten Aufruf 1 s (Testmenge und Logit-Grenze).

## Befunde (gemessen, keine Behauptungen)

Zähleinheit, wo nichts anderes steht: die **Stichprobe** (ein Datensatz aus n Containern); Standardstreuung der Log-Verweilzeit, sieben Spalten, Standgeld = Verweilzeit über 7 Tage, Container-Vehikel. Die Spalte „Test“ nennt, wo die Zahl geprüft wird (exakt = in Brüchen oder
geschlossener Form; Band = Neu-Rechnung mit Toleranz, siehe „Rechenregeln und Hinweise“; vorgerechnet = Konsistenzprüfung gegen die Datei, `tests/test_claims.py`).

### Handbeispiele (gesetzte Zahlen, kein Generatorlauf)

| Frage | Befund | Test |
|---|---|---|
| Neun Container: Zollhalt 0 vier Container mit einem Standgeld, Zollhalt 1 fünf Container mit vier. Erster IRLS-Schritt | β = (−1; 11/5) in Brüchen | exakt, `test_logit.py` |
| Grenzwert | β = (ln 1/3; ln 12): Odds 1/3 ohne und 4 mit Zollhalt, Odds-Ratio 12; IRLS erreicht ihn in 6 Schritten (Änderung unter 1e-13) | exakt (Form), Band 5 bis 7 Schritte, `test_logit.py` |
| IRLS gegen Newton (Hesse-Matrix per Differenzenquotient) | größter Unterschied über 8 Schritte 3,1e−10 (Zollhalt) und 3,3e−10 (Vorlauf), Schranke 1e-7 | Band, `test_claims.py` |
| Wald und Likelihood-Quotient (nicht getrennt, Zollhalt-Beispiel) | z = 1,546 (p = 0,122); Likelihood-Quotient 2,863 (p = 0,091); der exakte Test nach Fisher p = 0,206 | Band 5e-4, `test_logit.py`, `test_claims.py` |
| Gerade im Vorlauf-Beispiel (acht Container) | Steigung 1/6, Achsenabschnitt −1/4: bei Vorlauf 1 der Wert −1/12, bei Vorlauf 8 der Wert 13/12 | exakt, `test_logit.py` |
| Sechs Punkte x = −3, −2, −1, 1, 2, 3 mit y = 0, 0, 0, 1, 1, 1: erster Schritt | β = (0; 6/7) (Start bei β = 0 mit Deviance 12 ln 2 = 8,318) | exakt, `test_logit.py` |
| Zuwachs von β₁ je Schritt (ohne Kappung, stabile Formeln) | 1,000000 (zwischen 0,999999 und 1,000001) in den Schritten 40 bis 100; bei Skalierung des Merkmals mit 0,5 / 1 / 2 gleich 2 / 1 / 0,5 (1/kleinstes \|x\|, nur bei symmetrischen Daten) | Band 1e-6, `test_logit.py` |
| Abbruchregeln auf den sechs Punkten | Deviance-Regel hält bei Schritt 24 mit β₁ = 22,96; 25 Schritte: 23,96; 100: 98,96; 200: 198,96; die Koeffizientenregel greift in 200 Schritten nicht; der gekappte Nachbau hält bei Schritt 24 mit β₁ = 22,956 | Band, `test_logit.py` |
| Wald-Test und Likelihood-Quotient bei diesen Punkten | Wald z = 1,39 nach Schritt 1, danach fallend, bei Schritt 24 gleich 0,0003 (p = 1,00); Likelihood-Quotient wächst auf 12 ln 2 = 8,318 (p = 0,0039); größtes \|z\| in den ersten 60 Schritten unter 1,45 | Band, `test_logit.py`, `test_claims.py` |
| Acht Punkte x = −3, −2, −1, 0, 0, 1, 2, 3 mit y = 0, 0, 0, 0, 1, 1, 1, 1 | nur quasi getrennt: die Zulässigkeitsprüfung sieht es nicht, das Rezessions-LP schon; β₁ = 38,96 nach 40 Schritten, β₀ bleibt 0; dreht man y bei x = −1 auf 1, überlappen die Klassen und β ist endlich | Band, `test_logit.py` |
| Asymmetrisch x = −3, −2, −1, 2, 3, 4 | Zuwachs je Schritt (−1/3; 2/3) aus den zwei Stützpunkten; die Trennstelle strebt gegen 0,5 | exakt (Brüche) und Band 1e-6, `test_logit.py` |
| Firth bei Nullzelle (0 von 5 gegen 3 von 5) | p₀ = 1/12, p₁ = 7/12, Odds-Ratio 77/5 (ein halbes in jede Zelle); in 200 zufälligen Zählungen gleich der Iteration (1e-8) | exakt und Band, `test_logit.py`, `test_oracle_logit_separation.py` |
| Firth bei den sechs Punkten | β = (0; 0,938) | Band 1e-3, `test_logit.py` |
| Cover-Formel (kein Zusammenhang zwischen Merkmalen und Ausgang) | trennbare Beschriftungen: 12 von 64 (n = 6, p = 2), 32 von 64 (n = 6, p = 3) durch Aufzählung; für n = p alle, für n = 2p die Hälfte | exakt, `test_logit.py` |

### Container-Strom (vorgerechnet, mit Neu-Rechnung eines Teils als Band)

**Gerade auf Standgeld** (Testmenge mit 100 000 Containern, Mittel über die Stichproben): Anteil der Testcontainer mit Vorhersage unter 0 oder über 1 bei n = 100: 16,7 % (100 Stichproben), n = 500: 15,6 % (60), n = 3000: 15,5 % (40).
**Wahre Standgeld-Rate je Art** (Mittel über die Testmenge): Import 23,3 %, Export 25,9 %, Umschlag 3,3 %, Leer 50,5 %; insgesamt 21,8 %.

**Wie oft trennen die Daten?** Anteil getrennter Stichproben (Rezessions-LP, vollständig oder quasi); Band = Wilson-Intervall 95 %; „Nullzelle“ = eine von sechs Einzelgruppen (eine der vier Arten, Zollhalt 1, Wochenende 1; keine Kreuzzelle) mit mindestens einem Container, in der alle gleich ausgehen:

| n | Standardstreuung: getrennt | davon nur quasi | davon mit Nullzelle | halbe Streuung: getrennt | davon nur quasi | davon mit Nullzelle |
|---|---|---|---|---|---|---|
| 30 | 287 von 300 (95,7 %) | 272 | 280 | 300 von 300 (100,0 %) | 240 | 298 |
| 60 | 229 von 300 (76,3 %) | 229 | 216 | 297 von 300 (99,0 %) | 294 | 292 |
| 100 | 151 von 300 (50,3 %) | 151 | 147 | 288 von 300 (96,0 %) | 288 | 284 |
| 200 | 71 von 300 (23,7 %) | 71 | 71 | 259 von 300 (86,3 %) | 259 | 256 |
| 500 | 5 von 200 (2,5 %) | 5 | 5 | 139 von 200 (69,5 %) | 139 | 139 |
| 1000 | 0 von 200 | 0 | 0 | 105 von 200 (52,5 %) | 105 | 105 |
| 3000 | 0 von 100 | 0 | 0 | 13 von 100 (13,0 %) | 13 | 13 |

Ab n = 100 ist jede getrennte Stichprobe nur quasi getrennt (in beiden Streuungen und allen n). Bei n = 30 fehlt in 32 von 300 Stichproben eine ganze Spalte (Merkmal ohne Variation; sie wird weggelassen), bei n = 60 in 1 von 300.
Ein zweiter Zufallsstrom mit 300 anderen Stichproben bei n = 100 gab 136 von 300 (45,3 %) gegen 151 von 300: Die Zahl ist ein Band (Standardfehler eines Anteils bei 300 Stichproben um 0,5 etwa 2,9 Prozentpunkte), nicht eine Stelle genau.
„0 von 200“ heißt nicht „kann nicht vorkommen“: der obere Rand des 95-%-Bandes liegt bei 1,9 %. Die Schwelle 1e-9 des Rezessions-LP liegt in einer Lücke: in 600 Stichproben (n = 100 und 200, beide Streuungen) ist das Optimum entweder mindestens 4,0 (365 Stichproben)
oder null (235 Stichproben).

**Was hilft bei getrennten Stichproben?** Mehrverlust gegen die wahre Wahrscheinlichkeit (nat), Median über je 150 getrennte Stichproben (Standardstreuung); „besser als Firth in“ = Zahl der Stichproben, in denen das Verfahren einen kleineren Mehrverlust hat als Firth:

| Verfahren | n = 60: Median | n = 60: besser als Firth in | n = 100: Median | n = 100: besser als Firth in |
|---|---|---|---|---|
| ML, Abbruchregel (Kappung, aus dem Gedächtnis nachgebaut) | 0,711 | 0 von 150 | 0,184 | 0 von 150 |
| ML, 25 stabile Schritte | 0,982 | 0 von 150 | 0,241 | 0 von 150 |
| Firth | 0,074 | – | 0,055 | – |
| Ridge λ = 0,1 (Spalten standardisiert) | 0,105 | 18 von 150 | 0,067 | 6 von 150 |
| Ridge λ = 1 (Spalten standardisiert) | 0,092 | 19 von 150 | 0,071 | 15 von 150 |
| Streichen der Spalte mit dem größten \|β\| | 0,170 | 4 von 150 | 0,110 | 9 von 150 |

Gezogen wurden 200 Stichproben (n = 60; 150 getrennt, 75,0 %) und 306 (n = 100; 150 getrennt, 49,0 %). Der Mittelwert des Mehrverlusts ist bei ML von Ausreißern bestimmt (n = 60: 7,4e+08 bei der Abbruchregel, 2,40 bei 25 Schritten, gegen Median 0,711 und 0,982): die Mediane der ML-Zeilen sind wegen des schweren Rands keine auf drei Stellen gesicherten Werte (150 getrennte Stichproben je Zelle); gesichert ist die Ordnung, ML ist in keiner der 150 Stichproben besser als Firth. Ein getrennter Koeffizient kann die Vorhersage für ganze Arten
verderben. **Bezug ohne Trennung** (150 nicht getrennte Stichproben, n = 100): Mehrverlust im Median ML 0,062, Firth 0,057; Firth besser als ML in 107 von 150. **Grenze der Logit-Form: 0,024**; keine Methode erreicht sie, Ridge hängt an der Stärke (nur zwei feste Werte
gemessen), „Streichen“ ist nur als Streichen der Spalte mit dem größten |β| beim Abbruch gemessen (nicht jede denkbare Streichregel).
**Beispiel** (n = 100, erste getrennte Stichprobe ab Zufalls-Seed 1, in der App einstellbar): Umschlag 20 Container, kein Standgeld; Koeffizient der Spalte Umschlag −19,65 bei der Abbruchregel (−26,65 bei 25 Schritten), Firth −3,95, Ridge λ = 0,1 −5,50, λ = 1 −2,67, Wald z = −0,01 (p = 0,995),
Likelihood-Quotient beim Weglassen der Spalte 8,31 (p = 0,0040). Mehrverlust in dieser Stichprobe: ML 0,161, ML mit 25 Schritten 0,219, Firth 0,039, Ridge 0,053 und 0,059, Streichen 0,101.

**Gilt „n ≥ 5p“?** n = 1000 Container plus p_R reine Rauschmerkmale N(0, 1) (Wirkung null); „Streuung ÷ Standardfehler“ = Standardabweichung der Rausch-Schätzer geteilt durch den mittleren Standardfehler; Zollhalt ÷ Grenzwert = Median des Zollhalt-Koeffizienten geteilt durch den pseudowahren Wert (1,925):

| Rauschmerkmale | Spalten p | p ÷ n | Stichproben | Streuung ÷ Standardfehler | Wald \|z\| über 1,96 bei Rauschmerkmalen | Zollhalt ÷ Grenzwert |
|---|---|---|---|---|---|---|
| 0 | 7 | 0,007 | 150 | – | – | 0,98 |
| 50 | 57 | 0,057 | 150 | 1,07 | 6,1 % | 1,07 |
| 100 | 107 | 0,107 | 120 | 1,12 | 7,6 % | 1,23 |
| 200 | 207 | 0,207 | 100 | 1,38 | 14,0 % | 1,85 |

Die Verhältnisse Zollhalt ÷ Grenzwert sind Mediane aus 100 bis 150 Stichproben und nur auf etwa eine Stelle gesichert. Der Schätzer existierte in allen 520 Stichproben (nicht konvergiert in 30 Schritten: 0). Qualitativ wie die Quelle (Sur und Candès 2019: Streuung über dem klassischen Standardfehler, Aufblähung des Koeffizienten, dort bei p/n = 1/5 klassisch 2,66 gegen simuliert rund 4,75 mit Gauß-Merkmalen, n = 4000, p = 800);
quantitativ schwächer (hier 1,38 bei p/n = 0,207), andere Anordnung, nicht dieselbe Zahl.

**scikit-learn** (Version 1.9.1, `LogisticRegression` mit `penalty=None`, lbfgs, Windows) auf die sechs Punkte: β₁ = 8,35 (`max_iter` 100, `tol` 1e-4), 17,36 (1000, 1e-8), 27,06 (100000, 1e-12); **keine Konvergenzwarnung** (aufgefangen wurde nur eine `FutureWarning` zum veralteten Argument `penalty`).
Gilt für diese Version, diese Einstellungen und diese Punkte; die App rechnet scikit-learn nicht.

### Rechenregeln und Hinweise zu den Zahlen

- **Zählungen** („151 von 300“) kommen aus einem Zufallsstrom und sind ein Band, nicht eine Stelle genau: Neu-Rechnung in den Tests mit Toleranz (Zählungen ± 2 je 40 Stichproben, Floats relativ 1e-4 bis 1e-9), exakt nur, was in Brüchen oder geschlossener Form gerechnet ist.
- **Gleiche Zufalls-Seeds, gleiche Stichproben** auf jeder Plattform (SplitMix64 mit plattformfester Normalzahl); die Rechnung der Lösungen (HiGHS, BLAS, `numpy.exp`) kann an der letzten Binärstelle abweichen. Die Beschriftungen sind darum „Zahl der Stichproben“, nie „immer“.
- **Mehrverlust und Grenze** beziehen sich auf die wahre Wahrscheinlichkeit des erfundenen Stroms mit den zwölf nicht modellierten Kunden; in einer anderen Anordnung (andere Schwelle als 7 Tage, andere Spalten) ändern sich alle Häufigkeiten.
- **„Zahl → Test“:** jede Zahl dieser README steht in `tests/test_claims.py` (Zahl in der README, Band, Neu-Rechnung oder Konsistenz gegen `data/lsp_frozen.json`) oder in einer der Testdateien der Tabelle; Zusatzmessungen ohne Test sind als „Messung beim Bau, nicht im Test“ gekennzeichnet
  (Rechenzeiten, scikit-learn-Werte außerhalb von `importorskip`).

## Befunde und Korrekturen gegenüber früheren Messungen

Eine frühere Messung (Ausgabe `out_rn_v16_logit_separation.txt`, anderer Generator-Nachbau und anderer Zufallsstrom) nannte einige Zahlen, die sich hier verschoben haben; die neue Messung gilt.

- **Häufigkeit der Trennung bei n = 100:** früher 117 von 300 (39,0 %) und 127 von 300 (42,3 %), hier 151 von 300 (50,3 %) und 136 von 300 (45,3 %). Ursache: andere Zufallsströme und andere zwölf Kundeneffekte (hier aus SplitMix64 gezogen, dort aus `numpy`). Kontrolle beim Bau (nicht im Repo): der frühere Generator
  mit denselben Stichproben-Seeds gab wieder 117 von 300; die Merkmale dieser Demo mit den ursprünglichen zwölf Kundeneffekten gaben 136 von 300. Die Spanne „etwa 39 bis 50 %“ ist also eine Eigenschaft der Streuung zwischen Zufallsströmen und Kundeneffekten, nicht eine Stelle genau.
- **n = 500, Standardstreuung:** früher 0 von 200, hier 5 von 200 (2,5 %). Die Aussage „ab n = 500 trennt keine Stichprobe“ gilt hier nicht; erst bei n = 1000 und 3000 zeigt die Zählung 0 von 200 und 0 von 100 (Band bis 1,9 %).
- **Grenze der Logit-Form:** früher 0,0498, hier 0,024. Anderer Generator-Nachbau (andere Kundeneffekte) und andere Definition (kleinster erwarteter Verlust des Logit gegen die wahre Wahrscheinlichkeit auf Merkmalssätzen, statt ML auf einer großen Stichprobe).
- **Ridge gegen Firth:** früher in 14 und 18 von 150 besser als Firth bei n = 100, hier in 6 und 15 (λ = 0,1 und 1); bei n = 60 früher 15 und 18, hier 18 und 19. Firth war in beiden Messungen im Median am besten, ML in keiner der 150 getrennten Stichproben besser.
- **Faustregel:** quantitativ nahe der früheren Messung (Wald-Ablehnung früher 13,7 %, hier 14,0 %; Zollhalt früher 1,74-fach, hier 1,85-fach; Streuung ÷ Standardfehler früher 1,35, hier 1,38).
- **Gerade außerhalb [0, 1]:** früher 15,0 % bei n = 3000, hier 15,5 %.
- **Kappung:** eine frühere Messung meldete `nan` ab Schritt 15 für die naive Formel mit p(1 − p), nicht gerechnet hier; mit stabilen Formeln bleibt der Pfad bis Schritt 200 endlich (Test), die Kappung ist eine Rechengrenze der Software, keine Eigenschaft der Daten.
- **Schnitt:** Gerade und Logit stehen in Experiment 1 nebeneinander in einer Abbildung statt umschaltbar; die Brücke zum stetigen Ausgang (Gerade auf Standgeld) ist vorgerechnet.

## Quellen (so zitiert, wie sie gelesen sind)

- Hastie, Tibshirani, Friedman: *The Elements of Statistical Learning*, Abschn. 4.4.1 (IRLS als Newton-Schritt, gewichtete kleinste Quadrate auf der Arbeitsantwort; Konvergenz nicht garantiert, Schrittweitenhalbierung hilft). Wortlaut gelesen.
- Kosmidis und Firth: Jeffreys-prior penalty, finiteness and shrinkage in binomial-response GLMs (arXiv 1812.01938, Fassung 3), Abschn. 1: Begriffe vollständig, quasi-vollständig, überlappend; Wiedergabe von Albert und Anderson 1984 (Original nicht gelesen); Jeffreys-Strafe nach Firth 1993 (nur über diese Quelle),
  Endlichkeit der gestraften Schätzer. Wortlaut gelesen.
- Sur und Candès: A modern maximum-likelihood theory for high-dimensional logistic regression (arXiv 1803.06964, PNAS 2019): die Faustregel n ≥ 5p reicht für die klassische Standardfehlerformel nicht. Candès und Sur: The phase transition for the existence of the MLE in high-dimensional logistic regression (arXiv 1804.09753, 2020):
  Schwelle p/n = 1/2 für Merkmale ohne Zusammenhang mit dem Ausgang; die Cover-Formel (Cover 1965, nicht gelesen) ist hier für kleine Fälle durch Aufzählung nachgerechnet, nicht allgemein bewiesen. Wortlaut gelesen.
- ISLR2 (James, Witten, Hastie, Tibshirani): Konvention p ohne Achsenabschnitt.

Keine Neuheitsaussage: nichts hier ist wissenschaftlich neu.

## Ehrliche Grenzen

- **Ein Vehikel, eine Schwelle, ein Modell:** Container-Strom mit erfundenen Parametern, Standgeld ab 7 Tagen, sieben Spalten ohne Kunden. Andere Schwellen oder Modelle ändern die Häufigkeiten; die Zahlen gelten für diesen Generator und diese Zufalls-Seeds.
- **Die „gängige Abbruchregel“ ist ein Nachbau aus dem Gedächtnis** (Kappung |η| ≤ 30, Abbruch nach |ΔD| / (|D| + 0,1) < 1e-8, höchstens 25 Schritte); gegen eine bestimmte Software nicht geprüft. Die scikit-learn-Werte gelten für Version 1.9.1 und Windows.
- **Zuwachsregel:** „Zuwachs 1/kleinstes |x|“ gilt für die symmetrischen sechs Punkte und beim Skalieren; allgemein erhöht jeder Newton-Schritt den Abstand der Stützpunkte um 1, wenn es so viele Stützpunkte wie Parameter gibt (eigene Herleitung, am asymmetrischen Beispiel nachgerechnet);
  bei mehr Stützpunkten als Parametern nicht gemessen.
- **Quasi-Trennung bei Gleichständen:** der IRLS kann bei getrennten Daten „konvergiert“ melden (frühere Messung: bei 50 von 1565 getrennten Datensätzen mit einem Merkmal fiel der Schritt unter 1e-9); hier nicht wiederholt.
- **Firth** nur als Newton mit Schrittweitenhalbierung; keine profilierten Intervalle, keine Cauchy-Priors, keine Posterior-Mittel: „Firth gegen Prior“ bleibt offen. **Ridge** nur mit zwei festen Stärken, keine Wahl der Stärke. **Streichen** nur als „Spalte mit dem größten |β| beim Abbruch“.
- **Faustregel:** Container-Merkmale mit Gauß-Rauschmerkmalen, nicht die Anordnung der Quelle; die Zahlen sind schwächer als dort.
- **Cover-Formel:** für kleine Fälle durch Aufzählung gleich der Formel, nicht allgemein bewiesen; der Anteil gilt für Merkmale ohne Zusammenhang mit dem Ausgang, im Container-Strom kommt die Trennung aus Zellen mit kleiner Standgeld-Rate.
- **Zeitgrenze:** die Messungen zu Häufigkeit, Hilfe und Faustregel sind vorgerechnet; die Regler „Stichproben“ und „Rauschmerkmale“ wählen nur unter den vorgerechneten Werten.

## Verifikation

- **Handrechnung:** `tests/test_logit.py` rechnet die Handbeispiele in Brüchen und geschlossener Form (erste Schritte, Grenzwerte, Information, Wald und Likelihood-Quotient, Stützpunkte, Firth-Zählung, Cover-Aufzählung).
- **Orakel** (`tests/test_oracle_logit_separation.py`, anderer Rechenweg): IRLS gegen `scipy.optimize.minimize` mit analytischem Gradienten und gegen `scikit-learn` (`C=np.inf`, Newton-Cholesky, `importorskip`); Rezessions-LP gegen die geschlossene Form für ein Merkmal (300 Datensätze) und die Lücke der Optimalwerte;
  Firth-Score gegen den Differenzenquotienten der gestraften Log-Likelihood, Firth gegen BFGS mit Gradient und gegen „ein halbes in jede Zelle“ (200 Zählungen); Ridge gegen `scikit-learn` mit C = 1/λ; Wald-Standardfehler gegen die Hesse-Matrix per Differenzenquotient.
- **Eigenschaften** (`tests/test_eigenschaften_logit_separation.py`, je 20 Instanzen): Vertauschen, Spiegeln des Ausgangs, Verschieben und Skalieren eines Merkmals, Score null, Verdoppeln der Zeilen, Ridge-Grenzwerte, Deviance monoton ab Schritt 2, Firth nicht größer als ML (gemessene Eigenschaft dieser Instanzen).
- **Strom und Zufall** (`tests/test_scenario.py`): die vektorisierte Folge gleich der Klasse `SplitMix64`, Merkmale gleich der skalaren Ziehung, Verteilung der Merkmale, wahre Wahrscheinlichkeit gleich der Häufigkeit, die schnellen Normalzahlen gegen `numpy` und Kolmogorov-Smirnov.
- **Randfälle:** `tests/test_edge_cases.py` (konstanter Ausgang, Merkmal ohne Variation, n < p, n = 2, genau ein Ereignis, doppelte Zeilen, |η| über 36,7, Regler an den Grenzen).
- **Claims:** `tests/test_claims.py` prüft jede Zahl dieser README (Zahl steht in der README, Neu-Rechnung als Band oder Konsistenz gegen `data/lsp_frozen.json`); `tests/test_verdicts.py` die Verdikte an gesetzten Werten, `tests/test_presets.py` je Preset die erwartete Verdikt-Zeile.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an den Grenzen, jede Auswahl jedes Auswahlreglers, Permalink), Layout-Test (Schnellstart in der Seitenleiste, Hervorhebung, jede Änderung eines Reglers hebt sie auf), Footer-Test, Laufzeit-Importe.
- **Linux:** `demo_linux_check.py` (Docker) mit Ergebnis im Abschlussbericht.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2) |
| `data/lsp_frozen.json` | die vorgerechneten Messungen (Häufigkeit, Hilfe, Faustregel, Brücke) |
| `lsp_constants.py` | Vehikel, Handbeispiele, Regler-Grenzen, Farben, Presets |
| `lsp_evaluation.py` | die Experimente als reine Rechnungen, Verdikte als reine Funktionen, Anzeigeformate |
| `lsp_layout.py` | Fester Kopf (Stylesheet) |
| `lsp_logit.py` | Kern: IRLS, Newton, Wald, Likelihood-Quotient, Abbruchregeln, Firth, Ridge, Streichen, Bruch-Schritt |
| `lsp_messungen.py` | die vorgerechneten Messungen als Funktionen mit festen Zufalls-Seeds |
| `lsp_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `lsp_rng.py` | SplitMix64 mit plattformfester Normalzahl (Kopie der Vorlage plus `u64_array`) |
| `lsp_rngvec.py` | schnelle plattformfeste Normalzahlen für große Felder (Rauschmerkmale) |
| `lsp_scenario.py` | Containerstrom, Testmenge, Mehrverlust, Grenze der Logit-Form, Handbeispiele |
| `lsp_separation.py` | Rezessions-LP, vollständig gegen quasi, Cover-Formel, Nullzellen |
| `lsp_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scikit-learn (nur Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tools/lsp_make_frozen.py` | erzeugt `data/lsp_frozen.json`; `--check` rechnet Teile neu und vergleicht als Band |
| `tests/` | Handrechnung, Orakel, Eigenschaften, Strom, Randfälle, Verdikte, Claims, Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **PDF-Export** der Ergebnisse.
- **Cauchy-Priors, Posterior-Mittel und profilierte Intervalle** als Alternativen zu Firth.
- **Wahl der Ridge-Stärke** (Kreuzvalidierung, GCV, L-Kurve): dafür [tikhonov-demo](https://github.com/sebastian-hanisch/tikhonov-demo) und [parameterwahl-demo](https://github.com/sebastian-hanisch/parameterwahl-demo).
- **Existenz des Schätzers gegen den Spielplan** (Gewinngraph, Newton gegen Zermelo-Iteration) im Bradley-Terry-Modell: [bradley-terry-demo](https://github.com/sebastian-hanisch/bradley-terry-demo); ein Paarvergleichsmodell hat eine Designmatrix aus Differenzen von Indikatoren, die Trennung ist dort verwandt
  (nach der Modellform; nicht nachgerechnet).
- **Trennung bei der partiellen Likelihood** des Cox-Modells: [cox-regression-demo](https://github.com/sebastian-hanisch/cox-regression-demo).
- **Robuste Standardfehler, nicht kanonische Links, Deviance als Gütetest, Kalibrierung und Entscheidungsschwelle, Mehrklassen-Logit** sind nicht Gegenstand dieser Demo.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die vorgerechneten Messungen neu erzeugen: `python tools/lsp_make_frozen.py` (einige Minuten); prüfen: `python tools/lsp_make_frozen.py --check`.

Gebaut mit Streamlit, Plotly, NumPy, SciPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
