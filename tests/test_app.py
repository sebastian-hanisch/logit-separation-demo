"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_jede_auswahl_jedes_reglers_laeuft_ohne_fehler():
    """Radio, Auswahlfelder und Auswahl-Regler der Experimente: jede Option einmal in derselben App-Sitzung (Zwischenergebnisse sind gepuffert)."""
    import lsp_presets as P

    at = _fresh_app()
    for key, spec in P.SETTING_SPECS.items():
        if spec.options is None:
            continue
        for art in ("radio", "select_slider", "selectbox"):
            try:
                widget = getattr(at.main, art)(key=key)
                break
            except KeyError:
                continue
        else:
            raise AssertionError(key)
        for option in spec.options:
            widget.set_value(option).run(timeout=TIMEOUT)
            assert not at.exception, (key, option, [str(e) for e in at.exception])
            widget = getattr(at.main, art)(key=key)


def test_zufalls_seed_an_den_grenzen_und_neue_stichprobe():
    at = _fresh_app()
    for wert in (0, 9999):
        at.sidebar.number_input(key="seed_input").set_value(wert).run(timeout=TIMEOUT)
        assert not at.exception, [str(e) for e in at.exception]
    labels = [b.label for b in at.sidebar.button]
    at.sidebar.button[labels.index("🎲 Neue Stichprobe ziehen")].click().run(timeout=TIMEOUT)
    assert not at.exception and 0 <= at.sidebar.number_input(key="seed_input").value <= 9999


def test_permalink_uebernimmt_werte_und_verwirft_unerlaubte():
    at = AppTest.from_file(APP_PATH)
    at.query_params["n3"] = "1000"
    at.query_params["t3"] = "unbekannt"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.main.select_slider(key="e3_n").value == 1000
    assert at.main.radio(key="e3_streuung").value == "Standard (× 1)"


def test_ergebnis_erscheint_ohne_python_fehler_in_jedem_preset_und_hat_verdikte():
    import lsp_constants as C

    for name in C.PRESETS:
        at = _fresh_app()
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert len(at.success) + len(at.info) + len(at.warning) >= 6, name
