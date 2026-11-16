"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Die Regler der Experimente stehen im Hauptbereich beim Experiment, der Zufalls-Seed der frisch gezogenen Stichproben in der Seitenleiste; alle stehen in SETTING_SPECS, damit
das aktive Preset (zustandslos: jeder Regler gleich dem Wert des Presets oder dem Standard) beim Ändern eines beliebigen Reglers nicht mehr hervorgehoben ist."""

import random
from dataclasses import dataclass

import streamlit as st

import lsp_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None


SETTING_SPECS = {
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "e1_beispiel": SettingSpec("b1", str, C.E1_BEISPIELE[0], options=C.E1_BEISPIELE),
    "e1_steps": SettingSpec("s1", int, C.E1_STEPS_DEFAULT, C.E1_STEPS_MIN, C.E1_STEPS_MAX),
    "e2_daten": SettingSpec("d2", str, C.E2_DATEN[0], options=C.E2_DATEN),
    "e2_steps": SettingSpec("s2", int, C.E2_STEPS_DEFAULT, C.E2_STEPS_MIN, C.E2_STEPS_MAX),
    "e2_regel": SettingSpec("r2", str, C.E2_REGELN[0], options=C.E2_REGELN),
    "e2_skala": SettingSpec("k2", float, 1.0, options=C.E2_SKALEN),
    "e3_n": SettingSpec("n3", int, 200, options=C.E3_N),
    "e3_streuung": SettingSpec("t3", str, C.E3_STREUUNG[0], options=C.E3_STREUUNG),
    "e4_n": SettingSpec("n4", int, 100, options=C.E4_N),
    "e4_verfahren": SettingSpec("v4", str, C.E4_VERFAHREN[2], options=C.E4_VERFAHREN),
    "e4_k": SettingSpec("k4", int, C.E4_K_DEFAULT, C.E4_K_MIN, C.E4_K_MAX),
    "e5_pr": SettingSpec("p5", int, 50, options=C.E5_PR),
}
PRESET_KEYS = {k: k for k in SETTING_SPECS}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _begrenzen(spec: SettingSpec, value):
    """Wert auf den erlaubten Bereich begrenzen; None, wenn er nicht erlaubt ist (Auswahl) oder nicht lesbar."""
    try:
        value = spec.caster(value)
    except (ValueError, TypeError):
        return None
    if spec.options is not None:
        return value if value in spec.options else None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            value = _begrenzen(spec, qp[spec.url_param])
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Alle Regler setzen: der Wert des Presets, sonst der Standard (damit das gewählte Preset hervorgehoben ist)."""
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = C.PRESETS[name].get(state_key, spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State."""
    for name, preset in C.PRESETS.items():
        if all(st.session_state.get(state_key, spec.default) == preset.get(state_key, spec.default) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
