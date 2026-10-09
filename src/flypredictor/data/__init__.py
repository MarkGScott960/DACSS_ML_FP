"""Data layer: raw ERT/V-Dem -> democratic episodes -> country-year states -> labels -> panel.

Order of use (see notebooks/0x_data/02_panel_labels.ipynb):

    ert   = episodes.load_ert(paths)
    eps   = episodes.democratic_episodes(ert, cfg)
    rt    = ert_jobs.load_realtime(paths)              # needs R; skip if state_source = rule
    ev    = episodes.country_events(eps, rt, ert)
    st    = states.assign_states(ert, eps, cfg, rt)
    lab   = labels.build_labels(st, ev, cfg)
    panel = panel.build_panel(paths, cfg, st)
"""
