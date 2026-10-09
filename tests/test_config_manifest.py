"""Configuration rules and manifest hashing."""

import copy
import json

import pytest

from flypredictor import config as fconfig
from flypredictor import manifest


def test_clean_set_has_no_edi_components(base_cfg):
    for name, g in base_cfg["features"]["groups"].items():
        if g["set"] == "clean":
            assert not g.get("in_edi"), name
    clean = set(fconfig.feature_vars(base_cfg, "clean"))
    assert "v2x_polyarchy" not in clean and "v2x_regime" not in clean


def test_full_contains_clean(base_cfg):
    clean = set(fconfig.feature_vars(base_cfg, "clean"))
    full = set(fconfig.feature_vars(base_cfg, "full"))
    assert clean < full
    assert "v2x_polyarchy" in full


def test_party_vars_only_in_party_arm(base_cfg):
    party_only = set(fconfig.feature_vars(base_cfg, "party")) - set(fconfig.feature_vars(base_cfg, "clean"))
    assert party_only and not party_only & set(fconfig.feature_vars(base_cfg, "full"))


@pytest.mark.parametrize(
    "path,value",
    [
        (("labels", "stage1_event"), "wrong"),
        (("labels", "state_source"), "wrong"),
        (("labels", "horizons_stage1"), [0, 1]),
    ],
)
def test_validation_rejects_bad_values(base_cfg, path, value):
    c = copy.deepcopy(base_cfg)
    c[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        fconfig.validate(c)


def test_edi_component_in_clean_is_rejected(base_cfg):
    c = copy.deepcopy(base_cfg)
    c["features"]["groups"]["media"]["set"] = "clean"
    with pytest.raises(ValueError):
        fconfig.validate(c)


def test_sha256_known_value(tmp_path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"abc")
    assert manifest.sha256(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_manifest_detects_change(tmp_path):
    root = tmp_path
    f = root / "data" / "raw" / "a.csv"
    f.parent.mkdir(parents=True)
    f.write_text("1,2,3\n")
    entry = manifest.file_entry(f, root, url="u")
    manifest.write_manifest("t", [entry], root / "data" / "manifests")
    assert manifest.verify_manifest("t", root / "data" / "manifests", root) == []
    f.write_text("1,2,4\n")
    assert manifest.verify_manifest("t", root / "data" / "manifests", root) == ["hash changed: data/raw/a.csv"]
    saved = json.loads((root / "data" / "manifests" / "t.json").read_text())
    assert saved["files"][0]["path"] == "data/raw/a.csv"
