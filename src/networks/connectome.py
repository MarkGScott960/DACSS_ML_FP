"""FlyWire FAFB v783 mushroom body + olfactory projection neurons, as a reservoir graph.

Loader contract (Planning/Final Project Guide.md): signed weight matrix, mask of
modulatory neurons, neuron metadata. Imported by notebooks 07 and 08.

Node set = mushroom body (Kenyon cells, MBONs, PAM/PPL dopaminergic neurons, APL)
         + olfactory projection neurons (class "ALPN") that synapse onto Kenyon cells.
The PN rule was found empirically (notebook 07, section 1): of the 539 neurons outside
the mushroom body that feed Kenyon cells, 285 are ALPNs, with a median of 62 Kenyon
cells each. Those are the input layer.

Weights are defined per presynaptic neuron: sign = +1 (ACh), -1 (GABA, glutamate),
0 (serotonin, dopamine, octopamine: modulators carry no fast signal, so their outgoing
edges are removed from W but the neurons are kept, as the Tier 3 reward channel).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.paths import ROOT, PROCESSED

FAFB = ROOT / "Flywire_FAFBv783"
MB_PATTERN = r"^KC|^MBON|^PAM|^PPL1|^PPL2|^APL"
SIGN = {"ACH": 1, "GABA": -1, "GLUT": -1, "SER": 0, "DA": 0, "OCT": 0}
GROUPS = ["KC", "MBON", "PAM", "PPL", "APL", "PN"]


@dataclass
class Connectome:
    nodes: pd.DataFrame     # one row per node: root_id, primary_type, side, nt_type, group
    edges: pd.DataFrame     # pre, post (node indices), syn (summed synapse count)

    @property
    def n(self) -> int:
        return len(self.nodes)

    def idx(self, group: str) -> np.ndarray:
        return np.flatnonzero(self.nodes["group"].to_numpy() == group)


def _group(primary_type: pd.Series, cls: pd.Series) -> pd.Series:
    t = primary_type.astype(str)
    g = pd.Series("", index=primary_type.index)
    for name, pat in [("KC", r"^KC"), ("MBON", r"^MBON"), ("PAM", r"^PAM"),
                      ("PPL", r"^PPL"), ("APL", r"^APL")]:
        g[t.str.match(pat)] = name
    g[(g == "") & (cls == "ALPN")] = "PN"
    return g


def build_connectome(glutamate_sign: int = -1, cache: bool = True) -> Connectome:
    """Read the FlyWire files (or the cache) and return the MB + PN subgraph.

    The cache holds the sign-independent graph; `glutamate_sign` is applied in `signs()`.
    """
    nodes_p, edges_p = PROCESSED / "connectome_nodes.parquet", PROCESSED / "connectome_edges.parquet"
    if cache and nodes_p.exists() and edges_p.exists():
        return Connectome(pd.read_parquet(nodes_p), pd.read_parquet(edges_p))

    con = pd.read_csv(FAFB / "connections_princeton.csv.gz",
                      usecols=["pre_root_id", "post_root_id", "syn_count", "nt_type"])
    cls = pd.read_csv(FAFB / "classification.csv.gz")
    cts = pd.read_csv(FAFB / "consolidated_cell_types.csv.gz")
    meta = cls.merge(cts, on="root_id", how="left")

    # one row per pair per neuropil in the file: aggregate BEFORE anything else
    edges = con.groupby(["pre_root_id", "post_root_id"], as_index=False)["syn_count"].sum()
    nt = (con[["pre_root_id", "nt_type"]].drop_duplicates("pre_root_id")
          .rename(columns={"pre_root_id": "root_id"}))
    meta = meta.merge(nt, on="root_id", how="left")
    meta["group"] = _group(meta["primary_type"], meta["class"])

    # PN rule: ALPN with at least one edge onto a Kenyon cell
    kc = set(meta.loc[meta["group"] == "KC", "root_id"])
    feeds_kc = set(edges.loc[edges["post_root_id"].isin(kc), "pre_root_id"])
    pn_ok = (meta["group"] == "PN") & meta["root_id"].isin(feeds_kc)
    meta.loc[(meta["group"] == "PN") & ~pn_ok, "group"] = ""

    nodes = (meta[meta["group"] != ""][["root_id", "primary_type", "side", "nt_type", "group"]]
             .sort_values("root_id").reset_index(drop=True))
    index = pd.Series(np.arange(len(nodes)), index=nodes["root_id"])
    e = edges[edges["pre_root_id"].isin(index.index) & edges["post_root_id"].isin(index.index)]
    e = pd.DataFrame({"pre": index[e["pre_root_id"]].to_numpy(),
                      "post": index[e["post_root_id"]].to_numpy(),
                      "syn": e["syn_count"].to_numpy(float)})
    if cache:
        nodes.to_parquet(nodes_p); e.to_parquet(edges_p)
    return Connectome(nodes, e)


def signs(c: Connectome, glutamate_sign: int = -1) -> np.ndarray:
    """Per-node sign: +1 / -1 / 0 (modulators and unknown transmitter)."""
    m = dict(SIGN); m["GLUT"] = glutamate_sign
    return c.nodes["nt_type"].map(m).fillna(0).to_numpy(float)


def modulatory_mask(c: Connectome) -> np.ndarray:
    return c.nodes["nt_type"].isin(["SER", "DA", "OCT"]).to_numpy()


def weight_matrix(c: Connectome, scheme: str = "fraction", glutamate_sign: int = -1,
                  drop: np.ndarray | None = None) -> sp.csr_matrix:
    """Signed matrix A, A[i, j] = weight of the connection from neuron i onto neuron j.

    scheme: "fraction" (each neuron's inputs sum to 1; default), "log1p", "binary", "raw".
    `drop` = node indices to remove entirely (used for the APL ablation).
    """
    e = c.edges
    if drop is not None and len(drop):
        keep = ~(np.isin(e["pre"], drop) | np.isin(e["post"], drop))
        e = e[keep]
    syn = e["syn"].to_numpy(float)
    if scheme == "fraction":
        tot = np.bincount(e["post"], weights=syn, minlength=c.n)
        mag = syn / tot[e["post"].to_numpy()]
    elif scheme == "log1p":
        mag = np.log1p(syn)
    elif scheme == "binary":
        mag = np.ones_like(syn)
    elif scheme == "raw":
        mag = syn
    else:
        raise ValueError(scheme)
    s = signs(c, glutamate_sign)[e["pre"].to_numpy()]
    keep = s != 0                               # modulators send nothing through W
    return sp.csr_matrix((mag[keep] * s[keep], (e["pre"].to_numpy()[keep], e["post"].to_numpy()[keep])),
                         shape=(c.n, c.n))


def spectral_radius(A: sp.spmatrix) -> float:
    """Largest |eigenvalue| (ARPACK; dense fallback if it fails to converge)."""
    from scipy.sparse.linalg import eigs, ArpackNoConvergence
    A = A.astype(float)
    try:
        return float(abs(eigs(A, k=1, which="LM", return_eigenvectors=False, tol=1e-5, maxiter=20000)[0]))
    except ArpackNoConvergence:
        return float(np.abs(np.linalg.eigvals(A.toarray())).max())


def pn_evidence() -> pd.DataFrame:
    """Which neurons outside the mushroom body synapse onto Kenyon cells? Summarised by
    super_class / class. Used in notebook 07 to justify the input-layer rule (class ALPN)."""
    con = pd.read_csv(FAFB / "connections_princeton.csv.gz", usecols=["pre_root_id", "post_root_id", "syn_count"])
    cls = pd.read_csv(FAFB / "classification.csv.gz")
    cts = pd.read_csv(FAFB / "consolidated_cell_types.csv.gz")
    meta = cls.merge(cts, on="root_id", how="left")
    edges = con.groupby(["pre_root_id", "post_root_id"], as_index=False)["syn_count"].sum()
    t = meta["primary_type"].astype(str)
    mb = set(meta.loc[t.str.contains(MB_PATTERN, regex=True), "root_id"])
    kc = set(meta.loc[t.str.match(r"^KC"), "root_id"])
    up = edges[edges["post_root_id"].isin(kc) & ~edges["pre_root_id"].isin(mb)]
    g = (up.groupby("pre_root_id").agg(kcs=("post_root_id", "nunique"), syn=("syn_count", "sum"))
         .reset_index().merge(meta.rename(columns={"root_id": "pre_root_id"})[["pre_root_id", "super_class", "class"]],
                              on="pre_root_id", how="left"))
    g[["super_class", "class"]] = g[["super_class", "class"]].fillna("(unlabelled)")
    return (g.groupby(["super_class", "class"]).agg(neurons=("pre_root_id", "size"),
                                                    median_KCs_contacted=("kcs", "median"),
                                                    synapses=("syn", "sum"))
            .sort_values("neurons", ascending=False).reset_index())
