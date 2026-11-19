import datetime
import io
import json
import re
import zipfile
import networkx as nx
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import py3Dmol
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    Lipinski,
    rdFingerprintGenerator,
    rdMolDescriptors,
)

st.set_page_config(
    page_title="Drug Repurposing",
    page_icon="🧪",
    layout="wide",
)

RANDOM_SEED = 42
TARGET_SYNONYMS = {
    "COX1": "PTGS1",
    "COX-1": "PTGS1",
    "PGHS1": "PTGS1",
    "COX2": "PTGS2",
    "COX-2": "PTGS2",
    "PGHS2": "PTGS2",
}

DEMO_DRUGS = pd.DataFrame(
    [
        ["Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O", "PTGS1;PTGS2", "Approved", "Built-in demo"],
        ["Ibuprofen", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "PTGS1;PTGS2", "Approved", "Built-in demo"],
        ["Naproxen", "CC(C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O", "PTGS1;PTGS2", "Approved", "Built-in demo"],
        ["Diclofenac", "C1=CC=C(C(=C1)CC(=O)O)NC2=C(C=CC=C2Cl)Cl", "PTGS1;PTGS2", "Approved", "Built-in demo"],
        ["Celecoxib", "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F", "PTGS2", "Approved", "Built-in demo"],
        ["Indomethacin", "CC1=C(C2=C(N1C(=O)C3=CC=C(C=C3)Cl)C=CC(=C2)OC)CC(=O)O", "PTGS1;PTGS2", "Approved", "Built-in demo"],
    ],
    columns=["drug_name", "smiles", "targets", "approval_status", "source"],
)

DEMO_DISEASE = pd.DataFrame(
    [
        ["Inflammation (demo)", "PTGS1", "Arachidonic acid / prostaglandin synthesis", 0.90, ""],
        ["Inflammation (demo)", "PTGS2", "Arachidonic acid / prostaglandin synthesis", 0.95, ""],
    ],
    columns=["disease", "gene", "pathway", "association_score", "pdb_id"],
)


def _txt(x) -> str:
  return "" if pd.isna(x) else str(x).strip()


def _norm_gene(symbol: str) -> str:
  s = _txt(symbol).upper()
  return TARGET_SYNONYMS.get(s, s)


def _parse_targets(x) -> set[str]:
  return {_norm_gene(v) for v in re.split(r"[;,|]", _txt(x)) if _txt(v)}


def parse_any_file_to_df(filename: str, content: bytes) -> pd.DataFrame:
  lower = filename.lower()
  buf = io.BytesIO(content)

  if lower.endswith(".csv"):
    return pd.read_csv(buf)
  if lower.endswith((".tsv", ".tab")):
    return pd.read_csv(buf, sep="\t")
  if lower.endswith((".xlsx", ".xls", ".ods")):
    return pd.read_excel(buf)
  if lower.endswith(".parquet"):
    return pd.read_parquet(buf)
  if lower.endswith(".json"):
    text = content.decode("utf-8", errors="ignore").strip()
    data = json.loads(text)
    if isinstance(data, list):
      return pd.DataFrame(data)
    if isinstance(data, dict):
      for k in ("records", "data", "drugs", "diseases", "items"):
        if k in data and isinstance(data[k], list):
          return pd.DataFrame(data[k])
      return pd.DataFrame(data)
    return pd.DataFrame([{"value": str(data)}])

  text = content.decode("utf-8", errors="ignore").strip()
  lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
  if lines:
    for sep in ("\t", ",", ";", "|"):
      if sep in lines[0]:
        try:
          df_try = pd.read_csv(io.StringIO(text), sep=sep)
          if df_try.shape[1] >= 2:
            return df_try
        except Exception:
          pass
  return pd.DataFrame({"raw_entry": lines})


def split_uploaded_tables(uploaded_files):
  parsed = []
  for f in uploaded_files:
    df = parse_any_file_to_df(f.name, f.getvalue())
    df.columns = [str(c).strip() for c in df.columns]
    parsed.append((f.name, df))

  drugs_raw, disease_raw = None, None
  for fname, df in parsed:
    cols = {c.lower() for c in df.columns}
    if {"drug_name", "smiles", "targets"} & cols or "drug" in fname.lower():
      if drugs_raw is None:
        drugs_raw = df.copy()
    if {"disease", "gene"} & cols or "disease" in fname.lower():
      if disease_raw is None:
        disease_raw = df.copy()

  if len(parsed) == 1:
    _, df = parsed[0]
    if "raw_entry" in df.columns and drugs_raw is None:
      drugs_raw = pd.DataFrame({
          "drug_name": df["raw_entry"],
          "smiles": "",
          "targets": "",
          "approval_status": "",
          "source": "Uploaded file",
      })
    elif drugs_raw is None:
      drugs_raw = df.copy()
    if disease_raw is None:
      disease_raw = df.copy()

  return drugs_raw, disease_raw


def ensure_schemas(drugs_raw: pd.DataFrame, disease_raw: pd.DataFrame):
  d_df = drugs_raw.copy()
  dt_df = disease_raw.copy()

  if "drug_name" not in d_df.columns and len(d_df.columns) > 0:
    d_df["drug_name"] = d_df[d_df.columns[0]]
  if "smiles" not in d_df.columns:
    d_df["smiles"] = ""
  if "targets" not in d_df.columns:
    d_df["targets"] = ""
  for col in ["approval_status", "source", "pdb_id"]:
    if col not in d_df.columns:
      d_df[col] = ""

  if "disease" not in dt_df.columns or "gene" not in dt_df.columns:
    inferred_genes = set()
    for t_val in d_df["targets"].dropna():
      inferred_genes.update(_parse_targets(t_val))
    if not inferred_genes:
      inferred_genes = {"PTGS1", "PTGS2"}

    dt_df = pd.DataFrame([
        {
            "disease": "General Target Set (auto)",
            "gene": g,
            "pathway": "Inferred from drug targets",
            "association_score": 1.0,
            "pdb_id": "",
        }
        for g in sorted(inferred_genes)
    ])
  else:
    for col in ["pathway", "pdb_id"]:
      if col not in dt_df.columns:
        dt_df[col] = ""
    if "association_score" not in dt_df.columns:
      dt_df["association_score"] = 1.0

  return d_df, dt_df


@st.cache_data(show_spinner=False)
def fetch_smiles_pubchem(name: str) -> str:
  url = (
      "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
      + requests.utils.quote(name)
      + "/property/CanonicalSMILES,IsomericSMILES/JSON"
  )
  try:
    r = requests.get(url, timeout=20)
    if r.status_code == 200:
      props = r.json()["PropertyTable"]["Properties"][0]
      return props.get("CanonicalSMILES") or props.get("ConnectivitySMILES") or ""
  except Exception:
    pass
  return ""


def build_physicochemical_table(scored_df: pd.DataFrame) -> pd.DataFrame:
  rows = []
  for r in scored_df.itertuples():
    m = getattr(r, "mol", None)
    if m is None:
      continue
    mw = float(Descriptors.MolWt(m))
    clogp = float(Crippen.MolLogP(m))
    hbd = int(Lipinski.NumHDonors(m))
    hba = int(Lipinski.NumHAcceptors(m))
    tpsa = float(rdMolDescriptors.CalcTPSA(m))
    rotb = int(Lipinski.NumRotatableBonds(m))
    lipinski_ok = (mw <= 500) and (clogp <= 5) and (hbd <= 5) and (hba <= 10)
    rows.append({
        "drug_name": r.drug_name,
        "mw": round(mw, 2),
        "clogp": round(clogp, 2),
        "hbd": hbd,
        "hba": hba,
        "tpsa": round(tpsa, 2),
        "rot_bonds": rotb,
        "lipinski_pass": lipinski_ok,
    })
  return pd.DataFrame(rows)


def make_charts(
    ranked_df: pd.DataFrame,
    reference_drug: str,
    disease_choice: str,
    target_weight: float,
    chem_weight: float,
):
  fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
  cov = ranked_df["target_coverage"].fillna(0.0).to_numpy()
  sim = ranked_df["chemical_similarity"].fillna(0.0).to_numpy()
  c_part = target_weight * cov
  s_part = chem_weight * sim
  y_pos = np.arange(len(ranked_df))

  axes[0].barh(
      y_pos, c_part, color="#2a9d8f", label=f"{target_weight:.2f} × Target Coverage"
  )
  axes[0].barh(
      y_pos,
      s_part,
      left=c_part,
      color="#457b9d",
      label=f"{chem_weight:.2f} × Similarity ({reference_drug})",
  )
  axes[0].set_yticks(y_pos)
  axes[0].set_yticklabels(ranked_df["drug_name"])
  axes[0].invert_yaxis()
  axes[0].set_xlabel("Composite Screening Score")
  axes[0].set_title(f"Score Decomposition — {disease_choice}")
  axes[0].set_xlim(0, 1.05)
  axes[0].legend(loc="lower right", fontsize=8)
  axes[0].grid(axis="x", linestyle="--", alpha=0.4)

  sub = ranked_df[ranked_df["chemical_similarity"].notna()]
  axes[1].scatter(
      sub["chemical_similarity"],
      sub["target_coverage"],
      s=110,
      color="#e76f51",
      edgecolors="black",
      zorder=3,
  )
  for _, r in sub.iterrows():
    axes[1].annotate(
        r["drug_name"],
        (r["chemical_similarity"], r["target_coverage"]),
        xytext=(5, 4),
        textcoords="offset points",
        fontsize=9,
    )
  axes[1].set_xlabel(f"Tanimoto Similarity to {reference_drug}")
  axes[1].set_ylabel("Weighted Target Coverage")
  axes[1].set_title("Mechanism vs. Structural Similarity")
  axes[1].set_ylim(-0.05, 1.1)
  axes[1].grid(True, linestyle="--", alpha=0.4)

  plt.tight_layout()
  return fig


def build_3d_sdf_block(mol) -> str | None:
  if mol is None:
    return None
  try:
    m3 = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = RANDOM_SEED
    if AllChem.EmbedMolecule(m3, params) == 0:
      AllChem.MMFFOptimizeMolecule(m3, maxIters=200)
      return Chem.MolToMolBlock(m3)
  except Exception:
    pass
  return None


def render_side_by_side_3d(
    scored_df: pd.DataFrame, ranked_df: pd.DataFrame, selected_ref: str, top_n: int = 3
):
  mol_map = dict(zip(scored_df["drug_name"], scored_df["mol"]))
  sim_map = dict(zip(ranked_df["drug_name"], ranked_df["chemical_similarity"]))
  names_3d = [selected_ref] + [
      n for n in ranked_df["drug_name"].head(top_n).tolist() if n != selected_ref
  ]

  items = []
  skipped = []
  for name in names_3d:
    sdf_block = build_3d_sdf_block(mol_map.get(name))
    if not sdf_block:
      skipped.append(name)
      continue
    role = "Reference" if name == selected_ref else "Candidate"
    sim = sim_map.get(name)
    sim_str = f" | Sim: {sim:.2f}" if sim is not None and sim == sim else ""

    v = py3Dmol.view(width=280, height=230)
    v.addModel(sdf_block, "sdf")
    v.setStyle({
        "stick": {"radius": 0.18, "colorscheme": "Jmol"},
        "sphere": {"scale": 0.24, "colorscheme": "Jmol"},
    })
    v.setBackgroundColor("white", 0.0)
    v.zoomTo()
    items.append((role, name, sim_str, v._make_html()))

  expected_count = len(names_3d)
  actual_count = len(items)
  if actual_count < expected_count:
    reasons = []
    if len(ranked_df) < top_n:
      reasons.append(
          f"only {len(ranked_df)} candidate(s) available out of {top_n} requested"
      )
    if skipped:
      reasons.append(f"3D conformer generation failed for {', '.join(skipped)}")
    if not reasons:
      reasons.append("some molecules were excluded or lacked valid 3D coordinates")
    st.caption(
        f"Showing {actual_count} of {expected_count} requested 3D structures — "
        + "; ".join(reasons)
        + "."
    )

  if not items:
    st.info("No 3D conformers could be generated for display.")
    return

  cols_per_row = 3
  for start in range(0, len(items), cols_per_row):
    row_items = items[start : start + cols_per_row]
    cols = st.columns(cols_per_row)
    for col, (role, name, sim_str, viewer_html) in zip(cols, row_items):
      with col:
        st.markdown(
            f"<div style='text-align:center; font-size:13px; margin-bottom:2px;'>"
            f"<b>{role}: {name}</b>{sim_str}<br>"
            f"<span style='opacity:0.75; font-size:11px;'>Drag to rotate · Scroll to zoom</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        components.html(viewer_html, height=245, scrolling=False)


def build_route_graph(drugs_df: pd.DataFrame, dt_df: pd.DataFrame) -> nx.DiGraph:
  g = nx.DiGraph()
  for r in drugs_df.itertuples():
    drug = _txt(getattr(r, "drug_name", ""))
    if not drug:
      continue
    g.add_node(drug, node_type="drug")
    for gene in getattr(r, "targets_set", set()) or set():
      gene_c = _norm_gene(gene)
      if gene_c:
        g.add_node(gene_c, node_type="gene")
        g.add_edge(drug, gene_c, weight=1.0, distance=1.0, relation="targets")

  for r in dt_df.itertuples():
    gene = _norm_gene(getattr(r, "gene", ""))
    disease = _txt(getattr(r, "disease", ""))
    pathway = _txt(getattr(r, "pathway", ""))
    assoc = max(0.01, min(1.0, float(getattr(r, "association_score", 1.0) or 1.0)))
    cost = round(1.0 / assoc, 4)

    if gene and pathway:
      g.add_node(gene, node_type="gene")
      g.add_node(pathway, node_type="pathway")
      g.add_edge(gene, pathway, weight=assoc, distance=cost, relation="in_pathway")
    if pathway and disease:
      g.add_node(pathway, node_type="pathway")
      g.add_node(disease, node_type="disease")
      g.add_edge(pathway, disease, weight=assoc, distance=cost, relation="linked_to_disease")
    if gene and disease:
      g.add_node(gene, node_type="gene")
      g.add_node(disease, node_type="disease")
      g.add_edge(gene, disease, weight=assoc, distance=cost, relation="associated_with")
  return g


def find_best_route(g: nx.DiGraph, source: str, target: str) -> dict:
  if source not in g or target not in g:
    return {"found": False, "route": [], "total_cost": None, "steps": []}
  try:
    path = nx.shortest_path(g, source=source, target=target, weight="distance")
    total_cost = nx.shortest_path_length(g, source=source, target=target, weight="distance")
    steps = []
    for u, v in zip(path[:-1], path[1:]):
      ed = g.get_edge_data(u, v, default={})
      steps.append({
          "from": u,
          "to": v,
          "relation": ed.get("relation", ""),
          "weight": ed.get("weight", 1.0),
          "distance": ed.get("distance", 1.0),
      })
    return {"found": True, "route": path, "total_cost": round(float(total_cost), 4), "steps": steps}
  except nx.NetworkXNoPath:
    return {"found": False, "route": [], "total_cost": None, "steps": []}


# ==============================================================================
# UI & EXECUTION
# ==============================================================================
st.title("🧪 Drug Repurposing")
st.caption(
    "Upload any dataset format or use built-in demo data to rank candidates,"
    " inspect Lipinski descriptors, view diagnostic charts, and rotate 3D conformers side by side."
)

with st.sidebar:
  st.header("1. Data Input")
  uploaded_files = st.file_uploader(
      "Upload drug & disease files (any format)",
      accept_multiple_files=True,
  )
  use_demo = st.checkbox(
      "Use built-in demo data when no files uploaded",
      value=True,
  )

  st.header("2. Scoring Weights")
  target_weight = st.slider("Target Coverage Weight", 0.0, 1.0, 0.60, 0.05)
  chem_weight = round(1.0 - target_weight, 2)
  st.write(f"Chemical Similarity Weight: **{chem_weight:.2f}**")
  top_3d_n = st.slider("Top Candidates in 3D View", 1, 5, 3)

if uploaded_files:
  drugs_raw, disease_raw = split_uploaded_tables(uploaded_files)
  drugs_raw, disease_raw = ensure_schemas(drugs_raw, disease_raw)
elif use_demo:
  drugs_raw, disease_raw = DEMO_DRUGS.copy(), DEMO_DISEASE.copy()
else:
  st.warning("Upload at least one file or enable built-in demo data in the sidebar.")
  st.stop()

drugs = drugs_raw.copy()
drugs["drug_name"] = drugs["drug_name"].map(_txt)
drugs["smiles"] = drugs["smiles"].map(_txt)
drugs["targets_set"] = drugs["targets"].map(_parse_targets)
drugs = drugs[drugs["drug_name"].ne("")].drop_duplicates("drug_name").reset_index(drop=True)

dt = disease_raw.copy()
dt["disease"] = dt["disease"].map(_txt)
dt["gene"] = dt["gene"].map(_norm_gene)
dt["pathway"] = dt["pathway"].map(_txt)
dt["pdb_id"] = dt["pdb_id"].map(lambda v: _txt(v).upper())
dt["association_score"] = (
    pd.to_numeric(dt["association_score"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
)
dt = dt[dt["disease"].ne("") & dt["gene"].ne("")].drop_duplicates(["disease", "gene", "pathway"]).reset_index(drop=True)

if drugs.empty or dt.empty:
  st.error("No valid drug or disease-target rows found after parsing.")
  st.stop()

diseases = sorted(dt["disease"].unique())
drug_list = sorted(drugs["drug_name"].tolist())
default_ref_idx = drug_list.index("Diclofenac") if "Diclofenac" in drug_list else 0

col_a, col_b = st.columns(2)
with col_a:
  selected_disease = st.selectbox("Target Disease", diseases)
with col_b:
  selected_ref = st.selectbox("Reference Drug", drug_list, index=default_ref_idx)

if st.button("Run Screening", type="primary"):
  with st.spinner("Retrieving structures, computing fingerprints, and ranking candidates..."):
    for i in drugs.index:
      if not drugs.loc[i, "smiles"]:
        drugs
