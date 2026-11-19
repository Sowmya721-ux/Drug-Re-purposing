!pip install -q rdkit py3Dmol

import datetime
import io
import json
import os
import re
import zipfile
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import py3Dmol
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors

# ==============================================================================
# 1. CONFIGURATION, GENE SYNONYMS & SHARED STATE
# ==============================================================================
TARGET_WEIGHT = 0.60
CHEMICAL_WEIGHT = 0.40
RANDOM_SEED = 42

TARGET_SYNONYMS = {
    "COX1": "PTGS1",
    "COX-1": "PTGS1",
    "PGHS1": "PTGS1",
    "COX2": "PTGS2",
    "COX-2": "PTGS2",
    "PGHS2": "PTGS2",
}

upload_state = {"files": {}}
pipeline_outputs = {}


def _txt(x) -> str:
  return "" if pd.isna(x) else str(x).strip()


def _norm_gene(symbol: str) -> str:
  s = _txt(symbol).upper()
  return TARGET_SYNONYMS.get(s, s)


def _parse_targets(x) -> set[str]:
  return {_norm_gene(v) for v in re.split(r"[;,|]", _txt(x)) if _txt(v)}


# ==============================================================================
# 2. UNIVERSAL FILE PARSER & SCHEMA NORMALIZER
# ==============================================================================
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


def split_uploaded_tables(uploaded_map: dict[str, bytes]):
  parsed = []
  for fname, content in uploaded_map.items():
    df = parse_any_file_to_df(fname, content)
    df.columns = [str(c).strip() for c in df.columns]
    parsed.append((fname, df))

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


def _ensure_schemas(drugs_raw: pd.DataFrame, disease_raw: pd.DataFrame):
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


# ==============================================================================
# 3. MOLECULES, PHYSICOCHEMICAL DESCRIPTORS, CHARTS & TRANSPARENT 3D VIEW
# ==============================================================================
def fetch_smiles_pubchem(name: str) -> str:
  url = (
      "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
      + requests.utils.quote(name)
      + "/property/CanonicalSMILES,IsomericSMILES/JSON"
  )
  try:
    r = requests.get(url, timeout=25)
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


def plot_researcher_charts(
    ranked_df: pd.DataFrame,
    reference_drug: str,
    disease_choice: str,
    target_weight: float = TARGET_WEIGHT,
    chem_weight: float = CHEMICAL_WEIGHT,
):
  if ranked_df.empty:
    print("No ranked candidates available to plot.")
    return

  fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
  cov = ranked_df["target_coverage"].fillna(0.0).to_numpy()
  sim = ranked_df["chemical_similarity"].fillna(0.0).to_numpy()
  c_part = target_weight * cov
  s_part = chem_weight * sim
  y_pos = np.arange(len(ranked_df))

  axes[0].barh(
      y_pos,
      c_part,
      color="#2a9d8f",
      label=f"{target_weight:.2f} × Target Coverage",
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

  valid_mask = ranked_df["chemical_similarity"].notna()
  sub = ranked_df[valid_mask]
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
  plt.show()


def _build_3d_sdf_block(mol) -> str | None:
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


def _render_side_by_side_3d_no_bg(
    scored_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    selected_ref: str,
    top_n: int = 3,
):
  mol_map = dict(zip(scored_df["drug_name"], scored_df["mol"]))
  sim_map = (
      dict(zip(ranked_df["drug_name"], ranked_df["chemical_similarity"]))
      if "chemical_similarity" in ranked_df.columns
      else {}
  )

  names_3d = [selected_ref] + [
      n for n in ranked_df["drug_name"].head(top_n).tolist() if n != selected_ref
  ]

  cards_html = []
  for name in names_3d:
    sdf_block = _build_3d_sdf_block(mol_map.get(name))
    if not sdf_block:
      print(f"Skipping 3D view for {name}: 3D conformer could not be generated.")
      continue

    role = "Reference" if name == selected_ref else "Candidate"
    sim = sim_map.get(name)
    sim_str = f" | Sim: {sim:.2f}" if sim is not None and sim == sim else ""

    v = py3Dmol.view(width=320, height=260)
    v.addModel(sdf_block, "sdf")
    v.setStyle({
        "stick": {"radius": 0.18, "colorscheme": "Jmol"},
        "sphere": {"scale": 0.24, "colorscheme": "Jmol"},
    })
    v.setBackgroundColor("white", 0.0)
    v.zoomTo()

    viewer_html = v._make_html()
    card = (
        "<div style='display:inline-block; margin:6px 10px; "
        "background:transparent; vertical-align:top;'>"
        "<div style='text-align:center; font-size:12px; margin-bottom:4px; "
        "background:transparent;'>"
        f"<b>{role}: {name}</b>{sim_str}<br>"
        "<span style='opacity:0.75;'>Drag to rotate · Scroll to zoom</span>"
        "</div>"
        f"<div style='background:transparent;'>{viewer_html}</div>"
        "</div>"
    )
    cards_html.append(card)

  if cards_html:
    container = (
        "<div style='display:flex; flex-wrap:wrap; align-items:flex-start; "
        "background:transparent;'>"
        + "".join(cards_html)
        + "</div>"
    )
    display(HTML(container))
  else:
    print("No 3D conformers could be generated for display.")


def _export_results(
    ranked_df: pd.DataFrame,
    physicochem_df: pd.DataFrame,
    provenance_df: pd.DataFrame,
):
  ranked_df.to_csv("LITMUS_ranked_candidates.csv", index=False)
  physicochem_df.to_csv("LITMUS_physicochemical_properties.csv", index=False)
  provenance_df.to_csv("LITMUS_run_provenance.csv", index=False)

  zip_name = "LITMUS_research_bundle.zip"
  with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in [
        "LITMUS_ranked_candidates.csv",
        "LITMUS_physicochemical_properties.csv",
        "LITMUS_run_provenance.csv",
    ]:
      if os.path.exists(fname):
        zf.write(fname)
  print(f"Exported CSVs and bundle: {zip_name}")


# ==============================================================================
# 4. MAIN SCREENING PIPELINE & UPLOAD WIDGET UI
# ==============================================================================
def run_screening_from_uploads(
    disease_choice: str = "",
    reference_drug: str = "",
    out_widget: widgets.Output | None = None,
):
  target_out = out_widget if out_widget is not None else widgets.Output()
  if out_widget is None:
    display(target_out)

  with target_out:
    clear_output(wait=True)
    if not upload_state["files"]:
      print(
          "No files uploaded yet. Click 'Upload File(s)' first, then click"
          " 'Run Screening'."
      )
      return

    drugs_raw, disease_raw = split_uploaded_tables(upload_state["files"])
    if drugs_raw is None or disease_raw is None:
      print("Could not separate drug and disease-target tables.")
      return

    drugs_raw, disease_raw = _ensure_schemas(drugs_raw, disease_raw)

    drugs = drugs_raw.copy()
    drugs["drug_name"] = drugs["drug_name"].map(_txt)
    drugs["smiles"] = drugs["smiles"].map(_txt)
    drugs["targets_set"] = drugs["targets"].map(_parse_targets)
    drugs = (
        drugs[drugs["drug_name"].ne("")]
        .drop_duplicates("drug_name")
        .reset_index(drop=True)
    )

    dt = disease_raw.copy()
    dt["disease"] = dt["disease"].map(_txt)
    dt["gene"] = dt["gene"].map(_norm_gene)
    dt["pathway"] = dt["pathway"].map(_txt)
    dt["pdb_id"] = dt["pdb_id"].map(lambda v: _txt(v).upper())
    dt["association_score"] = (
        pd.to_numeric(dt["association_score"], errors="coerce")
        .fillna(1.0)
        .clip(0.0, 1.0)
    )
    dt = (
        dt[dt["disease"].ne("") & dt["gene"].ne("")]
        .drop_duplicates(["disease", "gene", "pathway"])
        .reset_index(drop=True)
    )

    if drugs.empty or dt.empty:
      print("Parsed tables have 0 valid rows.")
      return

    diseases = sorted(dt["disease"].unique())
    selected_disease = (
        disease_choice if disease_choice in diseases else diseases[0]
    )

    drug_list = sorted(drugs["drug_name"].tolist())
    default_ref = "Ibuprofen" if "Ibuprofen" in drug_list else drug_list[0]
    selected_ref = reference_drug if reference_drug in drug_list else default_ref

    print(f"Disease: {selected_disease} | Reference Drug: {selected_ref}")

    for i in drugs.index:
      if not drugs.loc[i, "smiles"]:
        drugs.loc[i, "smiles"] = fetch_smiles_pubchem(drugs.loc[i, "drug_name"])

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    drugs["mol"] = drugs["smiles"].map(
        lambda s: Chem.MolFromSmiles(s) if s else None
    )
    drugs["fingerprint"] = drugs["mol"].map(
        lambda m: fpgen.GetFingerprint(m) if m is not None else None
    )

    ref_fp = drugs.loc[drugs["drug_name"].eq(selected_ref), "fingerprint"].iloc[0]
    if ref_fp is None:
      drugs["chemical_similarity"] = np.nan
    else:
      drugs["chemical_similarity"] = drugs["fingerprint"].map(
          lambda fp: (
              float(DataStructs.TanimotoSimilarity(ref_fp, fp))
              if fp is not None
              else np.nan
          )
      )

    d_sub = (
        dt[dt["disease"].eq(selected_disease)]
        .groupby("gene", as_index=False)
        .agg(
            association_score=("association_score", "max"),
            pathway=(
                "pathway",
                lambda s: "; ".join(sorted({v for v in s if v})),
            ),
            pdb_id=("pdb_id", lambda s: next((v for v in s if v), "")),
        )
    )
    gene_weights = dict(zip(d_sub["gene"], d_sub["association_score"]))
    total_w = sum(gene_weights.values())

    matched_rows = []
    for tset in drugs["targets_set"]:
      matched = sorted(tset & set(gene_weights))
      w_hit = sum(gene_weights[g] for g in matched)
      matched_rows.append({
          "matched_targets": "; ".join(matched),
          "matched_target_count": len(matched),
          "target_coverage": (w_hit / total_w) if total_w else 0.0,
      })

    scored = pd.concat(
        [drugs.reset_index(drop=True), pd.DataFrame(matched_rows)], axis=1
    )
    cands = scored[scored["drug_name"].ne(selected_ref)].copy()
    cands["screening_score"] = np.where(
        cands["chemical_similarity"].isna(),
        cands["target_coverage"],
        TARGET_WEIGHT * cands["target_coverage"]
        + CHEMICAL_WEIGHT * cands["chemical_similarity"],
    )
    ranked = (
        cands.sort_values(
            ["screening_score", "target_coverage", "chemical_similarity"],
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    physicochem_df = build_physicochemical_table(scored)
    provenance_df = pd.DataFrame([{
        "timestamp_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "disease": selected_disease,
        "reference_drug": selected_ref,
        "target_weight": TARGET_WEIGHT,
        "chemical_weight": CHEMICAL_WEIGHT,
        "fingerprint": "Morgan(radius=2, nBits=2048)",
        "drugs_screened": len(ranked),
        "disease_targets": len(gene_weights),
        "top_candidate": (
            ranked.iloc[0]["drug_name"] if len(ranked) else "None"
        ),
        "disclaimer": (
            "Computational research shortlist only; experimental/clinical"
            " validation required."
        ),
    }])

    pipeline_outputs.update({
        "drugs": scored,
        "disease_targets": dt,
        "ranked": ranked,
        "physicochemical": physicochem_df,
        "provenance": provenance_df,
        "disease_choice": selected_disease,
        "reference_drug": selected_ref,
    })

    print("### 1. Ranked Candidates")
    display(
        ranked[[
            "rank",
            "drug_name",
            "chemical_similarity",
            "target_coverage",
            "screening_score",
            "matched_targets",
        ]]
    )

    print("### 2. Physicochemical & Lipinski Properties")
    display(physicochem_df)

    print("### 3. Diagnostic Charts")
    plot_researcher_charts(
        ranked, selected_ref, selected_disease, TARGET_WEIGHT, CHEMICAL_WEIGHT
    )

    print("### 4. Interactive Rotatable 3D Structures (No Background)")
    _render_side_by_side_3d_no_bg(scored, ranked, selected_ref, top_n=3)

    print("### 5. Run Provenance & Exports")
    display(provenance_df)
    _export_results(
        ranked[[
            "rank",
            "drug_name",
            "smiles",
            "approval_status",
            "source",
            "chemical_similarity",
            "target_coverage",
            "screening_score",
            "matched_targets",
        ]],
        physicochem_df,
        provenance_df,
    )


uploader = widgets.FileUpload(
    accept="",
    multiple=True,
    description="Upload File(s)",
    button_style="primary",
)
run_btn = widgets.Button(
    description="Run Screening",
    button_style="success",
    icon="play",
)
file_status = widgets.HTML("<i>No files uploaded yet.</i>")
results_box = widgets.Output()


def _handle_upload(change):
  val = change["new"]
  items = list(val.values()) if isinstance(val, dict) else list(val)
  for item in items:
    fname = item.get("name", "uploaded_file")
    content = item.get("content", b"")
    if isinstance(content, memoryview):
      content = content.tobytes()
    upload_state["files"][fname] = bytes(content)
  file_status.value = (
      f"<b>Uploaded ({len(upload_state['files'])}):</b> "
      + ", ".join(upload_state["files"].keys())
  )


def _handle_run(_btn):
  with results_box:
    clear_output(wait=True)
    try:
      run_screening_from_uploads(out_widget=results_box)
    except Exception as exc:
      print(f"Error while running pipeline: {exc}")


uploader.observe(_handle_upload, names="value")
run_btn.on_click(_handle_run)

display(
    widgets.VBox([
        widgets.HBox([uploader, run_btn]),
        file_status,
        results_box,
    ])
)
