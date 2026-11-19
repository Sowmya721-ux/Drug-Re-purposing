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
        "lipinski_pass": lipinski_
