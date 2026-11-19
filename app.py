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
      reasons.append(f"only {len(ranked_df)} candidate(s) available out of {top_n} requested")
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
