"""Export the numerical values reported in the manuscript to JSON."""
import json

import numpy as np
import pandas as pd

import paths as P

R = P.RESULTS

def J(name):
    return json.loads((R / name).read_text())

def f(x, d=2):
    s = f"{x:.{d}f}"
    return s

def ci(iv, d=2):
    return f"{f(iv[0], d)} to {f(iv[1], d)}"

def pct1(x):
    return f"{100 * x:.1f}"

def p_(x):
    return "<0.001" if x < 0.001 else f(x, 3) if x < 0.1 else f(x, 2)

def pct(x):
    return f"{100 * x:.0f}"

def main():
    v = {}
    cs = {c["corpus"]: c for c in J("corpus_structure.json")}; hl = J("corpus_structure_hlqc.json")
    v.update(DaicPersons=cs["daicwoz"]["units"], DaicWindows=f"{cs['daicwoz']['windows']:,}", DaicQuestions=cs["daicwoz"]["groups"],
             DaicWinPerPerson=int(cs["daicwoz"]["windows_per_unit_median"]), DaicWordsPerWin=int(cs["daicwoz"]["words_per_window_median"]),
             EdaicPersons=cs["edaic_new"]["units"], EdaicWindows=f"{cs['edaic_new']['windows']:,}",
             EdaicWinPerPerson=int(cs["edaic_new"]["windows_per_unit_median"]), EdaicWordsPerWin=int(cs["edaic_new"]["words_per_window_median"]),
             AllPersons=cs["daicwoz"]["units"] + cs["edaic_new"]["units"],
             AlexSessions=cs["alexst"]["units"], AlexTurns=f"{cs['alexst']['windows']:,}", AlexTurnsPerSession=int(cs["alexst"]["windows_per_unit_median"]),
             AnnomiSessions=cs["annomi"]["units"], AnnomiTurns=f"{cs['annomi']['windows']:,}",
             HlqcSessions=hl["sessions"], HlqcTurns=f"{hl['turns']:,}", HlqcShared=hl["sessions_also_in_annomi"], HlqcIndep=hl["sessions_independent_of_annomi"])
    for c, name in [("annomi", "Annomi"), ("hlqc", "Hlqc")]:
        W = pd.read_parquet(P.DERIVED / f"windows_{c}.parquet"); s = W.groupby("pid").first()
        if c == "hlqc":
            s = s[~s["in_annomi"].astype(bool)]; name = "HlqcIndep"
            v["HlqcIndepTurns"] = f"{int(W['pid'].isin(s.index).sum()):,}"
        v[f"{name}High"] = int((s["quality"] == "high").sum()); v[f"{name}Low"] = int((s["quality"] == "low").sum())
    ib = pd.read_csv(P.ITEMS / "item_bank.csv")
    v.update(SymItems=int(ib["instrument"].isin(["PHQ-8", "PCL-C"]).sum()), WaiItems=36,
             WaiReverse=int(((ib["instrument"] == "WAI-C") & (ib["polarity"] < 0)).sum()))
    cal = J("nli_calibration.json")
    v.update(NliAtoms=42, NliNeutral=f(cal["neutral_mean"]), NliKdiag=f(cal["K_diag_mean"]), NliKoff=f(cal["K_offdiag_mean"]))

    wg = J("wording_geometry.json")
    v.update(WordRankMini=f(wg["minilm"]["symptoms"]["effective_rank"], 1), WordRankBge=f(wg["bge"]["symptoms"]["effective_rank"], 1))
    nd = wg["minilm"]["symptoms"]["near_duplicates"]
    v.update(DupSleep=f(nd["PHQ8_3_Sleep~PCL-C_13_Sleep"]), DupConc=f(nd["PHQ8_7_Concentration~PCL-C_15_Concentration"]),
             DupInterest=f(nd["PHQ8_1_NoInterest~PCL-C_9_NoInterest"]))
    for form in ["WAI-C", "WAI-T", "WAI-O"]:
        ff = np.mean([wg["minilm"][form][s]["cos_forward_forward"] for s in ["Task", "Bond", "Goal"]])
        fr = np.mean([wg["minilm"][form][s]["cos_forward_reverse"] for s in ["Task", "Bond", "Goal"]])
        k = form.replace("-", "")
        v[f"{k}FF"] = f(ff); v[f"{k}FR"] = f(fr)

    sim = J("sim_wording_null.json"); g = sim["part1_population_geometry"]["grid_mean_sd"]
    v.update(SimTruthWording=f(sim["part1_population_geometry"]["r_truth_vs_wording"]),
             SimNaiveZero=f(g["g=0.0,n=189,w=155"]["naive"]["r_truth"][0]), SimNaiveWordZero=f(g["g=0.0,n=189,w=155"]["naive"]["r_wording"][0]),
             SimWhiteZero=f(g["g=0.0,n=189,w=155"]["whitened_N0"]["r_truth"][0]),
             SimWhiteFour=f(g["g=0.4,n=189,w=155"]["whitened_N0"]["r_truth"][0]), SimNaiveFour=f(g["g=0.4,n=189,w=155"]["naive"]["r_truth"][0]),
             SimWhiteBeyondFour=f(g["g=0.4,n=189,w=155"]["whitened_N0"]["r_truth_beyond_wording"][0]),
             SimNaiveBeyondFour=f(g["g=0.4,n=189,w=155"]["naive"]["r_truth_beyond_wording"][0]),
             SimDeconvSD=f(g["g=0.2,n=189,w=155"]["deconvolved_ridge0.1"]["r_truth"][1]),
             SimPermTypeOne=pct(sim["part1_population_geometry"]["null_type1_at_0.05_partial_item_permutation"]["whitened_N0"]),
             SimScriptRaw=f(sim["part2_interview_script"]["g=0.0"]["raw"][0], 3))
    ws = J("wording_selfreport.json")
    for split, S in [("discovery", "D"), ("confirmation", "C")]:
        for sc, k in [("minilm", "Mini"), ("bge", "Bge")]:
            x = ws[split][sc]
            v.update({f"{S}{k}SrWord": f(x["r_selfreport_wording"]), f"{S}{k}NaiveSr": f(x["r_naive_selfreport"]),
                      f"{S}{k}NaiveSrPartial": f(x["partial_naive_selfreport_given_wording"])})
    mf, me = J("method_figure.json"), J("method_examples.json")
    hb = J("hub_sources.json"); hn = hb["names"]; HS = np.array(hb["spearman"])
    def hs(a, b):
        return f(HS[hn.index(a), hn.index(b)])
    EmbD, EmbE, EntD, EntE = "language, embedding (DAIC-WOZ)", "language, embedding (E-DAIC)", "language, entailment (DAIC-WOZ)", "language, entailment (E-DAIC)"
    SelfD, SelfE, Word, Probe, Theory = "self-report (DAIC-WOZ)", "self-report (E-DAIC)", "item wording (MiniLM)", "entailment probe geometry", "theory: scale clusters"
    v.update(HubEmbDE=hs(EmbD, EmbE), HubEmbWordD=hs(EmbD, Word), HubEmbWordE=hs(EmbE, Word), HubEmbSelfD=hs(EmbD, SelfD),
             HubEmbSelfE=hs(EmbE, SelfE), HubSelfDE=hs(SelfD, SelfE), HubSelfWordD=hs(SelfD, Word), HubSelfWordE=hs(SelfE, Word),
             HubEntWordD=hs(EntD, Word), HubEntWordE=hs(EntE, Word), HubEntDE=hs(EntD, EntE), HubEntProbeD=hs(EntD, Probe),
             HubEntProbeE=hs(EntE, Probe),
             HubTheoryMax=f(max(HS[hn.index(Theory), k] for k in range(len(hn)) if hn[k] != Theory)),
             MethodRScore=f(mf["r_language_phq8_score_vs_selfreport_daic"]))
    rl = hb["split_half_reliability"]; HT = np.array(hb["kendall_tau_a"]); iu_h = np.triu_indices(len(hn), 1)
    from scipy.stats import spearmanr as _sr
    v.update(RelEmbD=f(rl[EmbD]), RelEmbE=f(rl[EmbE]), RelEntD=f(rl[EntD]), RelEntE=f(rl[EntE]), RelSelfD=f(rl[SelfD]), RelSelfE=f(rl[SelfE]),
             HubEmbSelfDNorm=f(HS[hn.index(EmbD), hn.index(SelfD)] / np.sqrt(rl[EmbD] * rl[SelfD])),
             HubEmbWordDNorm=f(HS[hn.index(EmbD), hn.index(Word)] / np.sqrt(rl[EmbD])),
             HubSelfDENorm=f(HS[hn.index(SelfD), hn.index(SelfE)] / np.sqrt(rl[SelfD] * rl[SelfE])),
             HubTauAgree=f(_sr(HS[iu_h], HT[iu_h])[0]))
    v.update(MethodRWording=f(mf["r_signal_free_geometry_vs_wording"]),
             StanceCosFwd=f(me["stance"]["cosine"][0]), StanceCosRev=f(me["stance"]["cosine"][1]),
             StanceNliFwd=f(me["stance"]["entailment"][0]), StanceNliRev=f(me["stance"]["entailment"][1]))
    fl = J("fl_calibration.json")
    v.update(FLNullN=f"{100 * fl['n=189,g=0.0']['share_p_below_005']:.1f}", FLNullS=f"{100 * fl['n=86,g=0.0']['share_p_below_005']:.1f}",
             FLPowerFourN=pct(fl["n=189,g=0.4"]["share_p_below_005"]), FLPowerTwoN=pct(fl["n=189,g=0.2"]["share_p_below_005"]),
             FLPowerFourS=pct(fl["n=86,g=0.4"]["share_p_below_005"]), FLReps=fl["n=189,g=0.0"]["reps"])
    rk = sim["part3_reverse_keying"]
    v.update(SimFwdRev=f(rk["cos_forward_vs_reverse_items"]))
    for s in ["Task", "Bond", "Goal"]:
        v[f"SimCos{s}"] = f(rk[s]["r_keyed_cosine_vs_alliance"]); v[f"SimStance{s}"] = f(rk[s]["r_keyed_stance_vs_alliance"])
        v[f"SimTalk{s}"] = f(rk[s]["r_keyed_cosine_vs_share_of_talk"])

    im = J("imposed_structure_minilm.json")
    for c, k in [("daicwoz", "Daic"), ("edaic_new", "Edaic")]:
        d = im[c]
        v.update({f"{k}EtaQ": f(d["eta2_group_median"], 3), f"{k}EtaQlo": f(d["eta2_group_range"][0], 3), f"{k}EtaQhi": f(d["eta2_group_range"][1], 3),
                  f"{k}EtaP": f(d["eta2_person_median"], 3), f"{k}RWindow": f(d["r_wording"]["window_raw"]),
                  f"{k}RPerson": f(d["r_wording"]["person_raw"]), f"{k}RPersonCentred": f(d["r_wording"]["person_centred"]),
                  f"{k}RWhitened": f(d["r_wording"]["person_centred_whitened"]),
                  f"{k}RelWhite": f(d["reliability_population"]["centred"]["whitened"]["spearman_brown"]),
                  f"{k}RelRaw": f(d["reliability_population"]["centred"]["raw"]["spearman_brown"]),
                  f"{k}RelDev": f(d["reliability_within_person"]["median_split_half_deviation"]),
                  f"{k}RelWithin": f(d["reliability_within_person"]["median_split_half_raw"])})

    names = {"minilm": "Mini", "bge": "Bge", "nli": "Nli"}
    for split, S in [("discovery", "D"), ("confirmation", "C")]:
        r = J(f"registered_{split}.json")
        for sc, k in names.items():
            x = r[sc]
            if "H1" in x:
                h = x["H1"]
                v.update({f"{S}{k}HoneW": f(h["r_naive_wording"]), f"{S}{k}HoneS": f(h["r_naive_selfreport"]),
                          f"{S}{k}HoneD": f(h["difference"]), f"{S}{k}HoneCI": ci(h["interval"])})
            h = x["H2"]
            v.update({f"{S}{k}HtwoR": f(h["partial_r"], 3), f"{S}{k}HtwoP": p_(h["p_freedman_lane"])})
            if "p_holm" in h:
                v[f"{S}{k}HtwoPholm"] = p_(h["p_holm"])
            h = x["H3"]
            v.update({f"{S}{k}HthreeSame": f(h["mean_same_item_r"], 3), f"{S}{k}HthreeDiff": f(h["mean_different_item_r"], 3),
                      f"{S}{k}HthreeD": f(h["difference"], 3), f"{S}{k}HthreeCI": ci(h["interval"], 3)})
    gf = J("general_factor.json")
    for split, S in [("discovery", "D"), ("confirmation", "C")]:
        for sc, k in names.items():
            x = gf[split][sc]
            v.update({f"{S}{k}G": f(x["r_glang_gself"]), f"{S}{k}GCI": ci(x["ci_r_glang_gself"]),
                      f"{S}{k}Spec": f(x["specific_delta"], 3), f"{S}{k}SpecCI": ci(x["ci_specific_delta"], 3),
                      f"{S}{k}MeanPHQ": f(x["r_langmean_phq"])})

    cm = J("compass1_reexamined_minilm.json")
    v.update(CompSessions=cm["sessions"], CompBoth=cm["sessions_with_both_roles"], CompMulti=cm["sessions_with_several_tags"],
             CompAnx=cm["tag_counts"]["anxiety"], CompDep=cm["tag_counts"]["depression"], CompScz=cm["tag_counts"]["schizophrenia"],
             CompSui=cm["tag_counts"]["suicidal"])
    for role, k in [("client", "Cl"), ("therapist", "Th")]:
        for s in ["Task", "Bond", "Goal"]:
            v[f"Comp{k}{s}Rsq"] = f(cm[role][s]["r2_keyed_by_topic"]); v[f"Comp{k}{s}Slope"] = f(cm[role][s]["slope"], 1)
            v[f"Comp{k}{s}EtaColl"] = f(cm[role][s]["eta2_collection_keyed"], 3)
            v[f"Comp{k}{s}EtaBeyond"] = f(cm[role][s]["eta2_collection_keyed_beyond_topic"], 3)
    v["CompEtaCollMax"] = f(max(cm[r][s]["eta2_collection_keyed"] for r in ["client", "therapist"] for s in ["Task", "Bond", "Goal"]), 2)
    off = cm["role_offset_therapist_minus_client"]
    v["CompGapBond"] = f(off["Bond"]["r_keyed_gap_with_topic_gap"]); v["CompGapTask"] = f(off["Task"]["r_keyed_gap_with_topic_gap"])
    v["CompGapGoal"] = f(off["Goal"]["r_keyed_gap_with_topic_gap"])

    rd = J("rdoc_wording.json")
    v.update(RdocDefs=rd["n_definitions"], RdocDomains=len(rd["per_domain"]), RdocNNMini=f(rd["minilm"]["nearest_neighbour_same_domain"]),
             RdocNNBge=f(rd["bge"]["nearest_neighbour_same_domain"]), RdocChance=f(rd["minilm"]["nearest_neighbour_chance"]),
             RdocRankMini=f(rd["minilm"]["effective_rank"], 1), RdocRankBge=f(rd["bge"]["effective_rank"], 1),
             RdocWithinMini=f(rd["minilm"]["mean_cos_within_domain"]), RdocBetweenMini=f(rd["minilm"]["mean_cos_between_domain"]))

    mb = J("multibank.json")
    v.update(MbItems=mb["items"], MbBanks=len(mb["banks"]),
             MbRankAllMini=f(mb["minilm"]["all_items_effective_rank"], 1), MbRankAllMiniMR=f(mb["minilm"]["all_items_effective_rank_mean_removed"], 1),
             MbRankAllBge=f(mb["bge"]["all_items_effective_rank"], 1), MbRankAllBgeMR=f(mb["bge"]["all_items_effective_rank_mean_removed"], 1))
    pbm, pbb = mb["minilm"]["per_bank"], mb["bge"]["per_bank"]
    v.update(MbBgeRawMin=f(min(x["effective_rank"] for x in pbb.values()), 1), MbBgeRawMax=f(max(x["effective_rank"] for x in pbb.values()), 1),
             MbBgeMRMax=f(max(x["effective_rank_mean_removed"] for x in pbb.values()), 1),
             MbMiniIpip=f(pbm["IPIP50"]["effective_rank"], 1), MbMiniIpipN=pbm["IPIP50"]["n"],
             MbNNBankPHQ=f(mb["minilm"]["nn_in_other_bank"]["PHQ-8"]), MbNNBankWAIT=f(mb["minilm"]["nn_in_other_bank"]["WAI-T"]),
             MbNNDomMax=f(max(mb["minilm"]["nn_in_other_domain"].values())),
             MbNNDomMedian=f(float(np.median(list(mb["minilm"]["nn_in_other_domain"].values())))),
             MbRankN=f(mb["bge"]["rank_vs_n_corr"]), MbInstruments=len(mb["banks"]) - 1,
             MbNNDomTIPI=f(mb["minilm"]["nn_in_other_domain"]["TIPI"]), MbNNDomWHO=f(mb["minilm"]["nn_in_other_domain"]["WHO-5"]),
             MbNNDomMedianBge=f(float(np.median(list(mb["bge"]["nn_in_other_domain"].values())))),
             MbNNBankGAD=f(mb["minilm"]["nn_in_other_bank"]["GAD-7"]) if "GAD-7" in mb["minilm"]["nn_in_other_bank"] else "--",
             MbNNBankPCLfive=f(mb["minilm"]["nn_in_other_bank"]["PCL-5"]) if "PCL-5" in mb["minilm"]["nn_in_other_bank"] else "--")

    for split, S in [("discovery", "D"), ("confirmation", "C")]:
        p = R / f"alliance_{split}.json"
        if p.exists():
            a = J(f"alliance_{split}.json")
            v.update({f"{S}AlSessions": a["sessions"], f"{S}AlHigh": a["high"], f"{S}AlLow": a["low"],
                      f"{S}AlAucStance": f(a["A1"]["auc_stance"]), f"{S}AlAucStanceCI": ci(a["A1"]["interval"]),
                      f"{S}AlAucCos": f(a["A2"]["auc_keyed_cosine"]), f"{S}AlDiff": f(a["A2"]["difference"], 3), f"{S}AlDiffCI": ci(a["A2"]["interval"], 3),
                      f"{S}AlAucCompass": f(a["secondary"]["compass2025_total"]["auc"]),
                      f"{S}AlAucCompassCI": ci(a["secondary"]["compass2025_total"]["interval"]),
                      f"{S}AlAOne": "supported" if a["A1"]["supported"] else "not supported",
                      f"{S}AlATwo": "supported" if a["A2"]["supported"] else ("not supported" if a["A2"]["tested"] else "not tested")})
            for s in ["Task", "Bond", "Goal"]:
                v[f"{S}AlStance{s}"] = f(a["secondary"][f"stance_{s}"]["auc"]); v[f"{S}AlCos{s}"] = f(a["secondary"][f"cos_{s}"]["auc"])
            for kk, val in a["secondary"]["correlations"].items():
                nm = "".join(w.capitalize() for w in kk.replace("~", "_").replace("2025", "").split("_"))
                v[f"{S}AlCor{nm}"] = f(val)

    ss, al, se = J("symptom_sensitivity.json"), J("alliance_sensitivity.json"), J("sim_estimators.json")
    for split, S_ in [("discovery", "D"), ("confirmation", "C")]:
        for sc, k in [("minilm", "Mini"), ("bge", "Bge"), ("nli", "Nli")]:
            key = f"{split}:{sc}"; sh = ss["shuffle"][key]; nn = ss["noise_normalized"][key]; bp = ss["h2_block_permutation"][key]
            v.update({f"{S_}{k}HtwoCI": ci(ss["h2_interval"][key]), f"{S_}{k}HtwoBlockP": p_(bp["p"]),
                      f"{S_}{k}ShufGeo": f(sh["geometry_agreement"]["true"]), f"{S_}{k}ShufGeoNull": f(sh["geometry_agreement"]["null_mean"]),
                      f"{S_}{k}ShufG": f(sh["general_factor_r"]["true"]), f"{S_}{k}ShufGNull": ci(sh["general_factor_r"]["null_95"]),
                      f"{S_}{k}ShufPHQ": f(sh["phq8_score_r"]["true"]), f"{S_}{k}ShufPHQNull": ci(sh["phq8_score_r"]["null_95"]),
                      f"{S_}{k}NNRef": f(nn["r_reference_vs_wording"]),
                      f"{S_}{k}NNr": f(nn["registered_controls"]["partial_r"], 3), f"{S_}{k}NNp": p_(nn["registered_controls"]["p_freedman_lane"]),
                      f"{S_}{k}NNCI": ci(nn["registered_controls"]["interval"])})
    nnr = [ss["noise_normalized"][k]["registered_controls"]["partial_r"] for k in ss["noise_normalized"]]
    nnref = [ss["noise_normalized"][k]["r_reference_vs_wording"] for k in ss["noise_normalized"] if not k.endswith("nli")]
    v.update(NNrMin=f(min(nnr), 3), NNrMax=f(max(nnr), 3), NNRefMin=f(min(nnref)), NNRefMax=f(max(nnref)),
             NNpMin=p_(min(ss["noise_normalized"][k]["registered_controls"]["p_freedman_lane"] for k in ss["noise_normalized"])))
    pb = J("paired_baselines.json")
    for split, S_ in [("discovery", "D"), ("confirmation", "C")]:
        for bk, bn in [("windows", "Win"), ("words", "Words"), ("topic_activation", "Topic")]:
            v[f"{S_}Base{bn}"] = f(pb[split]["baselines_r_phq8"][bk]); v[f"{S_}Base{bn}CI"] = ci(pb[split]["baselines_r_phq8_ci"][bk])
        for sc, k in [("minilm", "Mini"), ("bge", "Bge"), ("nli", "Nli")]:
            x = pb[split][sc]
            v.update({f"{S_}{k}PhqR": f(x["phq8_score_r"]), f"{S_}{k}PhqPart": f(x["phq8_score_partial_r"]),
                      f"{S_}{k}PhqPartCI": ci(x["ci"]["phq8_score_partial_r"]), f"{S_}{k}GPart": f(x["general_factor_partial_r"]),
                      f"{S_}{k}GPartCI": ci(x["ci"]["general_factor_partial_r"])})
    bpc = ss["h2_block_permutation"]["confirmation:nli"]["p"]
    v.update(BlockNliCP=p_(bpc), BlockNliCPHolmish=p_(min(1.0, 3 * bpc)))
    w683 = ss["confirmation_without_683"]
    v.update(PCLZeroPerson="683", NoSixEightThreeNliHtwoP=p_(w683["nli"]["H2"]["p_freedman_lane"]))
    pv = pd.DataFrame(al["provenance"]["counts"])
    cnt = lambda ex, ids, txt: int(pv[(pv.registered_excluded == ex) & (pv.id_status == ids) & (pv.text_status == txt)]["n"].sum())
    v.update(HlqcSharedID=cnt(True, "ID match", "text match") + cnt(True, "ID match", "screen-negative"),
             HlqcSharedIDScreenNeg=cnt(True, "ID match", "screen-negative"),
             HlqcUnresolved=int(pv[pv.registered_excluded & (pv.id_status != "ID match")]["n"].sum()),
             HlqcUnresolvedText=cnt(True, "ID unresolved", "text match") + cnt(True, "ID distinct", "text match"),
             HlqcUnresolvedNeg=cnt(True, "ID unresolved", "screen-negative") + cnt(True, "ID distinct", "screen-negative"),
             HlqcTextDependent=cnt(False, "ID distinct", "text match"), HlqcVideos=al["registered_184"]["clusters"],
             TextThreshold=f(al["provenance"]["threshold"], 1),
             GapBelow=f(al["provenance"]["registered_sample_containment_max_below"], 3), GapAbove=f(al["provenance"]["registered_sample_containment_min_above"]))
    for key, k in [("registered_184", "Clu"), ("registered_screen_negative", "ScreenNeg"), ("all_screen_negative", "AllNeg"),
                   ("registered_184_untruncated", "Untr")]:
        x = al[key]
        v.update({f"Al{k}N": x["sessions"], f"Al{k}V": x["clusters"], f"Al{k}Auc": f(x["auc_stance"]), f"Al{k}AucCI": ci(x["interval_stance"]),
                  f"Al{k}Cos": f(x["auc_cos"]), f"Al{k}DiffCI": ci(x["interval_difference"], 3)})
    v.update(AlTruncShare=pct1(al["registered_184_untruncated"]["segments_over_limit_share"]), AlTruncSegments=f"{al['registered_184_untruncated']['segments']:,}",
             AlSensAucMin=f(al["range_stance_auc"][0]), AlSensAucMax=f(al["range_stance_auc"][1]),
             AlSensLowMin=f(al["range_difference_lower_bound"][0], 3), AlSensLowMax=f(al["range_difference_lower_bound"][1], 3))
    A4, A0 = se["A_estimators"]["n=189,g=0.4"], se["A_estimators"]["n=189,g=0.0"]; B_ = se["B_anisotropic_nuisance"]
    v.update(EstWhiteFour=f(A4["whitened"]["r_truth"][0]), EstWhiteFourSD=f(A4["whitened"]["r_truth"][1]),
             InvFailG=A4["inverse_failures"]["g2_nonpositive"], InvFailDiag=A4["inverse_failures"]["diag_nonpositive"],
             InvIndef=A4["inverse_failures"]["indefinite"], InvAdmN=A4["inverse_admissible_n"],
             InvAdmR=f(A4["inverse_admissible"]["r_truth"][0]), InvAdmSD=f(A4["inverse_admissible"]["r_truth"][1]),
             InvPsdR=f(A4["inverse_psd"]["r_truth"][0]), InvPsdSD=f(A4["inverse_psd"]["r_truth"][1]),
             InvFailZero=A0["inverse_failures"]["inadmissible"])
    sc = {"Match": "matched nuisance", "Strong": "between-person nuisance stronger", "Dir": "nuisance in different directions", "Within": "genuine within-person dynamics"}
    for k, name in sc.items():
        for g, gk in [("g=0.0", "Null"), ("g=0.4", "Pow")]:
            v[f"Sc{k}{gk}Word"] = pct1(B_[name][g]["wording_reference"]["rejection_rate"])
            v[f"Sc{k}{gk}Within"] = pct1(B_[name][g]["within_person_reference"]["rejection_rate"])
            v[f"Sc{k}{gk}WordR"] = f(B_[name][g]["wording_reference"]["mean_partial_r"])
            v[f"Sc{k}{gk}WithinR"] = f(B_[name][g]["within_person_reference"]["mean_partial_r"])
    v.update(CondM=f(np.linalg.cond((lambda E: E @ E.T)(np.load(P.DERIVED / "emb_minilm_items.npy").astype(float)[ib["instrument"].isin(["PHQ-8", "PCL-C"]).to_numpy()])), 0))
    v.update(MbDupTexts=mb["minilm"]["duplicate_texts"])
    sing = 0
    for c in ["daicwoz", "edaic_new"]:
        Wc = pd.read_parquet(P.DERIVED / f"windows_{c}.parquet"); gp = Wc.assign(g=Wc["group"].fillna("none")).groupby("g")["pid"].nunique()
        sing += int(Wc["group"].fillna("none").map(gp).eq(1).sum())
    v.update(SingletonWindows=sing)
    out = P.RESULTS / "paper_values.json"
    out.write_text(json.dumps(v, indent=2))
    print(len(v), "reported values written")

if __name__ == "__main__":
    main()
