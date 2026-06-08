# crew.py
# Pipeline orchestrator — Mode A (ad sets only) and Mode B (ad sets + human responses)

import json, os, time
from datetime import datetime
from typing import Dict, List, Optional, Callable

from crewai import Crew, Process

from agents import (
    create_preprocessing_agent, create_persona_generator_agent,
    create_respondent_agent, create_risk_auditor_agent, create_analytics_agent,
)
from tasks import (
    task_preprocess, task_generate_personas, task_generate_responses,
    task_audit, task_analytics,
)
from tools.tools import (
    PersonaGeneratorTool, SurveyResponseTool, AnalyticsTool,
    generate_personas, generate_survey_response, compute_cohort_stats,
    compare_ai_human, export_responses_csv, get_rag_store, reset_rag_store,
    extract_text_from_file, RAGStore,
)
from fallback.fallback_handler import (
    validate_response, is_outlier, neutral_response,
    build_hitl_flag, handle_tool_error, needs_regeneration,
)
from monitoring.langfuse_config import (
    init_langfuse, log_event, TraceContext, flush,
)
from config import OUTPUTS_DIR


def run_pipeline(
    # ── Inputs ──────────────────────────────────────────────────────────────
    ads:               List[Dict],
    selected_scales:   Dict,
    persona_config:    Dict,
    brief_docs:        Dict[str, str],
    human_responses:   Optional[List[Dict]] = None,
    # ── Infrastructure ────────────────────────────────────────────────────
    api_key:           str = "",
    model:             str = "deepseek-chat",
    api_url:           str = "https://api.deepseek.com/v1/chat/completions",
    base_url:          str = "https://api.deepseek.com/v1",
    output_dir:        str = OUTPUTS_DIR,
    progress_cb:       Optional[Callable]  = None,
) -> Dict:
    """
    Run the full advertising pretest pipeline.

    Mode A: ads only           → synthetic responses CSV + visualisation stats
    Mode B: ads + human data   → Mode A + AI-human correlation/equivalence analysis

    Parameters
    ----------
    ads              : list of ad dicts (parsed from uploads or manual entry)
    selected_scales  : researcher-selected construct definitions
    persona_config   : persona generation settings from UI sliders
    brief_docs       : extracted text from uploaded brand/campaign brief PDFs
    human_responses  : parsed human survey responses (optional — triggers Mode B)
    api_key          : DeepSeek API key (from UI input)
    model            : LLM model name
    output_dir       : where to save CSV and JSON outputs
    progress_cb      : callback(step, total, message) for Gradio progress bar

    Returns
    -------
    dict with keys: mode, ads_processed, all_responses, cohort_stats,
                    comparison (Mode B only), hitl_flags, run_metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    mode       = "B" if human_responses else "A"
    start_time = datetime.now()
    scale_keys = list(selected_scales.keys())
    n_personas = persona_config.get("n_personas", 50)

    log_event("pipeline_start", "run", {
        "mode": mode, "n_ads": len(ads),
        "n_personas": n_personas, "scales": scale_keys,
    })

    def progress(step: int, total: int, msg: str):
        if progress_cb:
            progress_cb(step, total, msg)
        print(f"\n[{step}/{total}] {msg}")

    total_steps = 5
    all_results: Dict = {
        "mode": mode, "ads_processed": [],
        "all_responses": [], "cohort_stats": {},
        "comparison": None, "hitl_flags": [],
        "run_metadata": {},
    }

    # ── STEP 1: Build RAG store from uploaded brief documents ─────────────────
    progress(1, total_steps, "Building RAG store from uploaded documents...")
    reset_rag_store()
    rag = get_rag_store()

    if brief_docs:
        status = rag.build(brief_docs)
        log_event("rag_built", "rag_store", {"status": status})
        print(f"   {status}")
    else:
        print("   No brief documents uploaded — RAG will not be used")

    # ── STEP 2: Generate personas (once, reused across all ads) ───────────────
    progress(2, total_steps,
             f"Generating {n_personas} synthetic personas "
             f"(age {persona_config.get('age_min',18)}-{persona_config.get('age_max',65)}, "
             f"BF variance={persona_config.get('bf_variance',0.2)})...")

    with TraceContext("persona_generation", input_data=persona_config) as ctx:
        try:
            personas = generate_personas(
                n          = n_personas,
                seed       = persona_config.get("seed", 42),
                age_min    = persona_config.get("age_min", 18),
                age_max    = persona_config.get("age_max", 65),
                bf_variance= persona_config.get("bf_variance", 0.20),
                include_genders  = persona_config.get("genders"),
                income_levels    = persona_config.get("income_levels"),
            )
            ctx.set_output(f"Generated {len(personas)} personas")
            log_event("personas_ready", "persona_generation", {"n": len(personas)})
        except Exception as e:
            handle_tool_error("persona_generation", e)
            return {**all_results, "error": f"Persona generation failed: {e}"}

    # ── STEP 3: Evaluate each ad ───────────────────────────────────────────────
    progress(3, total_steps,
             f"Evaluating {len(ads)} ad(s) with {len(personas)} personas each...")

    for ad in ads:
        ad_id = ad.get("ad_id") or ad.get("brand", "ad") + "_1"
        ad["ad_id"] = ad_id
        print(f"\n  Processing: {ad_id}")

        # RAG retrieval for this ad
        rag_context = ""
        if rag.is_ready:
            query      = f"{ad.get('brand','')} {ad.get('product_category','')} brand positioning target audience"
            rag_context = rag.retrieve(query, top_k=4)
            if rag_context:
                print(f"   [RAG] Retrieved {len(rag_context)} chars of brand context")

        responses   = []
        hitl_flags  = []
        regen_count = 0

        for i, persona in enumerate(personas):
            if i % 10 == 0:
                print(f"   Responses: {i}/{len(personas)}...")

            with TraceContext(
                f"respondent_{persona['persona_id']}",
                input_data={"persona": persona["persona_id"], "ad": ad_id}
            ) as ctx:
                try:
                    response = generate_survey_response(
                        persona     = persona,
                        ad          = ad,
                        scales      = selected_scales,
                        api_key     = api_key,
                        rag_context = rag_context,
                        model       = model,
                        api_url     = api_url,
                    )
                    ctx.set_tokens(prompt=700, completion=250)

                    is_valid, issues = validate_response(response, scale_keys)

                    # Regenerate once if bias detected
                    if needs_regeneration(issues):
                        regen_count += 1
                        log_event("response_regenerated", persona["persona_id"], {"issues": issues})
                        response = generate_survey_response(
                            persona, ad, selected_scales, api_key, rag_context, model, api_url
                        )
                        is_valid, issues = validate_response(response, scale_keys)

                    # Outlier check (need ≥ 10 responses first)
                    if len(responses) >= 10 and issues:
                        cohort = compute_cohort_stats(responses, scale_keys)
                        outlier, reasons = is_outlier(response, cohort)
                        if outlier:
                            hitl_flags.append(
                                build_hitl_flag(persona["persona_id"], ad_id,
                                                issues, True, reasons)
                            )

                    responses.append(response)
                    ctx.set_output(
                        " ".join(f"{k}={response.get(k,{}).get('mean','?')}"
                                 for k in scale_keys)
                    )

                except Exception as e:
                    response = neutral_response(persona["persona_id"], ad_id, scale_keys, str(e))
                    responses.append(response)
                    hitl_flags.append(
                        build_hitl_flag(persona["persona_id"], ad_id,
                                        ["GENERATION_FAILED"], False, [str(e)])
                    )
                    ctx.set_error(str(e))

            time.sleep(0.05)

        # Save responses CSV for this ad
        csv_path = os.path.join(output_dir, f"responses_{ad_id}.csv")
        export_responses_csv(responses, scale_keys, csv_path)
        print(f"   CSV saved: {csv_path}")

        # Cohort stats for this ad
        cohort = compute_cohort_stats(responses, scale_keys)

        all_results["all_responses"].extend(responses)
        all_results["hitl_flags"].extend(hitl_flags)
        all_results["ads_processed"].append({
            "ad_id":       ad_id,
            "n_responses": len(responses),
            "n_flags":     len(hitl_flags),
            "n_regens":    regen_count,
            "cohort_stats": cohort,
            "csv_path":    csv_path,
        })

        log_event("ad_complete", ad_id, {
            "n_responses": len(responses),
            "n_flags":     len(hitl_flags),
        })

    # Overall cohort stats across all ads
    all_results["cohort_stats"] = compute_cohort_stats(
        all_results["all_responses"], scale_keys
    )

    # ── STEP 4: Risk Audit ─────────────────────────────────────────────────────
    progress(4, total_steps, "Running risk audit...")

    total_flags = len(all_results["hitl_flags"])
    total_resp  = len(all_results["all_responses"])
    flag_rate   = round(total_flags / total_resp * 100, 1) if total_resp else 0
    print(f"   Flagged: {total_flags}/{total_resp} ({flag_rate}%)")
    log_event("audit_complete", "risk_audit",
              {"n_flags": total_flags, "flag_rate_pct": flag_rate})

    # ── STEP 5: Analytics ──────────────────────────────────────────────────────
    progress(5, total_steps, f"Running analytics (Mode {mode})...")

    ai_stats = all_results["cohort_stats"]

    if mode == "B" and human_responses:
        human_stats = compute_cohort_stats(human_responses, scale_keys)
        comparison  = compare_ai_human(ai_stats, human_stats, scale_keys)
        all_results["comparison"]   = comparison
        all_results["human_stats"]  = human_stats
        print(f"   Verdict: {comparison.get('overall_verdict','')}")
        log_event("comparison_complete", "analytics",
                  {"verdict": comparison.get("overall_verdict", "")})

        # Save comparison CSV
        comp_path = os.path.join(output_dir, "ai_human_comparison.csv")
        _save_comparison_csv(comparison, scale_keys, comp_path)
        all_results["comparison_csv"] = comp_path
    else:
        print(f"   Mode A — descriptive stats only")
        for key, stats in ai_stats.items():
            label = selected_scales[key]["label"]
            print(f"   {label}: M={stats['mean']:.2f}  SD={stats['sd']:.2f}")

    # ── Run metadata ───────────────────────────────────────────────────────────
    elapsed = round((datetime.now() - start_time).total_seconds(), 1)
    all_results["run_metadata"] = {
        "mode":          mode,
        "n_ads":         len(ads),
        "n_personas":    n_personas,
        "n_total_resp":  len(all_results["all_responses"]),
        "n_total_flags": len(all_results["hitl_flags"]),
        "scales":        scale_keys,
        "elapsed_s":     elapsed,
        "timestamp":     start_time.isoformat(),
    }

    # Save full results JSON
    json_path = os.path.join(output_dir, "results_full.json")
    save_data  = {k: v for k, v in all_results.items() if k != "all_responses"}
    save_data["n_responses_saved_to_csv"] = len(all_results["all_responses"])
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n[Done] Mode {mode} | {len(ads)} ads | {len(personas)} personas | {elapsed}s")
    flush()
    return all_results


def _save_comparison_csv(comparison: Dict, scale_keys: List, path: str):
    import csv
    rows = []
    for key in scale_keys:
        c = comparison.get("construct_comparison", {}).get(key, {})
        rows.append({
            "construct":    key,
            "ai_mean":      c.get("ai_mean", ""),
            "ai_sd":        c.get("ai_sd", ""),
            "human_mean":   c.get("human_mean", ""),
            "human_sd":     c.get("human_sd", ""),
            "difference":   c.get("difference", ""),
            "cohens_d":     c.get("cohens_d", ""),
            "equivalent":   c.get("equivalent", ""),
            "alignment":    c.get("alignment", ""),
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# Type alias for import
from typing import List
