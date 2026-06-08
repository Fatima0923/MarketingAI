# tasks.py
import json
from crewai import Task


def task_preprocess(agent, ad: dict, rag_context: str, tools: list) -> Task:
    return Task(
        description=(
            f"Process this advertising stimulus and produce a structured characterisation.\n\n"
            f"AD:\n{json.dumps(ad, indent=2)}\n\n"
            f"RETRIEVED BRAND CONTEXT (from uploaded documents):\n"
            f"{rag_context if rag_context else 'No brand documents uploaded.'}\n\n"
            f"Produce:\n"
            f"1. Primary message claim (one sentence)\n"
            f"2. Message appeal classification (informational/emotional/mixed) with evidence\n"
            f"3. Brand-ad fit assessment: does the ad align with the brand context above?\n"
            f"4. Potential confounds or ambiguities affecting consumer responses\n"
            f"5. Which constructs this ad is likely to elicit strong responses on, and why\n"
        ),
        expected_output=(
            "JSON with keys: ad_id, primary_claim, appeal_classification, "
            "brand_fit_assessment, potential_confounds, construct_sensitivity_notes"
        ),
        agent=agent, tools=tools,
    )


def task_generate_personas(agent, persona_config: dict, tools: list) -> Task:
    return Task(
        description=(
            f"Generate synthetic consumer personas using these researcher-specified settings:\n\n"
            f"n_personas:   {persona_config['n_personas']}\n"
            f"age_min:      {persona_config['age_min']}\n"
            f"age_max:      {persona_config['age_max']}\n"
            f"bf_variance:  {persona_config['bf_variance']} (higher = more personality diversity)\n"
            f"genders:      {persona_config.get('genders', ['Male','Female','Non-binary'])}\n"
            f"income_levels: {persona_config.get('income_levels', ['Low','Middle','High'])}\n"
            f"seed:         {persona_config.get('seed', 42)}\n\n"
            f"Use the persona_generator tool with these exact parameters.\n"
            f"Verify the output shows diversity across age bands and Big Five scores.\n"
            f"Report any stratification concerns.\n"
        ),
        expected_output=(
            f"JSON with n_generated={persona_config['n_personas']}, "
            f"stratification summary, and full personas array."
        ),
        agent=agent, tools=tools,
    )


def task_generate_responses(agent, ad: dict, scales: dict,
                             n_personas: int, tools: list) -> Task:
    scale_names = ", ".join(m["label"] for m in scales.values())
    return Task(
        description=(
            f"Generate synthetic survey responses for ad '{ad.get('ad_id','unknown')}' "
            f"({ad.get('brand','?')} — {ad.get('message_appeal','?')}).\n\n"
            f"Use the survey_response_generator tool for each persona.\n\n"
            f"Constructs to measure: {scale_names}\n\n"
            f"Requirements:\n"
            f"- Every persona must produce DIFFERENT ratings\n"
            f"- Ratings must authentically reflect Big Five profile and demographics\n"
            f"- Include a rationale per construct\n"
            f"- Process all {n_personas} personas\n"
        ),
        expected_output=(
            f"JSON array of {n_personas} response objects each containing "
            f"persona_id, ad_id, and construct scores with items, mean, rationale."
        ),
        agent=agent, tools=tools,
    )


def task_audit(agent, responses_summary: str, scale_keys: list) -> Task:
    return Task(
        description=(
            f"Audit this synthetic response dataset for data quality:\n\n"
            f"{responses_summary}\n\n"
            f"Constructs: {scale_keys}\n\n"
            f"Check for:\n"
            f"1. Acquiescence bias (all-same item ratings)\n"
            f"2. Statistical outliers (> 2 SD from cohort mean per construct)\n"
            f"3. Missing construct data\n"
            f"4. Out-of-range values (items outside 1-7)\n\n"
            f"For each flagged response: state persona_id, issue, recommendation "
            f"(REGENERATE / HUMAN_REVIEW / ACCEPT_WITH_NOTE).\n\n"
            f"Provide overall quality verdict (PASS / REVIEW / FAIL).\n"
        ),
        expected_output=(
            "JSON audit report: total_responses, flagged_count, flag_rate_pct, "
            "cohort_stats (mean/SD per construct), flagged_responses list, "
            "overall_quality_verdict."
        ),
        agent=agent,
    )


def task_analytics(agent, synthetic_stats: str, human_summary: str,
                   scale_keys: list, mode: str) -> Task:
    if mode == "B":
        comparison_instruction = (
            f"\nHUMAN DATA AVAILABLE — run full four-level comparison:\n"
            f"Level 1: Descriptive comparison (mean, SD, difference per construct)\n"
            f"Level 2: Equivalence check (practical equivalence within ±0.5 on 7-pt scale)\n"
            f"Level 3: Effect size (Cohen's d per construct)\n"
            f"Level 4: Overall alignment verdict and research implications\n\n"
            f"Answer RQ1: Do AI responses align with human responses?\n"
            f"Answer RQ2: Does alignment vary by stimuli type or message appeal?\n"
            f"Answer RQ3: Is AI pretesting a reliable screening mechanism?\n"
        )
        expected = (
            "JSON report: descriptive_comparison, equivalence_results, "
            "cohens_d, overall_verdict, rq1_finding, rq2_finding, rq3_finding, "
            "limitations, visualisation_specs."
        )
    else:
        comparison_instruction = (
            f"\nMODE A (synthetic only) — report descriptive statistics only.\n"
            f"Describe the distribution of responses across constructs.\n"
            f"Identify which constructs show most / least variability.\n"
            f"Note any demographic or personality patterns in the responses.\n"
        )
        expected = (
            "JSON report: cohort_stats per construct, distribution_description, "
            "variability_notes, demographic_patterns, visualisation_specs."
        )

    return Task(
        description=(
            f"Perform statistical analysis of synthetic responses.\n\n"
            f"SYNTHETIC DATA:\n{synthetic_stats}\n\n"
            f"HUMAN DATA:\n{human_summary}\n\n"
            f"Constructs: {scale_keys}\n"
            f"{comparison_instruction}"
        ),
        expected_output=expected,
        agent=agent,
    )
