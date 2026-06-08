# app.py
# Gradio UI — all pipeline inputs collected here, nothing hardcoded.
#
# Tab 1: Study Setup     — API key, model, constructs, persona settings
# Tab 2: Ad Inputs       — upload ad brief + enter/upload ad sets
# Tab 3: Run Pipeline    — run button, live log, progress
# Tab 4: Results         — CSV download, visualisations, HITL review
# Tab 5: Comparison      — AI vs human analysis (Mode B, appears when human data uploaded)
# Tab 6: About / MCP     — methodology, MCP discussion

import csv, io, json, os, tempfile, statistics
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv
load_dotenv()

from config import DEFAULT_SCALES, OUTPUTS_DIR, PROVIDERS, PROVIDER_KEYS
from tools.tools import extract_text_from_file, generate_personas
from crew import run_pipeline
from monitoring.langfuse_config import init_langfuse

os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── Session state ─────────────────────────────────────────────────────────────
_session: Dict = {
    "results":      None,
    "ads":          [],
    "brief_docs":   {},
    "human_data":   None,
    "run_complete": False,
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def parse_human_csv(file_path: str, scale_keys: List[str]) -> Tuple[Optional[List], str]:
    """
    Parse uploaded human survey CSV.
    Expected columns: one mean column per construct (e.g. brand_attitude_mean)
    OR item columns (brand_attitude_item1, _item2, _item3).
    Returns (list_of_response_dicts, status_message).
    """
    try:
        responses = []
        with open(file_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = {}
                for key in scale_keys:
                    mean_col  = f"{key}_mean"
                    item_cols = [f"{key}_item{j+1}" for j in range(3)]
                    if mean_col in row:
                        mean = float(row[mean_col])
                        r[key] = {"items": [mean, mean, mean], "mean": mean, "rationale": "human"}
                    elif all(c in row for c in item_cols):
                        items = [float(row[c]) for c in item_cols]
                        r[key] = {"items": items, "mean": round(sum(items)/len(items),3),
                                  "rationale": "human"}
                    else:
                        continue
                if r:
                    responses.append(r)
        return responses, f"Loaded {len(responses)} human responses"
    except Exception as e:
        return None, f"Error parsing CSV: {e}"


def ad_from_manual(brand, product, stimuli_type, appeal,
                   headline, body, tagline, image_desc, positioning) -> Dict:
    return {
        "ad_id":             f"{brand.replace(' ','_')}_{appeal}_{stimuli_type}",
        "brand":             brand,
        "product_category":  product,
        "stimuli_type":      stimuli_type,
        "message_appeal":    appeal,
        "headline":          headline,
        "body":              body,
        "tagline":           tagline,
        "image_description": image_desc,
        "brand_positioning": positioning,
    }


def make_results_html(results: Dict, scale_labels: Dict) -> str:
    if not results:
        return "<p>No results yet.</p>"

    ads_info = results.get("ads_processed", [])
    lines    = ["<h3>Run Summary</h3>"]
    meta     = results.get("run_metadata", {})
    lines.append(f"<p><b>Mode {meta.get('mode','?')}</b> — "
                 f"{meta.get('n_ads','?')} ad(s), "
                 f"{meta.get('n_personas','?')} personas each, "
                 f"{meta.get('elapsed_s','?')}s</p>")

    for ad_info in ads_info:
        lines.append(f"<h4>Ad: {ad_info['ad_id']}</h4>")
        lines.append(f"<p>{ad_info['n_responses']} responses | "
                     f"{ad_info['n_flags']} flagged | "
                     f"CSV: <code>{ad_info['csv_path']}</code></p>")
        cohort = ad_info.get("cohort_stats", {})
        rows   = "".join(
            f"<tr><td>{scale_labels.get(k,k)}</td>"
            f"<td>{s.get('mean',''):.2f}</td>"
            f"<td>{s.get('sd',''):.2f}</td>"
            f"<td>{s.get('n','')}</td></tr>"
            for k, s in cohort.items()
        )
        lines.append(
            "<table style='border-collapse:collapse;font-size:12px;width:100%'>"
            "<thead><tr style='background:var(--color-background-info);color:var(--color-text-info)'>"
            "<th style='padding:4px 8px'>Construct</th><th>Mean</th><th>SD</th><th>N</th>"
            "</tr></thead><tbody>"
            + rows + "</tbody></table>"
        )

    return "\n".join(lines)


def make_comparison_html(results: Dict, scale_labels: Dict) -> str:
    comp = results.get("comparison")
    if not comp:
        return "<p>No human data provided — Mode A only.</p>"

    lines  = [f"<h3>AI vs Human Comparison</h3>"]
    lines.append(f"<p><b>Overall verdict:</b> {comp.get('overall_verdict','')}</p>")

    cc = comp.get("construct_comparison", {})
    rows = ""
    for key, c in cc.items():
        align_color = (
            "var(--color-text-success)" if c["alignment"] == "Strong" else
            "var(--color-text-warning)" if c["alignment"] == "Moderate" else
            "var(--color-text-danger)"
        )
        rows += (
            f"<tr>"
            f"<td>{scale_labels.get(key,key)}</td>"
            f"<td>{c.get('ai_mean',''):.2f} ({c.get('ai_sd',''):.2f})</td>"
            f"<td>{c.get('human_mean',''):.2f} ({c.get('human_sd',''):.2f})</td>"
            f"<td>{c.get('difference',''):+.3f}</td>"
            f"<td>{c.get('cohens_d',''):.3f}</td>"
            f"<td>{'Yes' if c.get('equivalent') else 'No'}</td>"
            f"<td style='color:{align_color}'>{c.get('alignment','')}</td>"
            f"</tr>"
        )

    lines.append(
        "<table style='border-collapse:collapse;font-size:12px;width:100%'>"
        "<thead><tr style='background:var(--color-background-info);color:var(--color-text-info)'>"
        "<th style='padding:4px 8px'>Construct</th>"
        "<th>AI M(SD)</th><th>Human M(SD)</th>"
        "<th>Diff</th><th>Cohen's d</th><th>Equiv.</th><th>Alignment</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
    )
    return "\n".join(lines)


def make_bar_chart_data(results: Dict, scale_labels: Dict) -> Optional[object]:
    """Build a simple matplotlib bar chart comparing AI and human means."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import tempfile

        ai_stats     = results.get("cohort_stats", {})
        human_stats  = results.get("human_stats", {})
        scale_keys   = list(ai_stats.keys())
        labels       = [scale_labels.get(k, k) for k in scale_keys]
        ai_means     = [ai_stats[k]["mean"] for k in scale_keys]
        ai_sds       = [ai_stats[k]["sd"]   for k in scale_keys]

        x     = range(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([i - width/2 for i in x], ai_means, width,
               label="AI Synthetic", color="#4472C4", alpha=0.85,
               yerr=ai_sds, capsize=4)

        if human_stats:
            hu_means = [human_stats.get(k, {}).get("mean", 0) for k in scale_keys]
            hu_sds   = [human_stats.get(k, {}).get("sd", 0)   for k in scale_keys]
            ax.bar([i + width/2 for i in x], hu_means, width,
                   label="Human", color="#ED7D31", alpha=0.85,
                   yerr=hu_sds, capsize=4)

        ax.set_ylabel("Mean Score (1-7 scale)")
        ax.set_title("Construct Means: AI Synthetic vs Human Responses")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(1, 7.5)
        ax.axhline(4.0, color="gray", linestyle="--", linewidth=0.8, label="Scale midpoint")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        plt.savefig(tmp.name, dpi=150)
        plt.close()
        return tmp.name
    except Exception as e:
        print(f"[Chart] Error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

with gr.Blocks(
    title="AI Advertising Pretest Pipeline",
    theme=gr.themes.Soft(primary_hue="blue", font=gr.themes.GoogleFont("Inter")),
) as demo:

    gr.Markdown("""
# 🎯 AI Advertising Pretest Pipeline
### CrewAI · LangGraph · FAISS RAG · Human-in-the-Loop · Langfuse
*Validating synthetic AI responses as proxies for human survey responses in advertising pretesting*
    """)

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════════
        # TAB 1: STUDY SETUP
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("⚙️ Study Setup"):
            gr.Markdown("### API & Model Configuration")
            with gr.Row():
                provider_selector = gr.Dropdown(
                    choices=list(PROVIDERS.keys()),
                    value="DeepSeek",
                    label="LLM Provider",
                    scale=1,
                )
                api_key_input = gr.Textbox(
                    label="API Key",
                    placeholder="Enter your API key...",
                    type="password",
                    value=PROVIDER_KEYS.get("DeepSeek", ""),
                    scale=3,
                )
            with gr.Row():
                model_input = gr.Textbox(
                    label="Model name",
                    value=PROVIDERS["DeepSeek"]["default_model"],
                    scale=2,
                )
                provider_note = gr.Textbox(
                    label="Available models",
                    value=PROVIDERS["DeepSeek"]["note"],
                    interactive=False,
                    scale=2,
                )

            def on_provider_change(provider):
                p = PROVIDERS.get(provider, PROVIDERS["DeepSeek"])
                return (
                    PROVIDER_KEYS.get(provider, ""),
                    p["default_model"],
                    p["note"],
                )

            provider_selector.change(
                fn=on_provider_change,
                inputs=[provider_selector],
                outputs=[api_key_input, model_input, provider_note],
            )

            gr.Markdown("### Measurement Constructs")
            gr.Markdown("Select which constructs to measure. All use validated 3-item 7-point Likert scales.")
            with gr.Row():
                scale_ba  = gr.Checkbox(label="Brand Attitude (MacKenzie & Lutz, 1989)",       value=True)
                scale_pi  = gr.Checkbox(label="Purchase Intention (Dodds et al., 1991)",        value=True)
                scale_bf  = gr.Checkbox(label="Brand Fit (Becker-Olsen et al., 2006)",          value=True)
                scale_ac  = gr.Checkbox(label="Ad Credibility (Newell & Goldsmith, 2001)",      value=False)

            gr.Markdown("### Persona Settings")
            gr.Markdown(
                "Personas are generated randomly within these ranges — "
                "like drawing a stratified sample of real respondents."
            )
            with gr.Row():
                age_min_slider = gr.Slider(18, 60, value=18, step=1,  label="Age minimum")
                age_max_slider = gr.Slider(25, 75, value=65, step=1,  label="Age maximum")
                n_personas_slider = gr.Slider(5, 200, value=50, step=5, label="Number of personas")
            with gr.Row():
                bf_var_slider  = gr.Slider(0.05, 0.40, value=0.20, step=0.05,
                                           label="Personality variance (Big Five SD — higher = more diversity)")
                seed_input     = gr.Number(value=42, label="Random seed (for reproducibility)", precision=0)
            with gr.Row():
                gender_check = gr.CheckboxGroup(
                    ["Male", "Female", "Non-binary"],
                    value=["Male", "Female", "Non-binary"],
                    label="Genders to include",
                )
                income_check = gr.CheckboxGroup(
                    ["Low", "Middle", "High"],
                    value=["Low", "Middle", "High"],
                    label="Income levels to include",
                )

            gr.Markdown("### Langfuse Observability (optional)")
            with gr.Row():
                lf_public = gr.Textbox(label="Langfuse Public Key",  type="password",
                                       value=os.getenv("LANGFUSE_PUBLIC_KEY",""), scale=2)
                lf_secret = gr.Textbox(label="Langfuse Secret Key",  type="password",
                                       value=os.getenv("LANGFUSE_SECRET_KEY",""), scale=2)
                lf_host   = gr.Textbox(label="Langfuse Host",
                                       value=os.getenv("LANGFUSE_HOST","https://cloud.langfuse.com"),
                                       scale=2)

            save_setup_btn = gr.Button("Save Setup", variant="secondary")
            setup_status   = gr.Textbox(label="Setup Status", interactive=False, lines=2)

            def save_setup(provider, api_key, model, ba, pi, bf, ac,
                           age_min, age_max, n_p, bfv, seed,
                           genders, incomes, lf_pub, lf_sec, lf_h):
                if not api_key:
                    return "⚠️  API key is required."
                selected = {}
                if ba: selected["brand_attitude"]    = DEFAULT_SCALES["brand_attitude"]
                if pi: selected["purchase_intention"] = DEFAULT_SCALES["purchase_intention"]
                if bf: selected["brand_fit"]          = DEFAULT_SCALES["brand_fit"]
                if ac: selected["ad_credibility"]     = DEFAULT_SCALES["ad_credibility"]
                if not selected:
                    return "⚠️  Select at least one construct."
                if not genders:
                    return "⚠️  Select at least one gender."
                if not incomes:
                    return "⚠️  Select at least one income level."

                # Init Langfuse if keys provided
                if lf_pub and lf_sec:
                    init_langfuse(lf_pub, lf_sec, lf_h)

                p = PROVIDERS.get(provider, PROVIDERS["DeepSeek"])
                _session["setup"] = {
                    "api_key":  api_key,
                    "model":    model,
                    "provider": provider,
                    "api_url":  p["api_url"],
                    "base_url": p["base_url"],
                    "scales":   selected,
                    "persona_config": {
                        "n_personas":   int(n_p),
                        "age_min":      int(age_min),
                        "age_max":      int(age_max),
                        "bf_variance":  float(bfv),
                        "seed":         int(seed),
                        "genders":      genders,
                        "income_levels": incomes,
                    },
                }
                constructs = ", ".join(v["label"] for v in selected.values())
                return (
                    f"✅  Setup saved.\n"
                    f"Provider:   {provider} ({model})\n"
                    f"Constructs: {constructs}\n"
                    f"Personas:   {int(n_p)}, age {int(age_min)}-{int(age_max)}, "
                    f"BF variance={float(bfv)}"
                )

            save_setup_btn.click(
                fn=save_setup,
                inputs=[provider_selector, api_key_input, model_input,
                        scale_ba, scale_pi, scale_bf, scale_ac,
                        age_min_slider, age_max_slider, n_personas_slider,
                        bf_var_slider, seed_input,
                        gender_check, income_check,
                        lf_public, lf_secret, lf_host],
                outputs=[setup_status],
            )

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2: AD INPUTS
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("📄 Ad Inputs"):

            gr.Markdown("### Ad Concept Brief (optional — uploaded documents are indexed into FAISS for RAG)")
            gr.Markdown(
                "Upload brand briefs, campaign briefs, positioning documents, "
                "or audience profiles. These are chunked, embedded, and retrieved "
                "by agents during evaluation to ground responses in actual brand knowledge."
            )
            brief_upload = gr.File(
                label="Upload brand/campaign brief documents (PDF, DOCX, TXT)",
                file_types=[".pdf", ".docx", ".txt", ".md"],
                file_count="multiple",
            )
            brief_status = gr.Textbox(label="Brief Status", interactive=False, lines=2)

            def process_briefs(files):
                if not files:
                    _session["brief_docs"] = {}
                    return "No brief documents uploaded — RAG will not be used"
                docs = {}
                for f in files:
                    text = extract_text_from_file(f.name)
                    name = os.path.splitext(os.path.basename(f.name))[0]
                    docs[name] = text
                    print(f"   Brief: {name} ({len(text)} chars)")
                _session["brief_docs"] = docs
                names = list(docs.keys())
                return (
                    f"✅  {len(docs)} brief document(s) loaded: {', '.join(names)}\n"
                    f"These will be indexed into FAISS at pipeline start."
                )

            brief_upload.change(fn=process_briefs, inputs=[brief_upload], outputs=[brief_status])

            gr.Markdown("### Ad Sets")
            gr.Markdown(
                "Add each ad manually below, or upload a JSON file. "
                "Each ad will be evaluated by all personas."
            )

            with gr.Accordion("Add ad manually", open=True):
                with gr.Row():
                    ad_brand       = gr.Textbox(label="Brand name",         placeholder="e.g. AquaPure")
                    ad_product     = gr.Textbox(label="Product category",    placeholder="e.g. Water Purifier")
                    ad_type        = gr.Dropdown(["text", "image", "video"], label="Stimuli type", value="text")
                    ad_appeal      = gr.Dropdown(["informational", "emotional", "mixed"],
                                                 label="Message appeal", value="informational")
                with gr.Row():
                    ad_headline = gr.Textbox(label="Headline",    placeholder="e.g. Clinically Proven. 99.9% Pure.")
                    ad_tagline  = gr.Textbox(label="Tagline",     placeholder="e.g. Science you can drink.")
                ad_body        = gr.Textbox(label="Body copy", lines=3,
                                             placeholder="Main ad copy text...")
                ad_image_desc  = gr.Textbox(label="Image description (if image ad)",
                                             placeholder="Describe the visual content...", lines=2)
                ad_positioning = gr.Textbox(label="Brand positioning (if no brief uploaded)",
                                             placeholder="Who is this brand for? What does it stand for?",
                                             lines=2)
                add_ad_btn     = gr.Button("+ Add this ad", variant="secondary")
                ads_status     = gr.Textbox(label="Ad Queue", interactive=False, lines=4)

            with gr.Accordion("Or upload ad sets as JSON", open=False):
                gr.Markdown("""
Each object in the JSON array must have at minimum: `brand`, `headline`, `body`.
Optional: `ad_id`, `product_category`, `stimuli_type`, `message_appeal`, `tagline`,
`image_description`, `brand_positioning`.
                """)
                json_upload = gr.File(label="Upload ads JSON file", file_types=[".json"])
                json_status = gr.Textbox(label="JSON Status", interactive=False)

            clear_ads_btn = gr.Button("Clear all ads", variant="stop", size="sm")

            def add_ad(brand, product, ad_type, appeal, headline,
                       tagline, body, image_desc, positioning):
                if not brand or not headline:
                    return "⚠️  Brand name and headline are required."
                ad = {
                    "ad_id":             f"{brand.replace(' ','_')}_{appeal}_{ad_type}",
                    "brand":             brand,
                    "product_category":  product,
                    "stimuli_type":      ad_type,
                    "message_appeal":    appeal,
                    "headline":          headline,
                    "body":              body,
                    "tagline":           tagline,
                    "image_description": image_desc,
                    "brand_positioning": positioning,
                }
                _session["ads"].append(ad)
                lines = [f"  {i+1}. [{a['ad_id']}] {a['brand']} — {a['message_appeal']}"
                         for i, a in enumerate(_session["ads"])]
                return "Ad queue:\n" + "\n".join(lines)

            def load_json_ads(file):
                if not file:
                    return "No file."
                try:
                    with open(file.name) as f:
                        ads = json.load(f)
                    if not isinstance(ads, list):
                        ads = [ads]
                    for ad in ads:
                        if "ad_id" not in ad:
                            ad["ad_id"] = f"{ad.get('brand','ad')}_{ad.get('message_appeal','x')}_{ad.get('stimuli_type','x')}"
                    _session["ads"].extend(ads)
                    return f"✅  Loaded {len(ads)} ad(s) from JSON. Total in queue: {len(_session['ads'])}"
                except Exception as e:
                    return f"Error: {e}"

            def clear_ads():
                _session["ads"] = []
                return "Ad queue cleared."

            add_ad_btn.click(
                fn=add_ad,
                inputs=[ad_brand, ad_product, ad_type, ad_appeal,
                        ad_headline, ad_tagline, ad_body, ad_image_desc, ad_positioning],
                outputs=[ads_status],
            )
            json_upload.change(fn=load_json_ads, inputs=[json_upload], outputs=[json_status])
            clear_ads_btn.click(fn=clear_ads, outputs=[ads_status])

            gr.Markdown("### Human Responses (optional — triggers Mode B)")
            gr.Markdown(
                "Upload a CSV of human survey responses collected in Phase B. "
                "Required columns: `{construct}_mean` or `{construct}_item1/2/3` "
                "for each selected construct."
            )
            human_upload    = gr.File(label="Upload human responses CSV", file_types=[".csv"])
            human_status    = gr.Textbox(label="Human Data Status", interactive=False)

            def process_human(file):
                if not file:
                    _session["human_data"] = None
                    return "No human data — pipeline will run in Mode A (synthetic only)"
                setup = _session.get("setup", {})
                scale_keys = list(setup.get("scales", DEFAULT_SCALES).keys())
                responses, msg = parse_human_csv(file.name, scale_keys)
                _session["human_data"] = responses
                if responses:
                    return f"✅  Mode B activated — {msg}"
                return f"⚠️  {msg}"

            human_upload.change(fn=process_human, inputs=[human_upload], outputs=[human_status])

        # ══════════════════════════════════════════════════════════════════════
        # TAB 3: RUN PIPELINE
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("🚀 Run Pipeline"):

            gr.Markdown("### Pre-flight check")
            check_btn = gr.Button("Check inputs before running", variant="secondary")
            check_out = gr.Textbox(label="Pre-flight status", interactive=False, lines=6)

            def preflight():
                setup = _session.get("setup")
                if not setup:
                    return "⚠️  Setup not saved. Go to Study Setup tab first."
                if not setup.get("api_key"):
                    return "⚠️  API key missing."
                if not _session.get("ads"):
                    return "⚠️  No ads in queue. Go to Ad Inputs tab."
                scales = setup.get("scales", {})
                pc     = setup.get("persona_config", {})
                mode   = "B (AI + Human comparison)" if _session.get("human_data") else "A (synthetic only)"
                return (
                    f"✅  Ready to run\n"
                    f"Provider      : {setup.get('provider','?')} ({setup.get('model','?')})\n"
                    f"Mode          : {mode}\n"
                    f"Ads in queue  : {len(_session['ads'])}\n"
                    f"Constructs    : {', '.join(v['label'] for v in scales.values())}\n"
                    f"Personas      : {pc.get('n_personas','?')}, "
                    f"age {pc.get('age_min','?')}-{pc.get('age_max','?')}\n"
                    f"Brief docs    : {len(_session.get('brief_docs',{}))} document(s) for RAG\n"
                    f"Human data    : {'Yes' if _session.get('human_data') else 'No'}"
                )

            check_btn.click(fn=preflight, outputs=[check_out])

            run_btn  = gr.Button("▶  Run Pipeline", variant="primary", size="lg")
            log_box  = gr.Textbox(label="Pipeline Log", lines=22, interactive=False)
            run_info = gr.Textbox(label="Run Summary", lines=4, interactive=False)

            def run(progress=gr.Progress()):
                setup = _session.get("setup")
                if not setup:
                    yield "Setup not saved.", ""
                    return
                if not _session.get("ads"):
                    yield "No ads in queue.", ""
                    return

                log = []

                def log_yield(msg):
                    log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
                    return "\n".join(log[-40:])

                yield log_yield("Starting pipeline..."), ""

                mode  = "B" if _session.get("human_data") else "A"
                yield log_yield(f"Mode {mode}: {'AI + Human comparison' if mode=='B' else 'Synthetic only'}"), ""
                yield log_yield(f"Ads: {len(_session['ads'])} | Personas: {setup['persona_config']['n_personas']}"), ""

                if _session.get("brief_docs"):
                    yield log_yield(f"Brief docs for RAG: {list(_session['brief_docs'].keys())}"), ""
                else:
                    yield log_yield("No brief documents — RAG disabled for this run"), ""

                def progress_cb(step, total, msg):
                    progress(step / total, desc=msg)

                try:
                    results = run_pipeline(
                        ads             = _session["ads"],
                        selected_scales = setup["scales"],
                        persona_config  = setup["persona_config"],
                        brief_docs      = _session.get("brief_docs", {}),
                        human_responses = _session.get("human_data"),
                        api_key         = setup["api_key"],
                        model           = setup.get("model", "deepseek-chat"),
                        api_url         = setup.get("api_url", "https://api.deepseek.com/v1/chat/completions"),
                        base_url        = setup.get("base_url", "https://api.deepseek.com/v1"),
                        output_dir      = OUTPUTS_DIR,
                        progress_cb     = progress_cb,
                    )
                    _session["results"]      = results
                    _session["run_complete"] = True

                    meta  = results.get("run_metadata", {})
                    yield log_yield(
                        f"Done — Mode {meta.get('mode','?')} | "
                        f"{meta.get('n_total_resp','?')} responses | "
                        f"{meta.get('n_total_flags','?')} flagged | "
                        f"{meta.get('elapsed_s','?')}s"
                    ), (
                        f"Mode {meta.get('mode','?')} complete.\n"
                        f"Total responses: {meta.get('n_total_resp','?')}\n"
                        f"Flagged for HITL review: {meta.get('n_total_flags','?')}\n"
                        f"Outputs saved to: {OUTPUTS_DIR}/"
                    )

                except Exception as e:
                    yield log_yield(f"[ERROR] {e}"), f"Pipeline failed: {e}"

            run_btn.click(fn=run, outputs=[log_box, run_info])

        # ══════════════════════════════════════════════════════════════════════
        # TAB 4: RESULTS
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("📊 Results"):

            refresh_btn  = gr.Button("Refresh Results", variant="secondary")
            results_html = gr.HTML()
            chart_output = gr.Image(label="Construct Means Chart", type="filepath")
            dl_csv_btn   = gr.Button("Download all responses CSV", variant="secondary")
            dl_status    = gr.Textbox(label="Download status", interactive=False)

            gr.Markdown("### HITL Review — Flagged Responses")
            flags_html   = gr.HTML()
            with gr.Row():
                flag_persona = gr.Textbox(label="Persona ID to review")
                flag_action  = gr.Radio(["Accept", "Regenerate", "Exclude"], label="Decision")
                flag_note    = gr.Textbox(label="Reviewer note")
            apply_flag   = gr.Button("Apply Decision", variant="secondary")
            flag_status  = gr.Textbox(label="Status", interactive=False)

            def refresh_results():
                results = _session.get("results")
                if not results:
                    return "<p>No results yet.</p>", None

                setup        = _session.get("setup", {})
                scale_labels = {k: v["label"] for k, v in setup.get("scales", DEFAULT_SCALES).items()}
                html         = make_results_html(results, scale_labels)
                chart        = make_bar_chart_data(results, scale_labels)

                flags     = results.get("hitl_flags", [])
                if flags:
                    rows = "".join(
                        f"<tr><td>{f['persona_id']}</td>"
                        f"<td>{f['ad_id']}</td>"
                        f"<td>{'; '.join(f.get('validation_issues',[]))}</td>"
                        f"<td>{'Yes' if f.get('is_outlier') else 'No'}</td></tr>"
                        for f in flags[:20]
                    )
                    flags_content = (
                        f"<p><b>{len(flags)} response(s) flagged</b></p>"
                        "<table style='font-size:12px;border-collapse:collapse;width:100%'>"
                        "<thead><tr style='background:var(--color-background-danger);color:var(--color-text-danger)'>"
                        "<th style='padding:4px 8px'>Persona</th><th>Ad</th><th>Issues</th><th>Outlier</th>"
                        "</tr></thead><tbody>" + rows + "</tbody></table>"
                    )
                else:
                    flags_content = "<p style='color:var(--color-text-success)'>No responses flagged.</p>"

                return html, chart, flags_content

            def download_csv():
                results = _session.get("results")
                if not results:
                    return "No results yet."
                paths = [ad["csv_path"] for ad in results.get("ads_processed", [])
                         if os.path.exists(ad.get("csv_path",""))]
                if not paths:
                    return "No CSV files found."
                return f"CSV files saved:\n" + "\n".join(paths)

            def apply_hitl(persona_id, action, note):
                if not persona_id:
                    return "Enter a persona ID."
                flags = _session.get("results", {}).get("hitl_flags", [])
                remaining = [f for f in flags if f["persona_id"] != persona_id]
                if _session.get("results"):
                    _session["results"]["hitl_flags"] = remaining
                return f"Decision '{action}' applied to {persona_id}. Note: '{note}'. {len(remaining)} flags remaining."

            refresh_btn.click(fn=refresh_results, outputs=[results_html, chart_output, flags_html])
            dl_csv_btn.click(fn=download_csv, outputs=[dl_status])
            apply_flag.click(fn=apply_hitl, inputs=[flag_persona, flag_action, flag_note], outputs=[flag_status])

        # ══════════════════════════════════════════════════════════════════════
        # TAB 5: AI vs HUMAN COMPARISON (Mode B)
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("🔬 AI vs Human (Mode B)"):

            gr.Markdown("""
### Mode B: Statistical Comparison

This tab is active when human responses have been uploaded.
Shows four-level analysis: descriptive comparison, equivalence (TOST),
effect size (Cohen's d), and alignment verdict per construct.
            """)
            comp_btn  = gr.Button("Show Comparison Results", variant="primary")
            comp_html = gr.HTML()
            comp_txt  = gr.Textbox(label="Full Comparison JSON", lines=12, interactive=False)

            def show_comparison():
                results = _session.get("results")
                if not results:
                    return "<p>Run the pipeline first.</p>", ""
                setup        = _session.get("setup", {})
                scale_labels = {k: v["label"] for k, v in setup.get("scales", DEFAULT_SCALES).items()}
                html = make_comparison_html(results, scale_labels)
                comp = results.get("comparison", {})
                return html, json.dumps(comp, indent=2) if comp else "Mode A — no comparison available"

            comp_btn.click(fn=show_comparison, outputs=[comp_html, comp_txt])

        # ══════════════════════════════════════════════════════════════════════
        # TAB 6: ABOUT & MCP
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("📖 About & MCP"):
            gr.Markdown("""
## Research Context

**RQ1:** To what extent do AI-generated synthetic responses, produced by persona-conditioned LLM agents, converge with human evaluations on established advertising pretest constructs?

**RQ2:** Does alignment vary across stimuli type (text/image) and message appeal (informational/emotional)?

**RQ3:** Can a multi-agent AI system with HITL oversight serve as a reliable and scalable screening mechanism for advertising pretesting?

---

## Pipeline Architecture

```
[INPUTS]
  Ad sets (upload / manual entry)
  Ad concept brief (PDF → FAISS RAG index)
  Persona settings (UI sliders → passed at runtime)
  Human responses (optional CSV → Mode B)
         ↓
[Step 1] Build FAISS RAG store from brief documents
[Step 2] Generate N personas (Big Five + demographics, fully parameterised)
[Step 3] For each ad × each persona:
           → Query FAISS for brand context
           → Respondent Agent generates Likert-scale response
           → Validate → flag outliers → regenerate if biased
[Step 4] Risk Auditor: cohort-level QA, HITL flag queue
[Step 5] Analytics:
           Mode A → descriptive stats + CSV
           Mode B → + AI-human comparison (equivalence, Cohen's d, verdict)
```

---

## Fallback Mechanisms

| Failure Mode | Fallback |
|---|---|
| LLM API failure | Exponential backoff (3s→6s→12s), then neutral midpoint |
| JSON parse failure | Recovery parser → structured default |
| Acquiescence bias | Detect all-same ratings → regenerate once |
| Statistical outlier | Flag for HITL review queue |
| Tool crash | Graceful error + neutral response, pipeline continues |
| Missing construct | Fill with midpoint 4.0, flag automatically |

---

## MCP Discussion

**What external services could be exposed as MCP servers:**

1. **Brand Asset Library** — brand documents, positioning, tone-of-voice guidelines exposed as a retrievable MCP resource. Agents query the latest version without manual upload.

2. **Ad Stimuli Repository** — a versioned library of ad creatives (with metadata) as an MCP server. Agents pull ads by campaign, brand, or stimuli type.

3. **Survey Platform Integration** — Qualtrics or SurveyMonkey API as MCP, enabling real-time Phase B human data ingestion directly into Phase C analytics.

4. **Persona Registry** — curated, validated persona libraries maintained as an MCP server, enabling consistent reuse across studies and brands.

**How MCP would improve this project:**
- Standardised tool integration — any agent connects to any data source via the same protocol
- Version-controlled brand context — agents always use the latest positioning
- Real-time human data — Phase B data flows directly into Phase C analytics
- Scalability across multiple brands without code changes

The `PersonaGeneratorTool` could itself be packaged as an MCP server, exposing a `generate_personas(n, seed, constraints)` endpoint usable by any downstream agent regardless of framework.
            """)


if __name__ == "__main__":
    print("=" * 60)
    print("  AI Advertising Pretest Pipeline")
    print("  CrewAI · FAISS RAG · LangGraph · Langfuse · Gradio")
    print("=" * 60)
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
