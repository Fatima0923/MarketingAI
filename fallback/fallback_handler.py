# fallback/fallback_handler.py
import json, statistics, traceback
from typing import Dict, List, Optional, Tuple
from monitoring.langfuse_config import log_event

OUTLIER_SD_THRESHOLD = 2.0


def validate_response(response: Dict, scale_keys: List[str]) -> Tuple[bool, List[str]]:
    """Validate one synthetic response. Returns (is_valid, issues)."""
    issues = []
    for key in scale_keys:
        if key not in response:
            issues.append(f"MISSING_CONSTRUCT:{key}"); continue
        items = response[key].get("items", [])
        if not all(isinstance(x, (int, float)) and 1 <= x <= 7 for x in items):
            issues.append(f"OUT_OF_RANGE:{key}")
        if len(items) != 3:
            issues.append(f"WRONG_ITEM_COUNT:{key}(got {len(items)})")
        if len(items) == 3 and len(set(items)) == 1:
            issues.append(f"ACQUIESCENCE_BIAS:{key}(all={items[0]})")
        if items:
            expected = sum(items) / len(items)
            if abs(expected - response[key].get("mean", 0)) > 0.1:
                response[key]["mean"] = round(expected, 3)
    fatal = [i for i in issues if i.startswith(("MISSING", "OUT_OF_RANGE"))]
    return len(fatal) == 0, issues


def is_outlier(response: Dict, cohort_stats: Dict) -> Tuple[bool, List[str]]:
    """Check if response is a statistical outlier vs cohort."""
    reasons = []
    for key, stats in cohort_stats.items():
        if key not in response: continue
        pm = response[key].get("mean", 4.0)
        cm = stats.get("mean", 4.0)
        sd = stats.get("sd", 1.0)
        if sd == 0: continue
        z = abs(pm - cm) / sd
        if z > OUTLIER_SD_THRESHOLD:
            reasons.append(f"{key}: z={z:.2f} (persona={pm:.2f}, cohort={cm:.2f})")
    return len(reasons) > 0, reasons


def compute_cohort_stats(responses: List[Dict], scale_keys: List[str]) -> Dict:
    stats = {}
    for key in scale_keys:
        means = [r[key]["mean"] for r in responses
                 if key in r and isinstance(r[key].get("mean"), (int, float))]
        if len(means) >= 2:
            stats[key] = {"mean": round(statistics.mean(means), 3),
                          "sd":   round(statistics.stdev(means), 3), "n": len(means)}
        elif means:
            stats[key] = {"mean": means[0], "sd": 1.0, "n": 1}
        else:
            stats[key] = {"mean": 4.0, "sd": 1.0, "n": 0}
    return stats


def neutral_response(persona_id: str, ad_id: str,
                     scale_keys: List[str], reason: str = "fallback") -> Dict:
    log_event("fallback", "neutral_response",
              {"persona_id": persona_id, "ad_id": ad_id, "reason": reason})
    r = {"persona_id": persona_id, "ad_id": ad_id, "fallback": True,
         "fallback_reason": reason}
    for key in scale_keys:
        r[key] = {"items": [4, 4, 4], "mean": 4.0,
                  "rationale": f"[Neutral fallback — {reason}]"}
    return r


def build_hitl_flag(persona_id, ad_id, issues, outlier_flag, outlier_reasons) -> Dict:
    return {"persona_id": persona_id, "ad_id": ad_id,
            "requires_review": True, "validation_issues": issues,
            "is_outlier": outlier_flag, "outlier_reasons": outlier_reasons,
            "recommended_action": "Review and optionally regenerate"}


def handle_tool_error(tool_name: str, error: Exception, fallback=None):
    tb = traceback.format_exc()
    log_event("tool_error", tool_name, {"error": str(error), "tb": tb[:400]}, error=str(error))
    print(f"[FALLBACK] Tool '{tool_name}' failed: {error}")
    return fallback


def needs_regeneration(issues: List[str]) -> bool:
    return any(k in i for i in issues
               for k in ("ACQUIESCENCE_BIAS", "OUT_OF_RANGE", "MISSING_CONSTRUCT"))
