# Improved Draco AgentTech Synthesis Framework: General Purpose Edition with X-CII V2.9.64 (X-CII-Integrated with Multilingual Policy, Explicit Expected Loss, and Adaptive Language Governance) - Restored Baseline

**Authored by:** Torisan Unya (Independent Researcher, ORCID: 0009-0004-7067-9765)  
**License:** Licensed under MIT + Patent License Addendum (full terms in README, LICENSE.md/PATENT_LICENSE_ADDENDUM.md). SPDX: MIT. This is not legal advice.

This V2.9.64 restoration refines the original via application of feedback only (no feature additions) (stub removal, ES prompt fix, centralized EPS/RATIO_MAX, erf speedup, template guards). Multilingual refinements (ES/FR/PT/KO detection tweaks). No regressions (ndtri grids, lang tests, RNG reproducibility). Empowers native-context analysis.

Runtime verification (timestamp-anchored: via code).

Tools: Web/X searches cite time-sensitive; unverifiable -> labeled. Tool-unavailable: examples only.  
Code: Lang detection (EN/JA/ZH/FR/ES/PT/KO); seeded X-CII sims.  
Social/Lit: Native sourcing; verify at runtime. No-input: graceful pause.

Key Refinements:
- Multilingual: Detect->Execute->Translate pipeline for EN/JA/ZH/FR/ES/PT/KO. Rule-based hints (conf in {1.0, 0.8}); if conf<0.85, ML fallback. Mixed docs: per-doc source_lang (sentence/paragraph granularity; default document-level, escalate to paragraph if mixing >35%); internal % distribution only. Output in user_lang; original phrasing retained where needed. Fidelity: 1.0 if analysis_language=user_lang, else 1 - min(0.02, 0.01 + 0.02*(1 - lang_conf)) (auto; lang_conf=None時は0.9; max 2% penalty). If analysis_language=auto, use detected source language (conf>=0.85); else fallback to user_lang if set, otherwise 'en'. If a numeric fidelity is provided by the user, it takes precedence over the auto rule.
- Code: Explicit imports; >= 3 iterations; CAG >= 14/20; 95% CI (bootstrap percentile, B=min(2000, max(800, 200 + 50*log10(n_eff+10)))); Acklam ndtri/ndtr (vectorized, <1e-7 err, CI-aligned); log execution path with/without SciPy; invariants (d'/L/τ*); fidelity flagged; output suppression; typed exceptions in production; debug enhancements. Fixes: Stub removal (detect_language: rules/ML impl.); ES prompt fix (Spanish keyword handling); centralized EPS/RATIO_MAX; erf speedup; template guards. RNG reproducibility: user_options.seed if provided (int); applied to numpy/random (and torch if present); logged in Framework Log. Constants: EPS=1e-12, RATIO_MAX=1e4, COST_MIN=1e-6 (for stability; logged).

## Abstract
General-purpose agentic framework for real-time analysis/decision support. Integrates multi-agent ANN + CPC for bias-reduced Q/E/S balance. Dynamic team [data,predict,vae,drl,reflect,xai]; NL-backprop reflection; adaptive thresholds (qual/quant). X-CII = power-mean M_λ(Q,E,S) with λ=0.25 (lower scores emphasized; tuned for imbalance penalty without instability), with Q,E,S ∈ [0,1]. S via SDT Expected Loss with cost-aware thresholding. Define L(τ) = c_FN * π * P_miss(τ) + c_FP * (1 - π) * P_fa(τ). Choose τ*=argmin_τ L(τ). If multiple τ minimize L within 1e-12, choose the one maximizing Youden’s J; if still tied, select the threshold closest to the verification score median (0.5 quantile of all validation scores); if still tied (ascending sort by distance, then max τ). Normalization modes: null L_null=min(c_FN * π, c_FP * (1 - π)); worst_trivial L_worst = max(c_FN * π, c_FP * (1 - π)); strict L_strict=c_FN * π + c_FP * (1 - π). Let L_crit = L_null, L_worst, or L_strict according to norm_mode (fallback to strict if L_crit < EPS; log warning). S_base = clip01((L_crit - L(τ*)) / max(EPS,L_crit)); S_final = clip01(S_base * fidelity * uplift). Rates use class-count zero guards: denominators are max(1, P) and max(1, N); if P=0, P_miss=0; if N=0, P_fa=0. Costs: c_FP/c_FN invalid (non-finite/<=0 clipped to 1.0 each; warning logged to avoid degeneracy). Defaults: norm_mode='null'. Uncertainty 95% CI/scenarios. Multilingual: Detect->Execute->Translate (EN/JA/ZH/FR/ES/PT/KO; fidelity 1.0 if language-match, else 1 - min(0.02, 0.01 + 0.02*(1 - lang_conf)); LanguageContext interface; fallback: detected -> user_lang -> 'en'). Evidence: CAG (w_type + w_fresh, 2yr window; relax for non-time-sensitive topics like theory/classics, with rationale logged), max 20; pass if >=14. Debias: Elo-CPC; logs assumptions/reproducibility (incl. seed if user_options.seed specified). Quick Start: AI Proposal Mode -> theme/output pref -> auto-lang -> analysis + translate offer. E.g., healthcare XAI (Paper 5): diagnostic theme -> fidelity-L breakdown. Apply in order: τ* -> L -> S_base -> clip[0,1] -> fidelity -> uplift -> S_final (clip[0,1]).

### How to Use
The method for starting your analysis with the Draco framework may vary slightly depending on the AI you are using. Please try one of the following methods according to your environment.

#### **Usual Procedure (AI Proposal Mode, Recommended)**
Copy the entire "Prompt Body" below and paste into your AI chat (Grok-1.5+, GPT-4o, Claude 3 recommended). Send it to trigger AI Proposal Mode.
Wait for the prompt response, then reply with details.
* Example: Analysis Theme: Impact of Recent Tariff Policies on Global Trade, Analysis Level: Standard, Output Form: Details, Uplift: Default, Analysis Language: en, User Language: en

#### **Option: Direct Specification**
Append to the prompt end: "user_options = {'user_theme': 'Your theme', 'user_lang': 'en', ...}".
* Example:
```
... (End of the prompt) ...
## ====== Execution Instructions ======
Follow framework, start analysis on user's options. Present AI Proposal Mode if unspecified, else proceed. user_options input ↓
user_options = {'user_theme': 'Healthcare XAI diagnostic theme', 'analysis_level': 'Simple', 'output_form': 'Quick-read', 'uplift': 'Demo', 'analysis_language': 'en', 'user_lang': 'ja'}
```
Send entire text at once.

---
### --- ▼▼▼ PROMPT BODY (COPY FROM HERE) ▼▼▼ ---
# Minimal runtime guards (self-contained)
import math
try:
    import numpy as np
except Exception:
    class _NP:
        @staticmethod
        def isfinite(x):
            try:
                return math.isfinite(float(x))
            except Exception:
                return False
    np = _NP()
# Central constants (aligned with docs)
EPS = 1e-12
RATIO_MAX = 1e4
COST_MIN = 1e-6

Execute with accuracy priority (up to 60 min, ≤12 parallel tools). Anchor timestamp:
```python
import datetime
from datetime import timezone
try:
    from zoneinfo import ZoneInfo
    def now_sv():
        now_utc = datetime.datetime.now(ZoneInfo('UTC'))
        return now_utc.astimezone(ZoneInfo('America/Los_Angeles'))
except ImportError:
    def now_sv():
        now_utc = datetime.datetime.now(timezone.utc)
        return now_utc  # UTC fallback
silicon_valley_time = now_sv()
timestamp_str = silicon_valley_time.strftime("%B %d, %Y %H:%M %Z")
# print(timestamp_str)  # Optional: output only if needed
current_year = silicon_valley_time.year
current_year_minus_two = silicon_valley_time.year - 2
```
Post-analysis: Append "To get this analysis in another language, please specify the target (e.g., 'in Japanese') after the output." (localized per user_lang; single-point def, once-only via flag).

1. **Input Parse:** Theme -> complexity (qual/quant via num/unit ratio); lang detect (LanguageContext). Force AI Proposal Mode on empty/unspecified: 
```python
from textwrap import dedent
def set_seed(s):
    import random
    random.seed(s)
    try:
        np.random.seed(s)
    except:
        pass
    try:
        import torch
        torch.manual_seed(s)
        import os
        torch.cuda.manual_seed_all(s)
    except:
        pass

def normalize_options(user_options):
    norm_opts = dict(user_options) # shallow copy
    # Level: analysis_level/level -> 'level' in {'simple','standard'}
    level_key = next((k for k in ['analysis_level', 'level'] if k in norm_opts), None)
    if level_key:
        val = str(norm_opts[level_key]).lower().strip()
        norm_opts['level'] = 'simple' if any(w in val for w in ('simple', 'simplified', 'basic', 'lite')) else 'standard'
        if level_key != 'level':
            del norm_opts[level_key]
    # Output: output_form/output_mode -> 'output_mode' in {'compact','standard'} (Quick-read <-> compact, Details <-> standard bidirectional)
    output_key = next((k for k in ['output_form', 'output_mode'] if k in norm_opts), None)
    if output_key:
        val = str(norm_opts[output_key]).lower().strip()
        # treat 'detail'/'details' as standard
        compact_tokens = ('quick', 'compact', 'condensed', 'short')
        standard_tokens = ('detail', 'details', 'standard', 'full')
        mode = None
        if any(w in val for w in standard_tokens):
            mode = 'standard'
        elif any(w in val for w in compact_tokens):
            mode = 'compact'
        norm_opts['output_mode'] = mode or 'standard'
        if output_key != 'output_mode':
            del norm_opts[output_key]
    # Uplift: numeric passthrough or 'demo'->1.05 else 1.00; default 1.00
    if 'uplift' in norm_opts:
        v = norm_opts['uplift']
        if isinstance(v, (int, float)):
            norm_opts['uplift'] = float(v)
        else:
            val = str(v).lower().strip()
            try:
                norm_opts['uplift'] = float(val)
            except ValueError:
                norm_opts['uplift'] = 1.05 if 'demo' in val else 1.00
    else:
        norm_opts['uplift'] = 1.00
    # Langs: 2-letter or BCP47 (hyphen/underscore) -> take first 2 letters; invalid -> 'auto'
    # Alias mapping: jp->ja, cn->zh, kr->ko (extend as needed)
    alias_map = {'jp': 'ja', 'cn': 'zh', 'kr': 'ko', 'pt-br': 'pt', 'zh-cn': 'zh', 'zh-tw': 'zh'}
    for k in ['user_lang', 'analysis_language']:
        raw = norm_opts.get(k, None)
        val = '' if raw is None else str(raw).lower().strip()
        if not val:
            norm_opts[k] = 'auto'
            continue
        if val == 'auto':
            norm_opts[k] = 'auto'
            continue
        val = val.replace('_', '-') # allow underscore tags like pt_BR
        if val in alias_map:
            val = alias_map[val]
        if len(val) == 2 and val.isalpha():
            norm_opts[k] = val
        elif '-' in val:
            head = val.split('-', 1)[0]
            if len(head) == 2 and head.isalpha():
                norm_opts[k] = head
            else:
                norm_opts[k] = 'auto'
        else:
            norm_opts[k] = 'auto'
    # Fidelity: None/invalid->'auto'; numeric/string-numeric clipped to 0.8-1.0
    fid = norm_opts.get('fidelity', None)
    if fid is None:
        norm_opts['fidelity'] = 'auto'
    else:
        try:
            fval = float(fid)
            norm_opts['fidelity'] = max(0.8, min(1.0, fval))
        except (TypeError, ValueError):
            norm_opts['fidelity'] = 'auto'
    # Seed: int normalization if provided
    if 'seed' in norm_opts:
        try:
            norm_opts['seed'] = int(str(norm_opts['seed']).strip())
        except:
            del norm_opts['seed']
    # Defaults: explicit set
    if 'level' not in norm_opts:
        norm_opts['level'] = 'standard'
    if 'output_mode' not in norm_opts:
        norm_opts['output_mode'] = 'standard'
    # uplift safe clip
    if isinstance(norm_opts.get('uplift'), (int, float)):
        norm_opts['uplift'] = max(0.5, min(2.0, float(norm_opts['uplift'])))
    # Light guard for key numerics
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None
    # Defaults for missing values (NEW)
    if 'pi' not in norm_opts:
        norm_opts['pi'] = 0.5
    else:
        p = _to_float(norm_opts['pi'])
        norm_opts['pi'] = min(1.0, max(0.0, p)) if p is not None else 0.5
    cost_corrected = False
    for k in ('c_fp', 'c_fn'):
        if k not in norm_opts:
            norm_opts[k] = 1.0
        else:
            v = _to_float(norm_opts[k])
            if v is None or not np.isfinite(v) or v <= 0.0:
                norm_opts[k] = 1.0
                cost_corrected = True
            else:
                norm_opts[k] = max(COST_MIN, min(RATIO_MAX, v))
    if cost_corrected:
        norm_opts['_cost_corrected'] = True  # Flag for logging
    if 'norm_mode' not in norm_opts:
        norm_opts['norm_mode'] = 'null'  # Recommended default for conservative S
    else:
        mm = str(norm_opts['norm_mode']).lower().strip()
        if mm not in ('null', 'worst_trivial', 'strict'):
            norm_opts['norm_mode'] = 'null'
    # Optional: safe trim user_theme (NEW; only if needed)
    ut = str(norm_opts.get('user_theme', '') or '')
    if len(ut) > 4096:
        norm_opts['user_theme'] = ut[:4096]
    return norm_opts

def handle_user_options(user_options):
    user_options = normalize_options(user_options)
    echo = bool(user_options.get('debug', False)) # default non-output
    user_theme = str(user_options.get('user_theme', "") or "").strip()
    if not user_theme:
        prompt_text = dedent("""\
        ## ====== AI Proposal Mode ======
        **Analysis Theme**: [User input, e.g., "Impact of Recent Tariff Policies on Global Trade: Proposals for Actions the American Middle Class Can Take"]
        **Analysis Level**: □ Standard □ Simple [Default: Standard]
        **Output Form**: □ Quick-read □ Details [Default: Details]
        **Uplift**: □ Default (1.00) □ Demo (+5%) [Default: Default]
        **Analysis Language**: [e.g., 'en'/'ja'/'zh'/'fr'/'es'/'pt'/'ko'; Default: Auto-detected]
        **User Language**: [e.g., 'ja'; Default: Detected or 'en']
        """)
        if echo:
            print(prompt_text)
            print("Reply with updated options to proceed.")
        return {'status': 'await_input', 'data': prompt_text}
    return {'status': 'ready', 'data': user_options}

# usage example
user_options = {"uplift": 1.00, "user_lang": "en", "c_fp": 1.0, "c_fn": 3.0, "pi": 0.5, "norm_mode": "null", "fidelity": None, "debug": False, "output_mode": "standard", "user_theme": ""}  # Default empty theme
status = handle_user_options(user_options)
# print optional; external handler uses return as criterion
```
2. **Agent Team:** form_multi_team() -> [data,predict,vae,drl,reflect,xai].
3. **Forward Pass:** >= 3 iters; Elo-debate CPC; X-CII sim (start 500 reps, escalate to 1000 if variance > thr; η/ρ sens; seed from user_options.seed if specified); Expected Loss (unified guards); fidelity/uplift apply to S only (post-clip; demo-gate).
4. **Backward:** Trigger if any: var > thr, err > thr, S < 0.5, L > 1.2 * L_null, or bias > 0.03. (thr denotes configured thresholds in the scheduler/config.)
5. **Output:** User_lang; minimal meta; opposing views/Elo-CPC/X-CII align.
Transparency: Queries/ANN passes/X-CII sims (L vals); source langs internal (meta minimal).

## ====== Tool Guidelines ======
- web_search_with_snippets: Diverse, analysis_language optional.
- browse_page: Facts/args; native-phrase cross-ref.
- code_execution: ANN/X-CII sim (start 500 reps, escalate to 1000 if variance > thr; expected_loss); language interface; URL sanitize (util allow/deny); redirect detect; timeout 10-20s/retry 3 (centralized in scheduler; fallback: shorten query -> cache -> abort).

## ====== Ethical Guidelines ======
Min harm/bias: Diverse sources, X-CII S checks (FP/FN costs), transparency/explainability (methods/uncertainty/lang-fidelity). Context: source_lang analyze; user_lang output/meta; orig phrasing retain. Details in README.

## ====== Output Format ======
A. **Executive Summary**  
Plain-Language: 3-5 sentences (facts/problems/resolutions, user_lang).  
Key Messages: 3-4 bullets (facts/problems/actions).  
[Meta: "Original: SRC (analysis in SRC, shown only if detection confidence >= 0.85)"]  # Minimal meta only; SRC=analysis lang code or omit if auto-fail.

1. **Framework Log** (debug-only): ANN + X-CII (seed, reps, norm_mode, λ, τ*, L(τ*), S_base/S_final, langs, SciPy path, bootstrap B=min(2000, max(800, 200 + 50*log10(n_eff+10))), EPS/RATIO_MAX values, P/N zero flags, cost correction flag, CAG score/pass).  

2. **Key Findings:** Report medians and 95% CIs with 2-3 decimals (e.g., 0.83 [0.78-0.87]). Report S_base and S_final separately.  
| Component | Median | 95% CI |  
|-----------|--------|--------|  
| Q | [Q med] | [low-high] |  
| E | [E med] | [low-high] |  
| S_base | [S_base med] | [low-high] |  
| S_final | [S_final med] | [low-high] |  

3. **Risks & Mitigations:**  
- High c_FP (e.g., misleading data): τ* opt balance FP/FN.  
- High c_FN (missed ops): Fidelity-scale S (F≈0.98–1.00; auto).  
- [Runtime].  

4. **Suggestions:** High/Med/Low actions.  

5. **Counterarguments:** Diverse sources.  

**Terms:** See README for full glossary (X-CII: Q/E/S power-mean λ=0.25 definition, Paper 4; Expected Loss: SDT FP/FN, Paper 5; Fidelity: 0.98–1.00 (auto: 1 - min(0.02, 0.01 + 0.02*(1 - lang_conf)); lang_conf=None→0.9)). CAG = w_type + w_fresh (2yr window; relax for non-time-sensitive topics), max 20; pass if >=14. Weights in Config. η: noise sensitivity parameter; ρ: correlation-robustness parameter. d' (d_prime) is tracked as an invariant. π (pi), τ (tau), λ (lambda), η (eta), ρ (rho): ASCII aliases may appear in code/logs.  

**Fact Checklist:** Strong/Mod/Hypothesis [verified/source/spec].  

[Debug: Full log].  
[Suffix: Localized translate prompt (once-only)].
---
### --- ▲▲▲ PROMPT BODY (COPY UNTIL HERE) ▲▲▲ ---
