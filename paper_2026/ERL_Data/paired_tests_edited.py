import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ─── File paths ───────────────────────────────────────────────────────────────
PRE_PAIRED_FP  = "data/erl_pre_paired_edited.csv"
POST_PAIRED_FP = "data/erl_post_paired_edited.csv"
OUTPUT_DIR     = "results/revised_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 1. Load & clean data ────────────────────────────────────────────────────
pre_raw  = pd.read_csv(PRE_PAIRED_FP, index_col=0)
post_raw = pd.read_csv(POST_PAIRED_FP, index_col=0)

# Pair by position (indices differ but both have 27 rows in matched order)
pre_raw  = pre_raw.reset_index(drop=True)
post_raw = post_raw.reset_index(drop=True)

N = len(pre_raw)
print(f"Paired sample size: N = {N}\n")

# ─── 2. Construct variable mapping ─────────────────────────────────────────
# Each entry: (short_label, pre_col_name_or_None, post_col_name_or_None)

PAIRED_ITEMS = {
    # ── Knowledge (K) ──
    "K: Sensor familiarity": (
        "I am familiar with how sensors are used in robotics",
        "I am familiar with how sensors are used in robotics."
    ),
    "K: LiDAR/mapping familiarity": (
        "I am familiar with concepts like LiDAR, mapping, or sensor noise.",
        "I am familiar with concepts like LiDAR, mapping, or sensor noise."
    ),
    "K: OGM confidence": (
        "I am familiar with occupancy grid mapping (OGM). I am confident in updating an occupancy grid map using sensor data.",
        "I am familiar with occupancy grid mapping (OGM). I am confident in updating an occupancy grid map using sensor data."
    ),
    "K: Simulation tools": (
        "I understand how simulation tools are used in robotics",
        "I understand how simulation tools are used in robotics"
    ),
    "K: Bayesian estimation": (
        "I understand the concept of Bayesian estimation.",
        "I understand the concept of Bayesian estimation."
    ),
    "K: Pose-map uncertainty": (
        "I understand the relationship between robot pose uncertainty and map uncertainty.",
        "I understand the relationship between robot pose uncertainty and map uncertainty."
    ),    
    # ── Interest and Career (IC) ──
    "IC: Interest in robotics": (
        "I am interested in learning more about robotics",
        "I am interested in learning more about robotics."
    ),
    "IC: Interest in AI": (
        "I am interested in artificial intelligence (AI)",
        "I am interested in artificial intelligence (AI)."
    ),
    "IC: Would take robotics class": (
        "I would consider taking a robotics-related class. ",
        "I would consider taking a robotics-related class."
    ),
    "IC: Workshop helps career": (
        "I believe that engaging in this workshop can help my career",
        "I believe that engaging in this workshop can help my career."
    ),
    # ── Self-Efficacy (SE) ──
    "SE: Python comfort": (
        "I am comfortable writing code in Python ",
        "I am comfortable writing code in Python."
    ),
    "SE: Jupyter/Colab familiarity": (
        "I am familiar with interactive computing tools (Jupyter Notebook, Google Colab, etc.)",
        "I am familiar with interactive computing tools (Jupyter Notebook, Google Colab, etc.)."
    ),
    "SE: LiDAR coordinate frames": (
        "I am confident in transforming LiDAR scan data between coordinate frames.",
        "I am confident in transforming LiDAR scan data between coordinate frames."
    ),
    "SE: Bresenham's algorithm": (
        "I am confident in implementing or modifying Bresenham\u2019s line algorithm.",
        "I am confident in implementing or modifying Bresenham\u2019s line algorithm."
    ),
    "SE: OGM confidence": (
        "I am familiar with occupancy grid mapping (OGM). I am confident in updating an occupancy grid map using sensor data.",
        "I am familiar with occupancy grid mapping (OGM). I am confident in updating an occupancy grid map using sensor data."
    ),
    "SE: SLAM confidence": (
        "I feel confident I could learn the basics of simultaneous localization and mapping (SLAM)",
        "I feel confident in learning simultaneous localization and mapping (SLAM)."
    ),
}

# Post-only items grouped by category
POST_ONLY_ITEMS = {
    "SE: Control over learning": "I had control over my personal learning during the workshop.",
    "SE: Effort proportional to learning": "How much I learned in this workshop is proportional to how much effort I put in.",
    "IC: Intend to explore website": "I intend to explore the website outside of the workshop.",
    "IC: More likely STEM career": "After this workshop, I am more likely to pursue a STEM-related career.",
    "K: Understand real-world use": "This workshop made me understand how robotics is used to solve real problems.",
    "Other: Website valuable resource": "I believe the website, Robot Proving Grounds, is a valuable resource for robotics education.",
    "Other: Mapping module comprehensive": "I believe the mapping module in the website is complete and comprehensive.",
    "K: PyBullet confidence": "I feel more confident in my ability to use simulation tools like PyBullet.",
    "SE: Python for simulations": "I could use Python to modify and run robotics simulations independently.",
    "SE: Discuss robotics w/ peers": "I feel confident discussing robotics concepts with my peers.",
    "SE: Can engage in robotics": "I believe I can engage in robotics because of this workshop.",
    "K: LiDAR for mapping": "I understand how LiDAR data can be used for mapping and localization.",
    "K: Sensor noise understanding": "I can explain the concept of sensor noise and its effect on robot perception.",
    "K: Bresenham purpose": "I understand the purpose of the Bresenham algorithm in mapping.",
    "IC: Increased interest": "This workshop increased my interest in robotics.",
    "R: Interest in research": "This workshop increased my interest in pursuing research in robotics or related fields.",
    "IC: More interested (post)": "This workshop made me more interested in robotics.",
    "IC: Join course/project": "I would like to take a course or join a project related to robotics.",
    "IC: Workshop made more interested": "I believe this workshop has made me more interested in robotics.",
    "Other: Instructors willing to assist": "The instructors hosting the workshop were willing to assist me.",
    "Other: Well organized": "The workshop was well organized and made it easier to learn.",
    "Other: Engaging methods": "  The instructional methods used in this workshop were engaging.   ",
    "Other: Hands-on helped theory": "The hands-on coding activities helped me understand the theory.",
    "Other: Theory-practice balance": "The balance between theory and practice was appropriate.",
    "Other: Notebooks helpful": "The Python notebooks were helpful and of appropriate difficulty.",
    "SE: Concepts difficult": "The concepts presented in the workshop are difficult to understand.",
    "Other: Experience rating": "Thank you for participating in the workshop! How would you rate your experience?",
}

# Pre-only items (no post-survey equivalent exists)
PRE_ONLY_ITEMS = {
    "K: Previous robotics experience": "I have previous experience in robotics",
    "SE: Robotic systems useful": "I understand what robotic systems may be useful for",
}


# ─── 3. Helper functions ─────────────────────────────────────────────────────

def rank_biserial(x, y):
    """Matched-pairs rank-biserial correlation effect size for Wilcoxon test."""
    diff = y - x
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    abs_ranks = stats.rankdata(np.abs(diff))
    r_plus = np.sum(abs_ranks[diff > 0])
    r_minus = np.sum(abs_ranks[diff < 0])
    n = len(diff)
    r = (r_plus - r_minus) / (n * (n + 1) / 2)
    return r


def effect_size_label(r):
    """Interpret rank-biserial r."""
    ar = abs(r)
    if ar < 0.1:
        return "negligible"
    elif ar < 0.3:
        return "small"
    elif ar < 0.5:
        return "medium"
    else:
        return "large"


def cronbach_alpha(df):
    """Compute Cronbach's alpha for a DataFrame of items."""
    df = df.dropna()
    n_items = df.shape[1]
    if n_items < 2 or len(df) < 2:
        return np.nan
    item_vars = df.var(axis=0, ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)
    return alpha


# ─── 4. Paired Wilcoxon signed-rank tests ────────────────────────────────────

print("=" * 80)
print("PAIRED WILCOXON SIGNED-RANK TESTS (Pre vs Post)")
print("=" * 80)

results = []
n_tests = len(PAIRED_ITEMS)

for label, (pre_col, post_col) in PAIRED_ITEMS.items():
    pre_vals = pd.to_numeric(pre_raw[pre_col], errors='coerce')
    post_vals = pd.to_numeric(post_raw[post_col], errors='coerce')

    # Drop pairs where either is NaN
    mask = pre_vals.notna() & post_vals.notna()
    pre_v = pre_vals[mask].values
    post_v = post_vals[mask].values
    n_valid = len(pre_v)

    diff = post_v - pre_v

    # Descriptive stats
    pre_median = np.median(pre_v)
    post_median = np.median(post_v)
    pre_mean = np.mean(pre_v)
    post_mean = np.mean(post_v)
    pre_sd = np.std(pre_v, ddof=1)
    post_sd = np.std(post_v, ddof=1)
    pre_iqr = np.subtract(*np.percentile(pre_v, [75, 25]))
    post_iqr = np.subtract(*np.percentile(post_v, [75, 25]))
    mean_diff = np.mean(diff)

    # Wilcoxon signed-rank test (two-sided)
    # Filter out zero differences for the test
    non_zero_diff = diff[diff != 0]
    if len(non_zero_diff) > 0:
        stat_w, p_val = stats.wilcoxon(pre_v, post_v, alternative='two-sided')
    else:
        stat_w, p_val = np.nan, 1.0

    # Effect size
    r = rank_biserial(pre_v, post_v)

    # Bonferroni-corrected p-value
    p_corrected = min(p_val * n_tests, 1.0)

    results.append({
        'Variable': label,
        'n': n_valid,
        'Pre Mean': round(pre_mean, 2),
        'Pre SD': round(pre_sd, 2),
        'Pre Median': pre_median,
        'Pre IQR': pre_iqr,
        'Post Mean': round(post_mean, 2),
        'Post SD': round(post_sd, 2),
        'Post Median': post_median,
        'Post IQR': post_iqr,
        'Mean Diff': round(mean_diff, 2),
        'W': stat_w,
        'p-value': round(p_val, 4),
        'p (Bonferroni)': round(p_corrected, 4),
        'Effect r': round(r, 3),
        'Effect Size': effect_size_label(r),
    })

    sig_marker = ""
    if p_corrected < 0.001:
        sig_marker = " ***"
    elif p_corrected < 0.01:
        sig_marker = " **"
    elif p_corrected < 0.05:
        sig_marker = " *"

    print(f"\n{label}")
    print(f"  n={n_valid}  Pre: M={pre_mean:.2f} (SD={pre_sd:.2f}), Mdn={pre_median:.1f}")
    print(f"          Post: M={post_mean:.2f} (SD={post_sd:.2f}), Mdn={post_median:.1f}")
    print(f"  Mean diff = {mean_diff:+.2f}")
    print(f"  W={stat_w}, p={p_val:.4f}, p(Bonf)={p_corrected:.4f}{sig_marker}")
    print(f"  Effect r={r:.3f} ({effect_size_label(r)})")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "paired_wilcoxon_results.csv"), index=False)
print(f"\n[Saved: {OUTPUT_DIR}/paired_wilcoxon_results.csv]")

# ─── 5. Construct composite scores (paired items) ────────────────────────────────

print("\n" + "=" * 80)
print("CATEGORY COMPOSITE SCORES — Paired Pre/Post Analysis")
print("=" * 80)

# Group labels by category prefix
CONSTRUCT_CATEGORIES = {
    'K (Knowledge)': 'K:',
    'IC (Interest and Career)': 'IC:',    
    'SE (Self-Efficacy)': 'SE:',
    'R (Research)': 'R:',
}

composite_results = []

for cat_name, prefix in CONSTRUCT_CATEGORIES.items():
    cat_items = {k: v for k, v in PAIRED_ITEMS.items() if k.startswith(prefix)}
    if not cat_items:
        continue

    pre_scores = []
    post_scores = []

    pre_item_df_cols = []
    post_item_df_cols = []

    for label, (pre_col, post_col) in cat_items.items():
        pre_s = pd.to_numeric(pre_raw[pre_col], errors='coerce')
        post_s = pd.to_numeric(post_raw[post_col], errors='coerce')
        pre_scores.append(pre_s)
        post_scores.append(post_s)
        pre_item_df_cols.append(pre_s.rename(label))
        post_item_df_cols.append(post_s.rename(label))

    pre_composite = pd.concat(pre_scores, axis=1).mean(axis=1)
    post_composite = pd.concat(post_scores, axis=1).mean(axis=1)

    # Cronbach's alpha
    pre_alpha = cronbach_alpha(pd.concat(pre_item_df_cols, axis=1).dropna())
    post_alpha = cronbach_alpha(pd.concat(post_item_df_cols, axis=1).dropna())

    mask = pre_composite.notna() & post_composite.notna()
    pre_c = pre_composite[mask].values
    post_c = post_composite[mask].values
    n_valid = len(pre_c)

    if n_valid > 0 and np.any(post_c - pre_c != 0):
        stat_w, p_val = stats.wilcoxon(pre_c, post_c)
    else:
        stat_w, p_val = np.nan, 1.0

    r = rank_biserial(pre_c, post_c)
    mean_diff = np.mean(post_c - pre_c)

    composite_results.append({
        'Construct Category': cat_name,
        'n_items': len(cat_items),
        'n': n_valid,
        'Pre Mean': round(np.mean(pre_c), 2),
        'Post Mean': round(np.mean(post_c), 2),
        'Mean Diff': round(mean_diff, 2),
        'W': stat_w,
        'p-value': round(p_val, 4),
        'Effect r': round(r, 3),
        'Effect Size': effect_size_label(r),
        "Cronbach alpha (pre)": round(pre_alpha, 3) if not np.isnan(pre_alpha) else 'N/A',
        "Cronbach alpha (post)": round(post_alpha, 3) if not np.isnan(post_alpha) else 'N/A',
    })

    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    print(f"\n{cat_name} ({len(cat_items)} items)")
    print(f"  Cronbach alpha: pre={pre_alpha:.3f}, post={post_alpha:.3f}")
    print(f"  Pre mean={np.mean(pre_c):.2f}, Post mean={np.mean(post_c):.2f}, Diff={mean_diff:+.2f}")
    print(f"  W={stat_w}, p={p_val:.4f} {sig}, r={r:.3f} ({effect_size_label(r)})")

comp_df = pd.DataFrame(composite_results)
comp_df.to_csv(os.path.join(OUTPUT_DIR, "construct_composite_results.csv"), index=False)
print(f"\n[Saved: {OUTPUT_DIR}/construct_composite_results.csv]")

# ─── 6. Post-only Construct descriptives ─────────────────────────────────────────

print("\n" + "=" * 80)
print("POST-ONLY CATEGORY DESCRIPTIVES")
print("=" * 80)

post_only_results = []

for label, col in POST_ONLY_ITEMS.items():
    vals = pd.to_numeric(post_raw[col], errors='coerce').dropna()
    cat = label.split(":")[0]
    post_only_results.append({
        'Variable': label,
        'Category': cat,
        'n': len(vals),
        'Mean': round(vals.mean(), 2),
        'SD': round(vals.std(ddof=1), 2),
        'Median': vals.median(),
        'IQR': round(np.subtract(*np.percentile(vals, [75, 25])), 2),
        'Min': vals.min(),
        'Max': vals.max(),
    })
    print(f"  {label}: M={vals.mean():.2f} (SD={vals.std(ddof=1):.2f}), Mdn={vals.median():.1f}, range=[{vals.min():.0f}, {vals.max():.0f}]")

post_only_df = pd.DataFrame(post_only_results)
post_only_df.to_csv(os.path.join(OUTPUT_DIR, "construct_descriptives.csv"), index=False)
print(f"\n[Saved: {OUTPUT_DIR}/post_only_construct_descriptives.csv]")

# Post-only Cronbach's alpha per construct category
print("\nPost-only Cronbach's alpha by construct category:")
for cat_prefix, cat_full in [('K', 'Knowledge'), ('IC', 'Interest and Career'), ('SE', 'Self-Efficacy'),
                              ('R', 'Research')]:
    cat_cols = [col for label, col in POST_ONLY_ITEMS.items() if label.startswith(cat_prefix + ":")]
    if len(cat_cols) >= 2:
        item_data = post_raw[cat_cols].apply(pd.to_numeric, errors='coerce')
        alpha = cronbach_alpha(item_data)
        print(f"  {cat_full} ({len(cat_cols)} items): alpha = {alpha:.3f}")

# ─── 6b. Pre-only descriptives ───────────────────────────────────────────────

print("\n" + "=" * 80)
print("PRE-ONLY DESCRIPTIVES")
print("=" * 80)

pre_only_results = []

for label, col in PRE_ONLY_ITEMS.items():
    vals = pd.to_numeric(pre_raw[col], errors='coerce').dropna()
    cat = label.split(":")[0]
    pre_only_results.append({
        'Variable': label,
        'Category': cat,
        'n': len(vals),
        'Mean': round(vals.mean(), 2),
        'SD': round(vals.std(ddof=1), 2),
        'Median': vals.median(),
        'IQR': round(np.subtract(*np.percentile(vals, [75, 25])), 2),
        'Min': vals.min(),
        'Max': vals.max(),
    })
    print(f"  {label}: M={vals.mean():.2f} (SD={vals.std(ddof=1):.2f}), Mdn={vals.median():.1f}, range=[{vals.min():.0f}, {vals.max():.0f}]")

pre_only_df = pd.DataFrame(pre_only_results)
pre_only_df.to_csv(os.path.join(OUTPUT_DIR, "pre_only_descriptives.csv"), index=False)
print(f"\n[Saved: {OUTPUT_DIR}/pre_only_descriptives.csv]")


# ─── 7. Visualizations ───────────────────────────────────────────────────────

# 7a. Grouped bar chart: Pre vs Post means by construct category (paired items)
fig, ax = plt.subplots(figsize=(10, 6))
cats = [r['Construct Category'] for r in composite_results]
pre_means = [r['Pre Mean'] for r in composite_results]
post_means = [r['Post Mean'] for r in composite_results]
x = np.arange(len(cats))
width = 0.35
bars1 = ax.bar(x - width/2, pre_means, width, label='Pre', color='#4C72B0', alpha=0.85)
bars2 = ax.bar(x + width/2, post_means, width, label='Post', color='#DD8452', alpha=0.85)
ax.set_ylabel('Mean Score (1-7 Likert)')
ax.set_title('Pre vs Post Composite Scores by Category')
ax.set_xticks(x)
ax.set_xticklabels(cats, rotation=15, ha='right')
ax.legend()
ax.set_ylim(0, 7.5)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "composite_pre_post_bar.png"), dpi=150)
plt.close()
print(f"\n[Saved: {OUTPUT_DIR}/composite_pre_post_bar.png]")

# 7b. Heatmap of mean change (post - pre) for all paired items
fig, ax = plt.subplots(figsize=(8, 10))
labels = [r['Variable'] for r in results]
diffs = [r['Mean Diff'] for r in results]
p_bonf = [r['p (Bonferroni)'] for r in results]

colors = []
for d, p in zip(diffs, p_bonf):
    colors.append(d)

y_pos = np.arange(len(labels))
hbars = ax.barh(y_pos, diffs, color=[plt.cm.RdYlGn((d + 3) / 6) for d in diffs])

# Add significance markers
for i, (d, p) in enumerate(zip(diffs, p_bonf)):
    marker = ""
    if p < 0.001:
        marker = "***"
    elif p < 0.01:
        marker = "**"
    elif p < 0.05:
        marker = "*"
    offset = 0.05 if d >= 0 else -0.05
    ax.text(d + offset, i, f"{d:+.2f} {marker}", va='center', fontsize=8,
            ha='left' if d >= 0 else 'right')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel('Mean Change (Post - Pre)')
ax.set_title('Pre→Post Mean Change by Item\n(* p<.05, ** p<.01, *** p<.001, Bonferroni)')
ax.axvline(x=0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mean_change_heatmap.png"), dpi=150)
plt.close()
print(f"[Saved: {OUTPUT_DIR}/mean_change_heatmap.png]")

# 7c. Box plots: Pre vs Post for paired items, color-coded by category
category_colors = {
    'K:': '#4C72B0',
    'IC:': '#55A868',
    'SE:': '#C44E52',
    'R:': '#8172B2',
    'Other:': '#937860',
}



fig, axes = plt.subplots(4, 4, figsize=(18, 16))
axes = axes.flatten()

for idx, (label, (pre_col, post_col)) in enumerate(PAIRED_ITEMS.items()):
    if idx >= len(axes):
        break
    ax = axes[idx]
    pre_v = pd.to_numeric(pre_raw[pre_col], errors='coerce').dropna().values
    post_v = pd.to_numeric(post_raw[post_col], errors='coerce').dropna().values

    prefix = label.split(":")[0] + ":"
    color = category_colors.get(prefix, '#333333')

    bp = ax.boxplot([pre_v, post_v], tick_labels=['Pre', 'Post'], patch_artist=True,
                    widths=0.5)
    bp['boxes'][0].set_facecolor(color)
    bp['boxes'][0].set_alpha(0.4)
    bp['boxes'][1].set_facecolor(color)
    bp['boxes'][1].set_alpha(0.8)
    ax.set_title(label, fontsize=8, fontweight='bold')
    ax.set_ylim(0.5, 7.5)
    ax.set_ylabel('Score')

# Hide unused subplots
for idx in range(len(PAIRED_ITEMS), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Pre vs Post Distributions by Item (color = category)', fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "paired_boxplots.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"[Saved: {OUTPUT_DIR}/paired_boxplots.png]")

# 7c-paired. Box plots using only paired (complete-case) data
fig, axes = plt.subplots(4, 4, figsize=(18, 16))
axes = axes.flatten()

for idx, (label, (pre_col, post_col)) in enumerate(PAIRED_ITEMS.items()):
    if idx >= len(axes):
        break
    ax = axes[idx]
    pre_vals = pd.to_numeric(pre_raw[pre_col], errors='coerce')
    post_vals = pd.to_numeric(post_raw[post_col], errors='coerce')

    # Keep only complete pairs
    mask = pre_vals.notna() & post_vals.notna()
    pre_v = pre_vals[mask].values
    post_v = post_vals[mask].values

    prefix = label.split(":")[0] + ":"
    color = category_colors.get(prefix, '#333333')

    bp = ax.boxplot([pre_v, post_v], tick_labels=['Pre', 'Post'], patch_artist=True,
                    widths=0.5)
    bp['boxes'][0].set_facecolor(color)
    bp['boxes'][0].set_alpha(0.4)
    bp['boxes'][1].set_facecolor(color)
    bp['boxes'][1].set_alpha(0.8)
    ax.set_title(f"{label} (n={len(pre_v)})", fontsize=8, fontweight='bold')
    ax.set_ylim(0.5, 7.5)
    ax.set_ylabel('Score')

# Hide unused subplots
for idx in range(len(PAIRED_ITEMS), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Pre vs Post Distributions — Paired Data Only (color = category)',
             fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "paired_only_boxplots.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"[Saved: {OUTPUT_DIR}/paired_only_boxplots.png]")

# 7d. Radar chart: Post-only construct composite scores
construct_post_composites = {}
for cat_prefix, cat_full in [('K', 'Knowledge'), ('IC', 'Interest and Career'), ('SE', 'Self-Efficacy'),
                              ('R', 'Research')]:
    cat_items = [col for label, col in POST_ONLY_ITEMS.items() if label.startswith(cat_prefix + ":")]
    if cat_items:
        vals = post_raw[cat_items].apply(pd.to_numeric, errors='coerce').mean(axis=1).dropna()
        construct_post_composites[cat_full] = vals.mean()

if construct_post_composites:
    categories_r = list(construct_post_composites.keys())
    values_r = list(construct_post_composites.values())
    num_vars = len(categories_r)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values_r += values_r[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.fill(angles, values_r, color='#4C72B0', alpha=0.25)
    ax.plot(angles, values_r, color='#4C72B0', linewidth=2, marker='o')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories_r, fontsize=11)
    ax.set_ylim(0, 7)
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_title('Post-Workshop Category Composite Scores', fontsize=13, pad=20)

    for angle, val, cat in zip(angles[:-1], values_r[:-1], categories_r):
        ax.text(angle, val + 0.3, f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "construct_radar_post.png"), dpi=150)
    plt.close()
    print(f"[Saved: {OUTPUT_DIR}/construct_radar_post.png]")

# 7e. P-value plot per question (paired items)
fig, ax = plt.subplots(figsize=(10, 8))
labels_p = [r['Variable'] for r in results]
p_vals_raw = [r['p-value'] for r in results]
p_vals_bonf = [r['p (Bonferroni)'] for r in results]

y_pos = np.arange(len(labels_p))

# Plot raw and Bonferroni-corrected p-values
ax.barh(y_pos, p_vals_raw, height=0.4, align='edge', label='Raw p-value',
        color='#4C72B0', alpha=0.8)
ax.barh(y_pos - 0.4, p_vals_bonf, height=0.4, align='edge',
        label='Bonferroni-corrected p', color='#DD8452', alpha=0.8)

# Significance thresholds
ax.axvline(x=0.05, color='red', linestyle='--', linewidth=1, label='α = 0.05')
ax.axvline(x=0.01, color='darkred', linestyle=':', linewidth=1, label='α = 0.01')

# Annotate raw p-values
for i, (p_raw, p_bonf) in enumerate(zip(p_vals_raw, p_vals_bonf)):
    ax.text(max(p_raw, 0.002) + 0.005, i + 0.2, f'{p_raw:.4f}', va='center',
            fontsize=7, color='#4C72B0')
    ax.text(max(p_bonf, 0.002) + 0.005, i - 0.2, f'{p_bonf:.4f}', va='center',
            fontsize=7, color='#DD8452')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels_p, fontsize=8)
ax.set_xlabel('p-value')
ax.set_title('P-Values per Question (Paired Wilcoxon Signed-Rank Test)')
ax.set_xlim(0, max(max(p_vals_bonf), max(p_vals_raw)) * 1.15)
ax.legend(loc='lower right', fontsize=8)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pvalue_per_question.png"), dpi=150)
plt.close()
print(f"[Saved: {OUTPUT_DIR}/pvalue_per_question.png]")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE — All results saved to results/")
print("=" * 80)
