import argparse
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ─── Style ────────────────────────────────────────────────────────────────────
PRE_COLOR = '#2a78d6'
POST_COLOR = '#eb6834'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
SURFACE = '#fcfcfb'
# Likert 1..7: red arm (disagree) -> gray neutral -> blue arm (agree)
LIKERT_COLORS = ['#a8302f', '#e34948', '#f2a09f', '#d6d5cf', '#86b6ef', '#2a78d6', '#104281']

plt.rcParams.update({
    'font.family': ['Segoe UI', 'DejaVu Sans', 'sans-serif'],
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'axes.edgecolor': AXIS,
    'axes.labelcolor': INK_2,
    'axes.titlecolor': INK,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': GRID,
    'grid.linewidth': 0.6,
    'axes.axisbelow': True,
    'xtick.color': INK_2,
    'ytick.color': INK_2,
    'legend.frameon': False,
    'savefig.facecolor': SURFACE,
})

# ─── Survey item mapping ──────────────────────────────────────────────────────
# Each entry: short_label -> (pre_col, post_col).
# Prefix = category: K (familiarity/knowledge), S (self-efficacy), I (interest).
# † = wording differs slightly between pre and post.
PAIRED_ITEMS = {
    "K: Planning in autonomous systems": (
        "I am familiar with planning in autonomous systems.",
        "I am familiar with planning in autonomous systems.",
    ),
    "K: Dijkstra / A*": (
        "I am familiar with Dijkstra's or A* algorithm.",
        "I am familiar with Dijkstra's/A* algorithm.",
    ),
    "K: RRT / RRT*": (
        "I am familiar with the RRT or RRT* algorithm.",
        "I am familiar with RRT/RRT* algorithm.",
    ),
    "K: Pure Pursuit": (
        "I am familiar with Pure Pursuit for path tracking control.",
        "I am familiar with Pure Pursuit.",
    ),
    "K: Simulation tools": (
        "I am familiar with robotics simulation tools (Gazebo, PyBullet, MuJoCo, etc.)",
        "I am familiar with robotics simulation tools (Gazebo, PyBullet, MuJoCo, etc.)",
    ),
    "S: Explain purpose of planning †": (
        "I believe I can explain the purpose of planning algorithms to a beginner.",
        "I believe I can explain the purpose of planning algorithms.",
    ),
    "S: Explain A*/Dijkstra/RRT †": (
        "I can confidently explain A*, Dijkstra, and RRT to a beginner.",
        "I can confidently explain the differences in A*, Dijkstra, and RRT.",
    ),
    "S: Implement a planner in code": (
        "I can confidently implement a planning algorithm (Dijkstra/A*/RRT) in code.",
        "I can confidently implement a planning algorithm (Dijkstra/A*/RRT) in code.",
    ),
    "S: Modify Pure Pursuit": (
        "I am confident in modifying pure pursuit to work on a new robot.",
        "I am confident in modifying pure pursuit to work on a new robot.",
    ),
    "S: Capable of robotics research": (
        "I believe I am capable of engaging in robotics research.",
        "I believe I am capable of engaging in robotics research.",
    ),
    "I: Learn more about robotics": (
        "I am interested in learning more about robotics.",
        "I am interested in learning more about robotics.",
    ),
    "I: Take a robotics class": (
        "I would consider taking a robotics-related class. ",
        "I would consider taking a robotics-related class. ",
    ),
    "I: Robotics research": (
        "I am interested in research related to robotics.",
        "I am interested in research related to robotics.",
    ),
}

CATEGORIES = {
    'K': 'Familiarity',
    'S': 'Self-efficacy',
    'I': 'Interest',
}

PRE_ONLY_ITEMS = {
    "Career in robotics": "I am interested in pursuing a career in robotics.",
    "Skills improve research/career opportunities":
        "Developing robotics skills will improve my opportunities to pursue research or a career in robotics.",
}

POST_ONLY_ITEMS = {
    "Can engage in robotics because of workshop": "I believe I can engage in robotics because of this workshop.",
    "Understand how robotics solves real problems":
        "This workshop made me understand how robotics is used to solve real problems.",
    "Workshop well organized and engaging": "The workshop was well organized and engaging.",
    "Concepts difficult to understand": "The concepts presented in the workshop are difficult to understand.",
    "Intend to explore the website": "I intend to explore the website outside of the workshop.",
    "Career in physical AI": "I am interested in pursuing a career in physical AI.",
}

# Multi-select questions asked in both surveys: label -> (pre_col, post_col)
PAIRED_MULTI = {
    "Robotics courses of interest at UCSD": (
        "Which robotics courses are you interested in at UCSD?",
        "Which robotics courses are you interested in at UCSD?",
    ),
    "Parts of robotics that interest you": (
        "Which parts of robotics interest you? (Select all that apply)  ",
        "Which parts of robotics interest you? (Select all that apply)  ",
    ),
    "What interests you about learning robotics": (
        "What interests you about learning robotics? (Select all that apply)",
        "What interests you about learning robotics? (Select all that apply)",
    ),
}

PRE_ONLY_MULTI = {
    "Pre: What prevented you from engaging in robotics research":
        "If interested, what has prevented you from engaging in robotics research? (Select all that apply)",
    "Pre: What do you hope to learn": "What do you hope to learn from this workshop? (Select all that apply)",
}

POST_ONLY_MULTI = {
    "Post: What did you finish in the workshop": "What did you finish in the workshop?",
    "Post: What did you learn": "What did you learn from this workshop? (Select all that apply)",
    "Post: What prevented you from engaging in robotics before":
        "What has prevented you from engaging in robotics before? (Select all that apply)",
    "Post: What might make it hard to continue":
        "What might make it hard for you to continue exploring robotics? (Select all that apply) ",
}

PRE_CATEGORICAL = {
    "Gender": "What is your gender?",
    "Education level": "What level of education are you currently enrolled in?",
    "Year of college": 'What year of college are you in? (only for "Undergraduate")',
}

POST_CATEGORICAL = {
    "Most challenging part": "What part of the workshop was most challenging? ",
    "Liked most": "What did you like most about the workshop?",
    "OK to contact for follow-up": "May we contact you in 3–6 months for a short follow-up survey?",
    "Used AI during workshop": "Did you use AI?",
    "Involved in research before": "Have you been involved in research before?",
}


# ─── Helper functions ─────────────────────────────────────────────────────────

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
    return (r_plus - r_minus) / (n * (n + 1) / 2)


def effect_size_label(r):
    ar = abs(r)
    if ar < 0.1:
        return "negligible"
    elif ar < 0.3:
        return "small"
    elif ar < 0.5:
        return "medium"
    return "large"


def cronbach_alpha(df):
    df = df.dropna()
    n_items = df.shape[1]
    if n_items < 2 or len(df) < 2:
        return np.nan
    total_var = df.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (n_items / (n_items - 1)) * (1 - df.var(axis=0, ddof=1).sum() / total_var)


def sig_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def split_multi(cell):
    """Splits a Google Forms multi-select cell on commas that are not inside parentheses."""
    if pd.isna(cell):
        return []
    parts, depth, cur = [], 0, ''
    for ch in str(cell):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(cur.strip())
            cur = ''
        else:
            cur += ch
    parts.append(cur.strip())
    return [p for p in parts if p]


def option_shares(series, fold_writeins=True):
    """Percent of respondents (non-blank) selecting each option.

    One-off answers are almost always "Other" free text, so they are folded into a single bar.
    """
    answered = series.dropna()
    cells = [split_multi(cell) for cell in answered]
    counts = pd.Series([opt for c in cells for opt in c]).value_counts()
    if fold_writeins:
        writeins = set(counts[counts == 1].index)
        if len(writeins) > 1:
            n_other = sum(any(opt in writeins for opt in c) for c in cells)
            counts = counts[counts > 1]
            counts['Other (write-in)'] = n_other
    return 100 * counts / max(len(answered), 1), len(answered)


def wrap(text, width=38):
    words, lines, cur = text.split(), [], ''
    for w in words:
        if cur and len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return '\n'.join(lines)


def style_box(bp, colors):
    for box, c in zip(bp['boxes'], colors):
        box.set_facecolor(c)
        box.set_alpha(0.35)
        box.set_edgecolor(c)
        box.set_linewidth(1.5)
    for key in ('whiskers', 'caps'):
        for line in bp[key]:
            line.set_color(MUTED)
    for med in bp['medians']:
        med.set_color(INK)
        med.set_linewidth(2)


def paired_box(ax, pre_v, post_v, title, p_corrected, rng):
    """Box plot of pre vs post with each participant's pair drawn as a faint line."""
    bp = ax.boxplot([pre_v, post_v], positions=[0, 1], widths=0.45, patch_artist=True,
                    showfliers=False, tick_labels=['Pre', 'Post'])
    style_box(bp, [PRE_COLOR, POST_COLOR])
    jitter = rng.uniform(-0.12, 0.12, size=(len(pre_v), 2))
    for i in range(len(pre_v)):
        ax.plot([0 + jitter[i, 0], 1 + jitter[i, 1]], [pre_v[i], post_v[i]],
                color=MUTED, alpha=0.25, linewidth=0.8, zorder=1)
    ax.scatter(jitter[:, 0], pre_v, s=12, color=PRE_COLOR, alpha=0.7, zorder=3, linewidths=0)
    ax.scatter(1 + jitter[:, 1], post_v, s=12, color=POST_COLOR, alpha=0.7, zorder=3, linewidths=0)
    p_txt = f"p(Bonf)={p_corrected:.3f} {sig_stars(p_corrected)}" if p_corrected >= 0.001 else "p(Bonf)<0.001 ***"
    ax.set_title(f"{wrap(title, 34)}\n", fontsize=9, fontweight='bold')
    ax.text(0.5, 1.0, f"n={len(pre_v)}, {p_txt}", transform=ax.transAxes, ha='center', va='bottom',
            fontsize=8, color=INK_2)
    ax.set_ylim(0.5, 7.5)
    ax.set_yticks(range(1, 8))
    ax.grid(axis='x', visible=False)


def diverging_likert(ax, data, labels, title):
    """Diverging stacked bar of 1-7 Likert responses, centered on the neutral (4) midpoint."""
    y = np.arange(len(labels))[::-1]
    for yi, vals in zip(y, data):
        vals = vals.dropna().astype(int)
        pct = np.array([(vals == k).mean() * 100 for k in range(1, 8)])
        left = -(pct[:3].sum() + pct[3] / 2)
        for k in range(7):
            if pct[k] > 0:
                ax.barh(yi, pct[k], left=left, color=LIKERT_COLORS[k], edgecolor=SURFACE,
                        linewidth=1.5, height=0.65)
                if pct[k] >= 8:
                    txt_color = INK if k in (2, 3, 4) else '#ffffff'
                    ax.text(left + pct[k] / 2, yi, f"{pct[k]:.0f}", ha='center', va='center',
                            fontsize=7, color=txt_color)
            left += pct[k]
        ax.text(102, yi, f"n={len(vals)}, M={vals.mean():.2f}", va='center', fontsize=8, color=INK_2)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap(lbl, 45) for lbl in labels], fontsize=8)
    ax.set_xlim(-100, 100)
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_xticklabels(['100%', '50%', '0', '50%', '100%'])
    ax.set_xlabel('% of respondents (disagree ← → agree)')
    ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
    ax.grid(axis='y', visible=False)


def likert_legend(fig):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in LIKERT_COLORS]
    names = ['1 Strongly disagree', '2', '3', '4 Neutral', '5', '6', '7 Strongly agree']
    fig.legend(handles, names, loc='lower center', ncol=7, fontsize=8, bbox_to_anchor=(0.5, -0.01))


def slug(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


# ─── Analysis ─────────────────────────────────────────────────────────────────

def run(data_dir, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    pre = pd.read_csv(os.path.join(data_dir, f"{prefix}_pre_paired.csv"), index_col=0)
    post = pd.read_csv(os.path.join(data_dir, f"{prefix}_post_paired.csv"), index_col=0)
    pre_all = pd.read_csv(os.path.join(data_dir, f"{prefix}_pre_processed.csv"), index_col=0)
    post_all = pd.read_csv(os.path.join(data_dir, f"{prefix}_post_processed.csv"), index_col=0)

    # Align by participant, never by row position
    pre = pre.set_index('email')
    post = post.set_index('email').loc[pre.index]
    N = len(pre)
    print(f"Paired sample size: N = {N}  (all consented: pre={len(pre_all)}, post={len(post_all)})\n")

    # 1. Paired Wilcoxon signed-rank tests
    print("=" * 80)
    print("PAIRED WILCOXON SIGNED-RANK TESTS (Pre vs Post)")
    print("=" * 80)
    n_tests = len(PAIRED_ITEMS)
    results, paired_values = [], {}
    for label, (pre_col, post_col) in PAIRED_ITEMS.items():
        pre_vals = pd.to_numeric(pre[pre_col], errors='coerce')
        post_vals = pd.to_numeric(post[post_col], errors='coerce')
        mask = pre_vals.notna() & post_vals.notna()
        pre_v, post_v = pre_vals[mask].values, post_vals[mask].values
        paired_values[label] = (pre_v, post_v)
        diff = post_v - pre_v

        if np.any(diff != 0):
            stat_w, p_val = stats.wilcoxon(pre_v, post_v, alternative='two-sided')
        else:
            stat_w, p_val = np.nan, 1.0
        r = rank_biserial(pre_v, post_v)
        p_corrected = min(p_val * n_tests, 1.0)

        results.append({
            'Variable': label,
            'n': len(pre_v),
            'Pre Mean': round(np.mean(pre_v), 2),
            'Pre SD': round(np.std(pre_v, ddof=1), 2),
            'Pre Median': np.median(pre_v),
            'Post Mean': round(np.mean(post_v), 2),
            'Post SD': round(np.std(post_v, ddof=1), 2),
            'Post Median': np.median(post_v),
            'Mean Diff': round(np.mean(diff), 2),
            'W': stat_w,
            'p-value': round(p_val, 4),
            'p (Bonferroni)': round(p_corrected, 4),
            'Effect r': round(r, 3),
            'Effect Size': effect_size_label(r),
        })
        print(f"{label:40s} n={len(pre_v):2d}  pre M={np.mean(pre_v):.2f}  post M={np.mean(post_v):.2f}  "
              f"diff={np.mean(diff):+.2f}  p(Bonf)={p_corrected:.4f} {sig_stars(p_corrected):3s} r={r:+.2f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out_dir, "paired_wilcoxon_results.csv"), index=False)

    # 2. Composite scores
    print("\n" + "=" * 80)
    print("COMPOSITE SCORES (paired)")
    print("=" * 80)
    composite_results, composite_values = [], {}
    for prefix_key, cat_name in CATEGORIES.items():
        items = {k: v for k, v in PAIRED_ITEMS.items() if k.startswith(prefix_key + ':')}
        pre_items = pd.concat([pd.to_numeric(pre[c], errors='coerce').rename(k) for k, (c, _) in items.items()], axis=1)
        post_items = pd.concat([pd.to_numeric(post[c], errors='coerce').rename(k) for k, (_, c) in items.items()], axis=1)
        pre_c, post_c = pre_items.mean(axis=1), post_items.mean(axis=1)
        mask = pre_c.notna() & post_c.notna()
        pre_c, post_c = pre_c[mask].values, post_c[mask].values
        composite_values[cat_name] = (pre_c, post_c)

        stat_w, p_val = stats.wilcoxon(pre_c, post_c) if np.any(post_c != pre_c) else (np.nan, 1.0)
        r = rank_biserial(pre_c, post_c)
        pre_alpha, post_alpha = cronbach_alpha(pre_items), cronbach_alpha(post_items)
        composite_results.append({
            'Category': cat_name,
            'n_items': len(items),
            'n': len(pre_c),
            'Pre Mean': round(np.mean(pre_c), 2),
            'Post Mean': round(np.mean(post_c), 2),
            'Mean Diff': round(np.mean(post_c - pre_c), 2),
            'W': stat_w,
            'p-value': round(p_val, 4),
            'Effect r': round(r, 3),
            'Effect Size': effect_size_label(r),
            'Cronbach alpha (pre)': round(pre_alpha, 3),
            'Cronbach alpha (post)': round(post_alpha, 3),
        })
        print(f"{cat_name:15s} ({len(items)} items) alpha pre={pre_alpha:.2f} post={post_alpha:.2f}  "
              f"pre M={np.mean(pre_c):.2f} post M={np.mean(post_c):.2f}  p={p_val:.4f} {sig_stars(p_val)} r={r:+.2f}")
    pd.DataFrame(composite_results).to_csv(os.path.join(out_dir, "composite_results.csv"), index=False)

    # 3. Unpaired comparison: all consenting respondents (Mann-Whitney U)
    print("\n" + "=" * 80)
    print("UNPAIRED COMPARISON — all consenting respondents (Mann-Whitney U)")
    print("=" * 80)
    unpaired_results, unpaired_values = [], {}
    for label, (pre_col, post_col) in PAIRED_ITEMS.items():
        pre_v = pd.to_numeric(pre_all[pre_col], errors='coerce').dropna().values
        post_v = pd.to_numeric(post_all[post_col], errors='coerce').dropna().values
        unpaired_values[label] = (pre_v, post_v)
        stat_u, p_val = stats.mannwhitneyu(pre_v, post_v, alternative='two-sided')
        p_corrected = min(p_val * n_tests, 1.0)
        unpaired_results.append({
            'Variable': label,
            'n pre': len(pre_v), 'n post': len(post_v),
            'Pre Mean': round(pre_v.mean(), 2), 'Post Mean': round(post_v.mean(), 2),
            'Pre Median': np.median(pre_v), 'Post Median': np.median(post_v),
            'U': stat_u, 'p-value': round(p_val, 4), 'p (Bonferroni)': round(p_corrected, 4),
        })
        print(f"{label:40s} pre n={len(pre_v):3d} M={pre_v.mean():.2f}  post n={len(post_v):2d} M={post_v.mean():.2f}  "
              f"p(Bonf)={p_corrected:.4f} {sig_stars(p_corrected)}")
    pd.DataFrame(unpaired_results).to_csv(os.path.join(out_dir, "unpaired_mannwhitney_results.csv"), index=False)

    # 4. Pre-only / post-only descriptives
    def describe(df, items):
        rows = []
        for label, col in items.items():
            v = pd.to_numeric(df[col], errors='coerce').dropna()
            rows.append({'Variable': label, 'n': len(v), 'Mean': round(v.mean(), 2), 'SD': round(v.std(ddof=1), 2),
                         'Median': v.median(), 'Min': v.min(), 'Max': v.max()})
        return pd.DataFrame(rows)
    describe(pre_all, PRE_ONLY_ITEMS).to_csv(os.path.join(out_dir, "pre_only_descriptives.csv"), index=False)
    describe(post_all, POST_ONLY_ITEMS).to_csv(os.path.join(out_dir, "post_only_descriptives.csv"), index=False)

    # ─── Visualizations ───────────────────────────────────────────────────────
    saved = []

    def save(fig, name):
        fig.savefig(os.path.join(out_dir, name), dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved.append(name)

    def legend_pre_post(fig, y=1.0):
        handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.6) for c in (PRE_COLOR, POST_COLOR)]
        fig.legend(handles, ['Pre', 'Post'], loc='upper right', ncol=2, bbox_to_anchor=(0.99, y))

    # 4a. Paired box plots per item
    ncols = 5
    nrows = int(np.ceil(len(PAIRED_ITEMS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.3 * nrows))
    axes = axes.flatten()
    for ax, row in zip(axes, results):
        pre_v, post_v = paired_values[row['Variable']]
        paired_box(ax, pre_v, post_v, row['Variable'], row['p (Bonferroni)'], rng)
    for ax in axes[len(PAIRED_ITEMS):]:
        ax.set_visible(False)
    fig.suptitle('Pre vs Post — paired participants (Wilcoxon signed-rank, Bonferroni-corrected)\n'
                 'K = familiarity, S = self-efficacy, I = interest; † = wording differs slightly pre/post; '
                 'lines connect each participant',
                 fontsize=12, y=1.02)
    legend_pre_post(fig, 1.04)
    fig.tight_layout()
    save(fig, "paired_boxplots.png")

    # 4b. Composite box plots
    fig, axes = plt.subplots(1, len(composite_values), figsize=(4.5 * len(composite_values), 4.6))
    for ax, row in zip(axes, composite_results):
        pre_c, post_c = composite_values[row['Category']]
        p_bonf = min(row['p-value'] * len(composite_results), 1.0)
        paired_box(ax, pre_c, post_c, f"{row['Category']} composite ({row['n_items']} items)", p_bonf, rng)
    fig.suptitle('Composite scores — paired participants (mean of items, 1–7)', fontsize=12, y=1.04)
    legend_pre_post(fig, 1.06)
    fig.tight_layout()
    save(fig, "composite_boxplots.png")

    # 4c. Dumbbell: mean change per item
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(results) + 1.5))
    y = np.arange(len(results))[::-1]
    for yi, row in zip(y, results):
        ax.plot([row['Pre Mean'], row['Post Mean']], [yi, yi], color=AXIS, linewidth=2, zorder=1)
        ax.scatter(row['Pre Mean'], yi, s=60, color=PRE_COLOR, zorder=3, edgecolors=SURFACE, linewidths=1.5)
        ax.scatter(row['Post Mean'], yi, s=60, color=POST_COLOR, zorder=3, edgecolors=SURFACE, linewidths=1.5)
        ax.text(max(row['Pre Mean'], row['Post Mean']) + 0.12, yi,
                f"{row['Mean Diff']:+.2f} {sig_stars(row['p (Bonferroni)'])}", va='center', fontsize=8, color=INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels([r['Variable'] for r in results], fontsize=9)
    ax.set_xlim(1, 7.6)
    ax.set_xlabel('Mean score (1–7 Likert)')
    ax.set_title(f'Mean change per item, paired participants (n={N})\n* p<.05, ** p<.01, *** p<.001 (Bonferroni)',
                 fontsize=11, loc='left')
    ax.grid(axis='y', visible=False)
    ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', color=c, markersize=8) for c in (PRE_COLOR, POST_COLOR)],
              labels=['Pre', 'Post'], loc='lower right')
    fig.tight_layout()
    save(fig, "mean_change_dumbbell.png")

    # 4d. Unpaired box plots: all consenting respondents
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.3 * nrows))
    axes = axes.flatten()
    for ax, row in zip(axes, unpaired_results):
        pre_v, post_v = unpaired_values[row['Variable']]
        bp = ax.boxplot([pre_v, post_v], positions=[0, 1], widths=0.45, patch_artist=True,
                        showfliers=False, tick_labels=[f"Pre\nn={len(pre_v)}", f"Post\nn={len(post_v)}"])
        style_box(bp, [PRE_COLOR, POST_COLOR])
        for pos, vals, c in ((0, pre_v, PRE_COLOR), (1, post_v, POST_COLOR)):
            ax.scatter(pos + rng.uniform(-0.14, 0.14, len(vals)), vals + rng.uniform(-0.12, 0.12, len(vals)),
                       s=8, color=c, alpha=0.45, linewidths=0, zorder=3)
        p = row['p (Bonferroni)']
        ax.set_title(f"{wrap(row['Variable'], 34)}\n", fontsize=9, fontweight='bold')
        ax.text(0.5, 1.0, f"p(Bonf)={p:.3f} {sig_stars(p)}" if p >= 0.001 else "p(Bonf)<0.001 ***",
                transform=ax.transAxes, ha='center', va='bottom', fontsize=8, color=INK_2)
        ax.set_ylim(0.5, 7.5)
        ax.set_yticks(range(1, 8))
        ax.grid(axis='x', visible=False)
    for ax in axes[len(PAIRED_ITEMS):]:
        ax.set_visible(False)
    fig.suptitle('Pre vs Post — ALL consenting respondents, unpaired (Mann-Whitney U, Bonferroni-corrected)',
                 fontsize=12, y=1.02)
    legend_pre_post(fig, 1.04)
    fig.tight_layout()
    save(fig, "unpaired_boxplots.png")

    # 4e. Likert distributions for pre-only and post-only items
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5),
                             gridspec_kw={'height_ratios': [len(PRE_ONLY_ITEMS) + 0.8, len(POST_ONLY_ITEMS) + 0.8]})
    diverging_likert(axes[0], [pre_all[c] for c in PRE_ONLY_ITEMS.values()], list(PRE_ONLY_ITEMS),
                     f'Pre-survey only (all consenting respondents)')
    diverging_likert(axes[1], [post_all[c] for c in POST_ONLY_ITEMS.values()], list(POST_ONLY_ITEMS),
                     f'Post-survey only (all consenting respondents)')
    likert_legend(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "unpaired_likert_distributions.png")

    # 4f. Likert distributions for the paired items, pre vs post side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 0.5 * len(PAIRED_ITEMS) + 2), sharey=True)
    labels = list(PAIRED_ITEMS)
    diverging_likert(axes[0], [pd.Series(paired_values[k][0]) for k in labels], labels, f'Pre (paired, n={N})')
    diverging_likert(axes[1], [pd.Series(paired_values[k][1]) for k in labels], labels, f'Post (paired, n={N})')
    likert_legend(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "paired_likert_distributions.png")

    # 4g. Multi-select asked in both surveys: % of paired participants, pre vs post
    multi_rows = []
    for label, (pre_col, post_col) in PAIRED_MULTI.items():
        pre_share, n_pre = option_shares(pre[pre_col])
        post_share, n_post = option_shares(post[post_col])
        opts = pre_share.add(post_share, fill_value=0).sort_values().index
        pre_share, post_share = pre_share.reindex(opts, fill_value=0), post_share.reindex(opts, fill_value=0)
        for o in opts:
            multi_rows.append({'Question': label, 'Option': o, 'Pre %': round(pre_share[o], 1),
                               'Post %': round(post_share[o], 1), 'n pre': n_pre, 'n post': n_post})

        fig, ax = plt.subplots(figsize=(10, 0.5 * len(opts) + 1.6))
        yy = np.arange(len(opts))
        h = 0.38
        ax.barh(yy + h / 2, pre_share.values, height=h, color=PRE_COLOR, label=f'Pre (n={n_pre})',
                edgecolor=SURFACE, linewidth=1)
        ax.barh(yy - h / 2, post_share.values, height=h, color=POST_COLOR, label=f'Post (n={n_post})',
                edgecolor=SURFACE, linewidth=1)
        for yi, a, b in zip(yy, pre_share.values, post_share.values):
            ax.text(a + 1, yi + h / 2, f"{a:.0f}%", va='center', fontsize=7, color=INK_2)
            ax.text(b + 1, yi - h / 2, f"{b:.0f}%", va='center', fontsize=7, color=INK_2)
        ax.set_yticks(yy)
        ax.set_yticklabels([wrap(o, 50) for o in opts], fontsize=8)
        ax.set_xlim(0, 110)
        ax.set_xlabel('% of paired participants selecting option')
        ax.set_title(f'{label} — pre vs post (select all that apply)', fontsize=11, loc='left')
        ax.grid(axis='y', visible=False)
        ax.legend(loc='lower right')
        fig.tight_layout()
        save(fig, f"multiselect_paired_{slug(label)}.png")
    pd.DataFrame(multi_rows).to_csv(os.path.join(out_dir, "multiselect_paired_shares.csv"), index=False)

    # 4h. Multi-select asked in only one survey: all consenting respondents
    def single_bar_grid(groups, filename, title, color_for):
        n = len(groups)
        heights = []
        shares = []
        for label, (df, col) in groups.items():
            s, n_resp = option_shares(df[col]) if 'multi' in filename else (
                100 * df[col].dropna().astype(str).str.strip().value_counts() / df[col].notna().sum(),
                df[col].notna().sum())
            shares.append((label, s.sort_values(), n_resp))
            heights.append(0.4 * len(s) + 1.2)
        fig, axes = plt.subplots(n, 1, figsize=(10, sum(heights)), gridspec_kw={'height_ratios': heights})
        for ax, (label, s, n_resp) in zip(np.atleast_1d(axes), shares):
            yy = np.arange(len(s))
            ax.barh(yy, s.values, color=color_for(label), height=0.65, edgecolor=SURFACE, linewidth=1)
            for yi, v, cnt in zip(yy, s.values, np.round(s.values * n_resp / 100).astype(int)):
                ax.text(v + 1, yi, f"{v:.0f}% ({cnt})", va='center', fontsize=7, color=INK_2)
            ax.set_yticks(yy)
            ax.set_yticklabels([wrap(o, 55) for o in s.index], fontsize=8)
            ax.set_xlim(0, 112)
            ax.set_title(f'{label} (n={n_resp})', fontsize=10, loc='left', fontweight='bold')
            ax.grid(axis='y', visible=False)
        np.atleast_1d(axes)[-1].set_xlabel('% of respondents')
        fig.suptitle(title, fontsize=12, x=0.01, ha='left', y=1.0)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        save(fig, filename)

    color_for = lambda label: PRE_COLOR if label.startswith('Pre') else POST_COLOR
    single_bar_grid({k: (pre_all, c) for k, c in PRE_ONLY_MULTI.items()} |
                    {k: (post_all, c) for k, c in POST_ONLY_MULTI.items()},
                    "multiselect_unpaired.png",
                    'Multi-select questions asked in only one survey (all consenting respondents)', color_for)

    # 4i. Single-choice categorical questions (demographics and post feedback)
    single_bar_grid({f"Pre: {k}": (pre_all, c) for k, c in PRE_CATEGORICAL.items()} |
                    {f"Post: {k}": (post_all, c) for k, c in POST_CATEGORICAL.items()},
                    "categorical_unpaired.png",
                    'Single-choice questions (all consenting respondents)', color_for)

    # 4j. Age histogram (pre)
    ages = pd.to_numeric(pre_all["What is your age?"], errors='coerce').dropna().astype(int)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    counts = ages.value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=PRE_COLOR, width=0.8, edgecolor=SURFACE, linewidth=1)
    ax.set_xlabel('Age')
    ax.set_ylabel('Respondents')
    ax.set_xticks(counts.index)
    ax.set_title(f'Pre: Age (n={len(ages)}, median={ages.median():.0f})', fontsize=11, loc='left')
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    save(fig, "age_distribution.png")

    print("\nSaved to", out_dir)
    for name in saved:
        print("  ", name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre/post analysis and plots for the Planning workshop survey.")
    parser.add_argument("--data-dir", required=True, help="directory with <prefix>_{pre,post}_{paired,processed}.csv")
    parser.add_argument("--out-dir", required=True, help="directory to write results and figures")
    parser.add_argument("--prefix", default="planning", help="file prefix used by parse_data.py (default: planning)")
    args = parser.parse_args()

    run(args.data_dir, args.out_dir, args.prefix)
