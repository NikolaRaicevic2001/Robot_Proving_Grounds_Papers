import argparse
import os

import pandas as pd


def find_consent_column(df):
    """Returns the consent column (first header mentioning consent), falling back to column 1."""
    for col in df.columns:
        if "consent" in col.lower():
            return col
    return df.columns[1]


def process(raw_df):
    """Renames consent/email columns and keeps only rows that explicitly consented."""
    df = raw_df.rename(columns={find_consent_column(raw_df): "consent", "Email Address": "email"})
    df["email"] = df["email"].str.strip().str.lower()

    # Filters for consent (blank consent is treated as no consent)
    return df[df["consent"].str.strip() == "Yes"]


def parse_data(pre_raw_fp, post_raw_fp, out_dir, prefix):
    pre_processed_df = process(pd.read_csv(pre_raw_fp))
    post_processed_df = process(pd.read_csv(post_raw_fp))

    # Removes duplicates and gets last submission
    pre_dedup = pre_processed_df.drop_duplicates(subset=["email"], keep="last")
    post_dedup = post_processed_df.drop_duplicates(subset=["email"], keep="last")

    # Pairs data
    pre_df_filt = pre_dedup[pre_dedup["email"].isin(post_dedup["email"])]
    post_df_filt = post_dedup[post_dedup["email"].isin(pre_dedup["email"])]

    # Sorts both by email so row i of pre and row i of post are the same participant
    pre_df_filt = pre_df_filt.sort_values("email")
    post_df_filt = post_df_filt.sort_values("email")

    assert len(pre_df_filt) == len(post_df_filt), f"length {len(pre_df_filt)} != {len(post_df_filt)}"
    assert (pre_df_filt["email"].values == post_df_filt["email"].values).all(), "pre/post rows are misaligned"

    print(f"Pre:  {len(pre_processed_df)} consented ({len(pre_processed_df) - len(pre_dedup)} duplicate submissions dropped)")
    print(f"Post: {len(post_processed_df)} consented ({len(post_processed_df) - len(post_dedup)} duplicate submissions dropped)")
    print(f"Paired: {len(pre_df_filt)}")

    # Saves data
    os.makedirs(out_dir, exist_ok=True)
    pre_processed_df.to_csv(os.path.join(out_dir, f"{prefix}_pre_processed.csv"))
    post_processed_df.to_csv(os.path.join(out_dir, f"{prefix}_post_processed.csv"))
    pre_df_filt.to_csv(os.path.join(out_dir, f"{prefix}_pre_paired.csv"))
    post_df_filt.to_csv(os.path.join(out_dir, f"{prefix}_post_paired.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter survey responses for consent and pair pre/post responses by email.")
    parser.add_argument("--pre", required=True, help="raw pre-survey CSV")
    parser.add_argument("--post", required=True, help="raw post-survey CSV")
    parser.add_argument("--out-dir", required=True, help="directory to write processed/paired CSVs")
    parser.add_argument("--prefix", default="erl", help="output file prefix (default: erl)")
    args = parser.parse_args()

    parse_data(args.pre, args.post, args.out_dir, args.prefix)
