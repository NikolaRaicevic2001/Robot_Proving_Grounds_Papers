import pandas as pd

POST_RAW_FP = "data/erl_post_raw.csv"
PRE_RAW_FP  = "data/erl_pre_raw.csv"

# Said yes to consent
PRE_PROCESSED_FP = "data/erl_pre_processed.csv"
POST_PROCESSED_FP = "data/erl_post_processed.csv"


# Has pre and post response
PRE_PAIRED_FP  = "data/erl_pre_paired.csv"
POST_PAIRED_FP  = "data/erl_post_paired.csv"

def parse_data():
    pre_raw_df = pd.read_csv(PRE_RAW_FP)
    post_raw_df = pd.read_csv(POST_RAW_FP)

    # Rename to "consent"
    pre_processed_df = pre_raw_df.rename(columns={pre_raw_df.columns[1]: "consent"})
    post_processed_df = post_raw_df.rename(columns={post_raw_df.columns[1]: "consent"})

    # Rename email address to "email"
    post_processed_df.rename(columns={"Email Address": "email"}, inplace=True)
    pre_processed_df.rename(columns={"Email Address": "email"}, inplace=True)

    # Filters for consent
    pre_processed_df = pre_processed_df[pre_processed_df["consent"] != "No"]
    post_processed_df = post_processed_df[post_processed_df["consent"] != "No"]

    # Pairs data
    pre_df_filt = pre_processed_df[pre_processed_df["email"].isin(post_processed_df["email"])]
    post_df_filt = post_processed_df[post_processed_df["email"].isin(pre_processed_df["email"])]

    # Removes duplicates and gets last submission
    mask = pre_df_filt.duplicated(subset=["email"], keep="last")
    pre_df_filt = pre_df_filt[~mask]


    assert len(pre_df_filt) == len(post_df_filt), f"length {len(pre_df_filt)} != {len(post_df_filt)}"

    # Saves data
    post_df_filt.to_csv(POST_PAIRED_FP)
    pre_df_filt.to_csv(PRE_PAIRED_FP)
    pre_processed_df.to_csv(PRE_PROCESSED_FP)
    post_processed_df.to_csv(POST_PROCESSED_FP)

if __name__ == "__main__":
    parse_data()