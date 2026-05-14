# lgbm_competition_fixed.py

import re
import gc
import numpy as np
import pandas as pd

from scipy.sparse import hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb


SEED = 42
N_SPLITS = 5


def clean_text(x):
    x = "" if pd.isna(x) else str(x)
    x = x.replace("\n", " ").replace("\r", " ").strip()
    x = re.sub(r"\s+", " ", x)
    return x


def make_binary_numeric(series):
    return (
        series.replace({
            True: 1, False: 0,
            "True": 1, "False": 0,
            "true": 1, "false": 0,
            "YES": 1, "NO": 0,
            "Yes": 1, "No": 0,
            "yes": 1, "no": 0
        })
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype("int8")
    )


def build_basic_features(df):
    df = df.copy()

    df["comment"] = df["comment"].fillna("").astype(str).map(clean_text)

    dt = pd.to_datetime(df["created_date"], errors="coerce", utc=True)
    df["year"] = dt.dt.year.fillna(0).astype("int16")
    df["month"] = dt.dt.month.fillna(0).astype("int8")
    df["day"] = dt.dt.day.fillna(0).astype("int8")
    df["dayofweek"] = dt.dt.dayofweek.fillna(0).astype("int8")
    df["hour"] = dt.dt.hour.fillna(0).astype("int8")
    df["minute"] = dt.dt.minute.fillna(0).astype("int8")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype("float32")
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7).astype("float32")
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7).astype("float32")

    txt = df["comment"]
    txt_len = txt.str.len().replace(0, 1)

    df["comment_len"] = txt.str.len().astype("int32")
    df["word_count"] = txt.str.split().str.len().fillna(0).astype("int32")
    df["unique_word_count"] = txt.apply(lambda s: len(set(s.split()))).astype("int32")
    df["char_count_no_space"] = txt.str.replace(" ", "", regex=False).str.len().astype("int32")

    df["uppercase_count"] = txt.str.count(r"[A-Z]").astype("int32")
    df["lowercase_count"] = txt.str.count(r"[a-z]").astype("int32")
    df["digit_count"] = txt.str.count(r"\d").astype("int32")
    df["punct_count"] = txt.str.count(r"[^\w\s]").astype("int32")
    df["exclamation_count"] = txt.str.count("!").astype("int32")
    df["question_count"] = txt.str.count(r"\?").astype("int32")
    df["quote_count"] = txt.str.count(r"\"").astype("int32")
    df["url_count"] = txt.str.count(r"http[s]?://|www\.").astype("int32")
    df["mention_count"] = txt.str.count(r"@\w+").astype("int32")
    df["has_url"] = (df["url_count"] > 0).astype("int8")

    df["uppercase_ratio"] = (df["uppercase_count"] / txt_len).astype("float32")
    df["digit_ratio"] = (df["digit_count"] / txt_len).astype("float32")
    df["punct_ratio"] = (df["punct_count"] / txt_len).astype("float32")
    df["space_ratio"] = ((df["comment_len"] - df["char_count_no_space"]) / txt_len).astype("float32")
    df["avg_word_len"] = (df["char_count_no_space"] / df["word_count"].replace(0, 1)).astype("float32")
    df["lexical_diversity"] = (df["unique_word_count"] / df["word_count"].replace(0, 1)).astype("float32")

    df["starts_with_capital"] = txt.str.match(r"^[A-Z]").fillna(False).astype("int8")
    df["ends_with_punct"] = txt.str.contains(r"[.!?]\s*$", regex=True).fillna(False).astype("int8")

    df["upvote"] = pd.to_numeric(df["upvote"], errors="coerce").fillna(0).astype("float32")
    df["downvote"] = pd.to_numeric(df["downvote"], errors="coerce").fillna(0).astype("float32")
    df["if_1"] = pd.to_numeric(df["if_1"], errors="coerce").fillna(0).astype("float32")
    df["if_2"] = pd.to_numeric(df["if_2"], errors="coerce").fillna(0).astype("float32")
    df["post_id"] = pd.to_numeric(df["post_id"], errors="coerce").fillna(-1).astype("int64")

    df["vote_sum"] = (df["upvote"] + df["downvote"]).astype("float32")
    df["vote_diff"] = (df["upvote"] - df["downvote"]).astype("float32")
    df["vote_ratio"] = (df["upvote"] / (df["downvote"] + 1)).astype("float32")
    df["downvote_ratio"] = (df["downvote"] / (df["vote_sum"] + 1)).astype("float32")
    df["engagement_log"] = np.log1p(df["vote_sum"]).astype("float32")

    binary_cols = ["emoticon_1", "emoticon_2", "emoticon_3", "race", "religion", "gender", "disability"]
    for c in binary_cols:
        df[c] = make_binary_numeric(df[c])

    df["emoticon_sum"] = (df["emoticon_1"] + df["emoticon_2"] + df["emoticon_3"]).astype("float32")
    df["identity_sum"] = (df["race"] + df["religion"] + df["gender"] + df["disability"]).astype("float32")

    return df


def add_group_features(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    full = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    post_agg = full.groupby("post_id").agg(
        post_comment_count=("comment", "size"),
        post_upvote_mean=("upvote", "mean"),
        post_upvote_std=("upvote", "std"),
        post_downvote_mean=("downvote", "mean"),
        post_vote_sum_mean=("vote_sum", "mean"),
        post_vote_diff_mean=("vote_diff", "mean"),
        post_comment_len_mean=("comment_len", "mean"),
        post_word_count_mean=("word_count", "mean"),
        post_identity_sum_mean=("identity_sum", "mean"),
        post_emoticon_sum_mean=("emoticon_sum", "mean"),
        post_if1_mean=("if_1", "mean"),
        post_if2_mean=("if_2", "mean"),
    ).reset_index()

    for col in post_agg.columns:
        if col != "post_id":
            post_agg[col] = post_agg[col].fillna(0)

    train_df = train_df.merge(post_agg, on="post_id", how="left")
    test_df = test_df.merge(post_agg, on="post_id", how="left")

    rel_cols = [
        ("upvote", "post_upvote_mean", "upvote_vs_post_mean"),
        ("downvote", "post_downvote_mean", "downvote_vs_post_mean"),
        ("vote_sum", "post_vote_sum_mean", "vote_sum_vs_post_mean"),
        ("vote_diff", "post_vote_diff_mean", "vote_diff_vs_post_mean"),
        ("comment_len", "post_comment_len_mean", "comment_len_vs_post_mean"),
        ("word_count", "post_word_count_mean", "word_count_vs_post_mean"),
        ("identity_sum", "post_identity_sum_mean", "identity_sum_vs_post_mean"),
        ("emoticon_sum", "post_emoticon_sum_mean", "emoticon_sum_vs_post_mean"),
        ("if_1", "post_if1_mean", "if1_vs_post_mean"),
        ("if_2", "post_if2_mean", "if2_vs_post_mean"),
    ]

    for raw_col, mean_col, new_col in rel_cols:
        train_df[new_col] = (train_df[raw_col] - train_df[mean_col]).astype("float32")
        test_df[new_col] = (test_df[raw_col] - test_df[mean_col]).astype("float32")

    return train_df, test_df


def add_frequency_features(train_df, test_df, cols):
    train_df = train_df.copy()
    test_df = test_df.copy()

    for c in cols:
        full_vals = pd.concat([train_df[c], test_df[c]], axis=0)
        freq = full_vals.value_counts(dropna=False).to_dict()
        train_df[f"{c}_freq"] = train_df[c].map(freq).fillna(0).astype("float32")
        test_df[f"{c}_freq"] = test_df[c].map(freq).fillna(0).astype("float32")

    return train_df, test_df


def build_text_matrices(train_text, test_text):
    word_vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.98,
        max_features=60000,
        sublinear_tf=True
    )

    char_vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=40000,
        sublinear_tf=True
    )

    Xw_train = word_vec.fit_transform(train_text)
    Xw_test = word_vec.transform(test_text)

    Xc_train = char_vec.fit_transform(train_text)
    Xc_test = char_vec.transform(test_text)

    X_text_train = hstack([Xw_train, Xc_train]).tocsr()
    X_text_test = hstack([Xw_test, Xc_test]).tocsr()

    return X_text_train, X_text_test


def build_svd_features(X_text_train, X_text_test, n_components=256):
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    Xs_train = svd.fit_transform(X_text_train).astype(np.float32)
    Xs_test = svd.transform(X_text_test).astype(np.float32)
    return Xs_train, Xs_test


def build_oof_lr_features(X_text_train, y, X_text_test):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    n_classes = len(np.unique(y))
    oof_lr = np.zeros((X_text_train.shape[0], n_classes), dtype=np.float32)
    test_lr = np.zeros((X_text_test.shape[0], n_classes), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_text_train, y), 1):
        X_tr = X_text_train[tr_idx]
        X_va = X_text_train[va_idx]
        y_tr = y[tr_idx]

        lr = LogisticRegression(
            C=4.0,
            max_iter=3000,
            solver="saga",
            multi_class="multinomial",
            n_jobs=-1,
            random_state=SEED + fold
        )

        lr.fit(X_tr, y_tr)
        oof_lr[va_idx] = lr.predict_proba(X_va)
        test_lr += lr.predict_proba(X_text_test) / N_SPLITS

        print(f"Built OOF LR fold {fold}")

    return oof_lr.astype(np.float32), test_lr.astype(np.float32)


def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("Sample.csv")

    train = build_basic_features(train)
    test = build_basic_features(test)

    train, test = add_group_features(train, test)
    train, test = add_frequency_features(
        train, test,
        cols=["post_id", "if_1", "if_2", "race", "religion", "gender", "disability"]
    )

    text_train = train["comment"].astype(str)
    text_test = test["comment"].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(train["label"])

    X_text_train, X_text_test = build_text_matrices(text_train, text_test)
    X_svd_train, X_svd_test = build_svd_features(X_text_train, X_text_test, n_components=256)
    X_lr_oof_train, X_lr_test = build_oof_lr_features(X_text_train, y, X_text_test)

    feature_cols = [
        "post_id", "emoticon_1", "emoticon_2", "emoticon_3",
        "upvote", "downvote", "if_1", "if_2",
        "race", "religion", "gender", "disability",

        "year", "month", "day", "dayofweek", "hour", "minute", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",

        "comment_len", "word_count", "unique_word_count", "char_count_no_space",
        "uppercase_count", "lowercase_count", "digit_count", "punct_count",
        "exclamation_count", "question_count", "quote_count", "url_count", "mention_count",
        "has_url", "uppercase_ratio", "digit_ratio", "punct_ratio", "space_ratio",
        "avg_word_len", "lexical_diversity", "starts_with_capital", "ends_with_punct",

        "vote_sum", "vote_diff", "vote_ratio", "downvote_ratio", "engagement_log",
        "emoticon_sum", "identity_sum",

        "post_comment_count", "post_upvote_mean", "post_upvote_std",
        "post_downvote_mean", "post_vote_sum_mean", "post_vote_diff_mean",
        "post_comment_len_mean", "post_word_count_mean", "post_identity_sum_mean",
        "post_emoticon_sum_mean", "post_if1_mean", "post_if2_mean",

        "upvote_vs_post_mean", "downvote_vs_post_mean", "vote_sum_vs_post_mean",
        "vote_diff_vs_post_mean", "comment_len_vs_post_mean", "word_count_vs_post_mean",
        "identity_sum_vs_post_mean", "emoticon_sum_vs_post_mean",
        "if1_vs_post_mean", "if2_vs_post_mean",

        "post_id_freq", "if_1_freq", "if_2_freq",
        "race_freq", "religion_freq", "gender_freq", "disability_freq"
    ]

    X_num_train = train[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    X_num_test = test[feature_cols].fillna(0).to_numpy(dtype=np.float32)

    X_train = np.hstack([X_num_train, X_svd_train, X_lr_oof_train]).astype(np.float32)
    X_test = np.hstack([X_num_test, X_svd_test, X_lr_test]).astype(np.float32)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_pred = np.zeros((len(train), len(le.classes_)), dtype=np.float32)
    test_pred = np.zeros((len(test), len(le.classes_)), dtype=np.float32)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(le.classes_),
            learning_rate=0.02,
            n_estimators=5000,
            num_leaves=255,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=2.0,
            random_state=SEED + fold,
            n_jobs=-1,
            force_col_wise=True,
            verbosity=-1
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(250, verbose=False),
                lgb.log_evaluation(200)
            ]
        )

        oof_pred[va_idx] = model.predict_proba(X_va)
        test_pred += model.predict_proba(X_test) / N_SPLITS

        fold_acc = accuracy_score(y_va, np.argmax(oof_pred[va_idx], axis=1))
        fold_scores.append(fold_acc)
        print(f"Fold {fold} accuracy: {fold_acc:.6f}")

        del model, X_tr, X_va, y_tr, y_va
        gc.collect()

    cv_acc = accuracy_score(y, np.argmax(oof_pred, axis=1))
    print("\n==============================")
    print(f"Overall CV accuracy: {cv_acc:.6f}")
    print(f"Fold accuracies: {[round(x, 6) for x in fold_scores]}")
    print("==============================\n")

    sample["label"] = le.inverse_transform(np.argmax(test_pred, axis=1))
    sample.to_csv("submission_lgbm_engineered.csv", index=False)
    print("Saved: submission_lgbm_engineered.csv")


if __name__ == "__main__":
    main()