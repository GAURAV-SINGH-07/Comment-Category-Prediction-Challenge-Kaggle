# lgbm_optimized.py

import re
import gc
import numpy as np
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb


SEED = 42
N_SPLITS = 5


def clean_text(x):
    x = "" if pd.isna(x) else str(x)
    x = x.replace("\n", " ").replace("\r", " ").strip()
    x = re.sub(r"\s+", " ", x)
    return x


def add_features(df):
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

    txt = df["comment"]
    txt_len = txt.str.len().replace(0, 1)

    df["comment_len"] = txt.str.len().astype("int32")
    df["word_count"] = txt.str.split().str.len().fillna(0).astype("int32")
    df["unique_word_count"] = txt.apply(lambda s: len(set(s.split()))).astype("int32")
    df["uppercase_count"] = txt.str.count(r"[A-Z]").astype("int32")
    df["digit_count"] = txt.str.count(r"\d").astype("int32")
    df["punct_count"] = txt.str.count(r"[^\w\s]").astype("int32")
    df["exclamation_count"] = txt.str.count("!").astype("int32")
    df["question_count"] = txt.str.count(r"\?").astype("int32")
    df["uppercase_ratio"] = (df["uppercase_count"] / txt_len).astype("float32")
    df["digit_ratio"] = (df["digit_count"] / txt_len).astype("float32")
    df["punct_ratio"] = (df["punct_count"] / txt_len).astype("float32")
    df["avg_word_len"] = (df["comment_len"] / df["word_count"].replace(0, 1)).astype("float32")
    df["lexical_diversity"] = (df["unique_word_count"] / df["word_count"].replace(0, 1)).astype("float32")

    df["upvote"] = df["upvote"].fillna(0)
    df["downvote"] = df["downvote"].fillna(0)
    df["vote_sum"] = (df["upvote"] + df["downvote"]).astype("float32")
    df["vote_diff"] = (df["upvote"] - df["downvote"]).astype("float32")
    df["vote_ratio"] = (df["upvote"] / (df["downvote"] + 1)).astype("float32")

    cat_cols = ["race", "religion", "gender", "disability"]
    for c in cat_cols:
        df[c] = df[c].fillna("missing").astype(str)

    base_num_cols = [
        "post_id", "emoticon_1", "emoticon_2", "emoticon_3",
        "upvote", "downvote", "if_1", "if_2",
        "year", "month", "day", "dayofweek", "hour", "minute", "is_weekend",
        "comment_len", "word_count", "unique_word_count", "uppercase_count",
        "digit_count", "punct_count", "exclamation_count", "question_count",
        "uppercase_ratio", "digit_ratio", "punct_ratio", "avg_word_len",
        "lexical_diversity", "vote_sum", "vote_diff", "vote_ratio"
    ]

    return df, base_num_cols, cat_cols


def encode_categories(train_df, test_df, cat_cols):
    for c in cat_cols:
        all_vals = pd.concat([train_df[c], test_df[c]], axis=0).astype(str)
        mapping = {v: i for i, v in enumerate(all_vals.unique())}
        train_df[c] = train_df[c].map(mapping).astype("int32")
        test_df[c] = test_df[c].map(mapping).astype("int32")
    return train_df, test_df


def build_text_features(train_text, test_text):
    word_tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.98,
        max_features=40000,
        sublinear_tf=True
    )

    char_tfidf = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=30000,
        sublinear_tf=True
    )

    Xw_train = word_tfidf.fit_transform(train_text)
    Xw_test = word_tfidf.transform(test_text)

    Xc_train = char_tfidf.fit_transform(train_text)
    Xc_test = char_tfidf.transform(test_text)

    X_text_train = hstack([Xw_train, Xc_train]).tocsr()
    X_text_test = hstack([Xw_test, Xc_test]).tocsr()

    svd = TruncatedSVD(n_components=192, random_state=SEED)
    Xs_train = svd.fit_transform(X_text_train).astype(np.float32)
    Xs_test = svd.transform(X_text_test).astype(np.float32)

    return Xs_train, Xs_test


def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    sample = pd.read_csv("Sample.csv")

    train, num_cols, cat_cols = add_features(train)
    test, _, _ = add_features(test)

    train, test = encode_categories(train, test, cat_cols)

    Xs_train, Xs_test = build_text_features(train["comment"], test["comment"])

    X_num_train = train[num_cols + cat_cols].fillna(0).to_numpy(dtype=np.float32)
    X_num_test = test[num_cols + cat_cols].fillna(0).to_numpy(dtype=np.float32)

    X_train = np.hstack([X_num_train, Xs_train]).astype(np.float32)
    X_test = np.hstack([X_num_test, Xs_test]).astype(np.float32)

    le = LabelEncoder()
    y = le.fit_transform(train["label"])

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
            learning_rate=0.03,
            n_estimators=4000,
            num_leaves=127,
            max_depth=-1,
            min_child_samples=25,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.15,
            reg_lambda=1.5,
            random_state=SEED + fold,
            n_jobs=-1
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(200, verbose=False),
                lgb.log_evaluation(0)
            ]
        )

        oof_pred[va_idx] = model.predict_proba(X_va)
        test_pred += model.predict_proba(X_test) / N_SPLITS

        fold_acc = accuracy_score(y_va, np.argmax(oof_pred[va_idx], axis=1))
        fold_scores.append(fold_acc)
        print(f"Fold {fold} accuracy: {fold_acc:.6f}")

        del model, X_tr, X_va, y_tr, y_va
        gc.collect()

    oof_labels = np.argmax(oof_pred, axis=1)
    cv_acc = accuracy_score(y, oof_labels)
    print(f"\nOverall CV accuracy: {cv_acc:.6f}")
    print(f"Fold accuracies: {[round(x, 6) for x in fold_scores]}")

    final_test_labels = le.inverse_transform(np.argmax(test_pred, axis=1))
    sample["label"] = final_test_labels
    sample.to_csv("lgbm_submission.csv", index=False)
    print("Saved: lgbm_submission.csv")


if __name__ == "__main__":
    main()