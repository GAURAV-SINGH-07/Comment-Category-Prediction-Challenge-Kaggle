# ============================================================
# COMMENT CATEGORY PREDICTION - OPTIMIZED LOG REG MODEL
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import re
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

SEED = 42

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SUB_PATH = "Sample.csv"

# ============================================================
# 1. LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SUB_PATH)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def add_features(df, post_map=None):

    df = df.copy()

    df["comment"] = df["comment"].fillna("").apply(clean_text)

    for col in ["race","religion","gender"]:
        df[col] = df[col].fillna("missing").astype(str)

    df["disability"] = df["disability"].fillna(False).astype(int)

    dt = pd.to_datetime(df["created_date"], errors="coerce")

    df["year"] = dt.dt.year.fillna(-1)
    df["month"] = dt.dt.month.fillna(-1)
    df["day"] = dt.dt.day.fillna(-1)
    df["dayofweek"] = dt.dt.dayofweek.fillna(-1)
    df["hour"] = dt.dt.hour.fillna(-1)

    df["comment_len"] = df["comment"].str.len()
    df["word_count"] = df["comment"].str.split().str.len()

    df["uppercase_count"] = df["comment"].str.count(r"[A-Z]")
    df["punct_count"] = df["comment"].str.count(r"[^\w\s]")

    df["upvote"] = df["upvote"].fillna(0)
    df["downvote"] = df["downvote"].fillna(0)

    df["net_votes"] = df["upvote"] - df["downvote"]
    df["total_votes"] = df["upvote"] + df["downvote"]

    df["if_1"] = df["if_1"].fillna(0)
    df["if_2"] = df["if_2"].fillna(0)

    df["if_sum"] = df["if_1"] + df["if_2"]

    if post_map is None:
        post_map = df["post_id"].value_counts().to_dict()

    df["post_comment_count"] = df["post_id"].map(post_map).fillna(1)

    return df, post_map

train_fe, post_map = add_features(train_df)
test_fe, _ = add_features(test_df, post_map)

TEXT_COL = "comment"

CAT_COLS = ["race","religion","gender"]

NUM_COLS = [
"emoticon_1","emoticon_2","emoticon_3",
"upvote","downvote","net_votes","total_votes",
"if_1","if_2","if_sum",
"year","month","day","dayofweek","hour",
"comment_len","word_count",
"uppercase_count","punct_count",
"post_comment_count"
]

# ============================================================
# 3. FEATURE BUILDER
# ============================================================

class SparseBuilder():

    def __init__(self):

        self.word = TfidfVectorizer(
            ngram_range=(1,2),
            min_df=3,
            max_df=0.98,
            max_features=60000,
            sublinear_tf=True
        )

        self.char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3,5),
            max_features=40000,
            min_df=3
        )

        self.ohe = OneHotEncoder(handle_unknown="ignore")

        self.imp = SimpleImputer(strategy="median")

        self.scaler = StandardScaler(with_mean=False)

    def fit(self,df):

        self.word.fit(df[TEXT_COL])
        self.char.fit(df[TEXT_COL])

        self.ohe.fit(df[CAT_COLS])

        num = self.imp.fit_transform(df[NUM_COLS])
        num = sparse.csr_matrix(num)

        self.scaler.fit(num)

    def transform(self,df):

        X1 = self.word.transform(df[TEXT_COL])
        X2 = self.char.transform(df[TEXT_COL])

        Xcat = self.ohe.transform(df[CAT_COLS])

        num = self.imp.transform(df[NUM_COLS])
        num = sparse.csr_matrix(num)
        num = self.scaler.transform(num)

        return sparse.hstack([X1,X2,Xcat,num]).tocsr()

# ============================================================
# 4. GROUP SPLIT VALIDATION
# ============================================================

gss = GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=SEED)

train_idx,valid_idx = next(
gss.split(train_fe,train_fe["label"],groups=train_fe["post_id"])
)

tr_df = train_fe.iloc[train_idx]
va_df = train_fe.iloc[valid_idx]

print("Train split:",tr_df.shape)
print("Valid split:",va_df.shape)

builder = SparseBuilder()
builder.fit(tr_df)

X_tr = builder.transform(tr_df)
X_va = builder.transform(va_df)

y_tr = tr_df["label"].values
y_va = va_df["label"].values

# ============================================================
# 5. TRAIN OPTIMIZED LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(
C=1.0,
solver="saga",
max_iter=2500,
n_jobs=-1
)

model.fit(X_tr,y_tr)

train_preds = model.predict(X_tr)
valid_preds = model.predict(X_va)

train_acc = accuracy_score(y_tr,train_preds)
valid_acc = accuracy_score(y_va,valid_preds)

print("\n==============================")
print("TRAIN ACCURACY :",train_acc)
print("VALID ACCURACY :",valid_acc)
print("==============================\n")

print(classification_report(y_va,valid_preds))

# ============================================================
# 6. TRAIN FINAL MODEL ON FULL DATA
# ============================================================

builder_final = SparseBuilder()
builder_final.fit(train_fe)

X_train_full = builder_final.transform(train_fe)
X_test_full = builder_final.transform(test_fe)

y_train_full = train_fe["label"].values

final_model = LogisticRegression(
C=1.0,
solver="saga",
max_iter=2500,
n_jobs=-1
)

final_model.fit(X_train_full,y_train_full)

# ============================================================
# 7. CREATE SUBMISSION
# ============================================================

test_preds = final_model.predict(X_test_full)

submission = sample_sub.copy()

submission["label"] = test_preds

submission.to_csv("submission_logreg_optimized.csv",index=False)

print("Submission file created: submission_logreg_optimized.csv")
print(submission.head())