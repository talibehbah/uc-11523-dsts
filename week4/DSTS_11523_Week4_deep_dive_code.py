import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
np.random.seed(42)

def hdr(t): print("\n" + "="*70 + f"\n### {t}\n" + "="*70)

# ----------------------------------------------------------------------
hdr("0. DATA — a real binary-classification problem")
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target          # 30 features, y: 0=malignant, 1=benign
print("shape:", X.shape, "| positive rate:", round(y.mean(), 3))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print("train:", X_train.shape[0], "test:", X_test.shape[0])

# ----------------------------------------------------------------------
hdr("1. METRICS — beyond accuracy")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
clf = RandomForestClassifier(n_estimators=300, random_state=42).fit(X_train, y_train)
pred  = clf.predict(X_test)
proba = clf.predict_proba(X_test)[:, 1]
print("accuracy :", round(accuracy_score(y_test, pred), 4))
print("precision:", round(precision_score(y_test, pred), 4))
print("recall   :", round(recall_score(y_test, pred), 4))
print("f1       :", round(f1_score(y_test, pred), 4))
print("roc_auc  :", round(roc_auc_score(y_test, proba), 4))
print("confusion matrix [[TN FP][FN TP]]:\n", confusion_matrix(y_test, pred))

# ----------------------------------------------------------------------
hdr("2. CROSS-VALIDATION — a stabler estimate")
from sklearn.model_selection import cross_val_score, StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring="f1")
print("per-fold f1:", np.round(scores, 4))
print(f"mean f1: {scores.mean():.4f}  +/- {scores.std():.4f}")

# ----------------------------------------------------------------------
hdr("3. DATA LEAKAGE — the classic noise-feature trap")
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
# PURE NOISE: 200 samples, 1000 random features, random labels -> no real signal
rng = np.random.RandomState(0)
Xn = rng.randn(200, 1000)
yn = rng.randint(0, 2, 200)
# WRONG: choose the 20 features most correlated with y using ALL the data,
#        THEN cross-validate on just those -> the test folds already leaked in
corr = np.abs([np.corrcoef(Xn[:, j], yn)[0, 1] for j in range(Xn.shape[1])])
top = np.argsort(corr)[-20:]
leaky = cross_val_score(LogisticRegression(max_iter=1000), Xn[:, top], yn, cv=5)
# RIGHT: selection is INSIDE the pipeline, so it is redone on each training fold
pipe = Pipeline([("select", SelectKBest(f_classif, k=20)),
                 ("clf", LogisticRegression(max_iter=1000))])
honest = cross_val_score(pipe, Xn, yn, cv=5)
print(f"leaky  (select on all data, then CV): {leaky.mean():.4f}  <- looks predictive")
print(f"honest (select inside each CV fold) : {honest.mean():.4f}  <- ~chance (0.50)")
print(f"pure illusion created by leakage    : {leaky.mean()-honest.mean():+.4f}")

# ----------------------------------------------------------------------
hdr("4. BIAS-VARIANCE — learning & validation curves")
from sklearn.model_selection import learning_curve, validation_curve
# validation curve over tree depth: see under- vs over-fit
depths = [1, 2, 3, 5, 8, 12, None]
depth_lbl = [str(d) for d in depths]
tr, va = validation_curve(
    RandomForestClassifier(n_estimators=150, random_state=42),
    X, y, param_name="max_depth", param_range=depths, cv=cv, scoring="f1")
print("depth :", depth_lbl)
print("train :", np.round(tr.mean(1), 3))
print("valid :", np.round(va.mean(1), 3))
gap = tr.mean(1) - va.mean(1)
print("gap   :", np.round(gap, 3), " <- large gap = variance/overfit")

# ----------------------------------------------------------------------
hdr("5. HYPERPARAMETER TUNING — GridSearchCV & RandomizedSearchCV")
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={"n_estimators":[100,300], "max_depth":[3,5,None],
                "min_samples_leaf":[1,3]},
    cv=cv, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)
print("GridSearch best_params_:", grid.best_params_)
print("GridSearch best_score_ :", round(grid.best_score_, 4))
print("held-out test f1       :", round(f1_score(y_test, grid.predict(X_test)), 4))

rnd = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions={"n_estimators":randint(100,400),
                         "max_depth":randint(2,15),
                         "min_samples_leaf":randint(1,6)},
    n_iter=15, cv=cv, scoring="f1", random_state=42, n_jobs=-1)
rnd.fit(X_train, y_train)
print("RandomizedSearch best   :", rnd.best_params_, "-> f1", round(rnd.best_score_,4))

# ----------------------------------------------------------------------
hdr("6. ABLATION — drop each feature GROUP, measure the drop")
groups = {"mean":[c for c in X.columns if c.startswith("mean")],
          "error":[c for c in X.columns if "error" in c],
          "worst":[c for c in X.columns if c.startswith("worst")]}
base = cross_val_score(clf, X, y, cv=cv, scoring="f1").mean()
print(f"full model f1 = {base:.4f}")
rows=[]
for name, cols in groups.items():
    s = cross_val_score(clf, X.drop(columns=cols), y, cv=cv, scoring="f1").mean()
    rows.append((name, len(cols), round(s,4), round(base-s,4)))
for name, n, s, d in sorted(rows, key=lambda r:-r[3]):
    print(f"  - {name:6s} ({n:2d} feats): f1={s:.4f}   drop={d:+.4f}")

# ----------------------------------------------------------------------
hdr("7. SLICE-BASED ERROR ANALYSIS — where does it fail?")
res = X_test.copy()
res["y_true"] = y_test.values
res["y_pred"] = pred
# slice by a clinically meaningful feature: tumour size (mean radius) buckets
res["size_bucket"] = pd.qcut(res["mean radius"], q=4,
                             labels=["small","medium","large","x-large"])
overall = (res.y_true == res.y_pred).mean()
print(f"overall accuracy = {overall:.4f}")
per = res.groupby("size_bucket").apply(
    lambda g: pd.Series({"n":len(g),
                         "accuracy":round((g.y_true==g.y_pred).mean(),4),
                         "recall":round(recall_score(g.y_true,g.y_pred,zero_division=0),4)}))
print(per)

# ----------------------------------------------------------------------
hdr("8. MLflow — track every experiment, then compare")
import mlflow, mlflow.sklearn, os, shutil
shutil.rmtree("/home/claude/dd/mlartifacts", ignore_errors=True)
for f in ["/home/claude/dd/mlflow.db"]:
    if os.path.exists(f): os.remove(f)
mlflow.set_tracking_uri("sqlite:////home/claude/dd/mlflow.db")   # MLflow 3.x backend
mlflow.set_experiment("bc-classifier")
configs = [dict(n_estimators=100, max_depth=3),
           dict(n_estimators=300, max_depth=5),
           dict(n_estimators=300, max_depth=None)]
for cfg in configs:
    with mlflow.start_run(run_name=f"rf_d{cfg['max_depth']}"):
        m = RandomForestClassifier(random_state=42, **cfg).fit(X_train, y_train)
        f1 = f1_score(y_test, m.predict(X_test))
        auc = roc_auc_score(y_test, m.predict_proba(X_test)[:,1])
        mlflow.log_params(cfg)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(m, name="model")
# query the tracked runs back
runs = mlflow.search_runs(experiment_names=["bc-classifier"],
                          order_by=["metrics.f1 DESC"])
cols = ["params.n_estimators","params.max_depth","metrics.f1","metrics.roc_auc"]
print(runs[cols].round(4).to_string(index=False))
best = runs.iloc[0]
print(f"\nbest run id: {best.run_id[:8]}...  f1={best['metrics.f1']:.4f}")

print("\nALL SECTIONS COMPLETE")
