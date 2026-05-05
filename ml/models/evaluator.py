from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, f1_score, roc_curve, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize

POSITIVE_KEYWORDS = {
    '1', '1.0', 'yes', 'true', 'positive', 'malignant', 'pathological', 'abnormal',
    'ckd', 'disease', 'sick'
}
NEGATIVE_KEYWORDS = {
    '0', '0.0', 'no', 'false', 'negative', 'normal', 'notckd', 'not', 'healthy'
}

def _choose_positive_label(labels: np.ndarray) -> Any:
    default_label = labels[1] if len(labels) > 1 else labels[0]
    chosen = default_label

    for lbl in labels:
        if str(lbl).strip().lower() in POSITIVE_KEYWORDS:
            return lbl

    for lbl in labels:
        norm = str(lbl).strip().lower()
        if norm not in NEGATIVE_KEYWORDS and "not" not in norm and "non" not in norm:
            chosen = lbl
            break

    return chosen

def calculate_metrics(model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics.
    """
    y_train_pred = model.predict(X_train)
    train_acc = float(accuracy_score(y_train, y_train_pred))
    test_acc = float(accuracy_score(y_test, y_pred))
    
    # Cross-validation
    cv_folds = 3 if len(X_train) < 100 else 5
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))

    labels = np.unique(y_test)
    is_multiclass = len(labels) > 2
    sens = spec = prec = f1 = auc_val = 0.0
    tn = fp = fn = tp = 0
    roc_points = []
    pos_label_final = None
    macro_sens = macro_prec = macro_f1 = None

    if len(labels) >= 2:
        # Positive label detection
        pos_label = _choose_positive_label(labels)
        pos_label_final = pos_label
        
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        if pos_label in labels:
            pos_i = int(np.where(labels == pos_label)[0][0])
            tp = int(cm[pos_i, pos_i])
            fn = int(cm[pos_i, :].sum() - cm[pos_i, pos_i])
            fp = int(cm[:, pos_i].sum() - cm[pos_i, pos_i])
            tn = int(cm.sum() - tp - fn - fp)

            # Keep displayed metrics in sync with displayed TP/TN/FP/FN.
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = (2.0 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        if cm.shape != (2, 2):
            macro_sens = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
            macro_prec = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
            macro_f1 = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
            specs: List[float] = []
            for i in range(cm.shape[0]):
                tp_i = float(cm[i, i])
                fn_i = float(cm[i, :].sum() - tp_i)
                fp_i = float(cm[:, i].sum() - tp_i)
                tn_i = float(cm.sum() - tp_i - fn_i - fp_i)
                denom = tn_i + fp_i
                specs.append((tn_i / denom) if denom > 0 else 0.0)
            spec = float(np.mean(specs)) if specs else 0.0
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            model_classes = np.asarray(getattr(model, "classes_", labels))
            if is_multiclass:
                try:
                    y_true_bin = label_binarize(y_test, classes=model_classes)
                    if y_true_bin.shape[1] >= 2:
                        valid_auc = []
                        fpr_grid = np.linspace(0.0, 1.0, 100)
                        mean_tpr = np.zeros_like(fpr_grid)
                        valid_classes_for_roc = 0

                        for i in range(y_true_bin.shape[1]):
                            if len(np.unique(y_true_bin[:, i])) == 2:
                                valid_auc.append(roc_auc_score(y_true_bin[:, i], y_prob[:, i]))
                                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                                mean_tpr += np.interp(fpr_grid, fpr, tpr)
                                valid_classes_for_roc += 1

                        if valid_classes_for_roc > 0:
                            auc_val = float(np.mean(valid_auc))
                            mean_tpr /= valid_classes_for_roc
                            mean_tpr[0] = 0.0
                            mean_tpr[-1] = 1.0
                            roc_points = [{"x": float(f), "y": float(t)} for f, t in zip(fpr_grid, mean_tpr)]
                        else:
                            auc_val = 0.0
                    else:
                        auc_val = 0.0
                except Exception as e:
                    auc_val = 0.0
            elif pos_label in model_classes:
                pos_idx = int(np.where(model_classes == pos_label)[0][0])
                try:
                    y_true_bin = (pd.Series(y_test).astype(str).values == str(pos_label)).astype(int)
                    y_score = np.asarray(y_prob[:, pos_idx]).astype(np.float64)
                    if len(np.unique(y_true_bin)) >= 2:
                        auc_val = float(roc_auc_score(y_true_bin, y_score))
                        if np.isnan(auc_val):
                            auc_val = 0.0
                        fpr, tpr, _ = roc_curve(y_true_bin, y_score, pos_label=1)
                        # Downsample for frontend
                        if len(fpr) > 100:
                            indices = np.linspace(0, len(fpr)-1, 100, dtype=int)
                            fpr, tpr = fpr[indices], tpr[indices]
                        roc_points = [{"x": float(f), "y": float(t)} for f, t in zip(fpr, tpr)]
                    else:
                        auc_val = 0.0
                except:
                    auc_val = 0.0

    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1_score": f1,
        "auc": auc_val,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "roc_points": roc_points,
        "positive_label": pos_label_final,
        "labels": labels.tolist(),
        "macro_sensitivity": macro_sens,
        "macro_precision": macro_prec,
        "macro_f1_score": macro_f1
    }

def diagnose_overfit(metrics: Dict[str, Any], y_test: pd.Series) -> Tuple[bool, bool, str]:
    """
    Diagnose overfitting or data leakage based on metrics.
    """
    train_acc = metrics["train_accuracy"]
    test_acc = metrics["test_accuracy"]
    spec = metrics["specificity"]
    prec = metrics["precision"]
    auc = metrics["auc"]
    cv_std = metrics["cv_std"]

    overfit_suspected = (train_acc - test_acc) > 0.10
    perfect_score = (spec >= 0.99 or prec >= 0.99 or auc >= 0.97)
    
    reasons = []
    if perfect_score:
        perf_details = []
        if spec >= 0.99: perf_details.append(f"Specificity: {spec*100:.0f}%")
        if prec >= 0.99: perf_details.append(f"Precision: {prec*100:.0f}%")
        if auc >= 0.97: perf_details.append(f"AUC: {auc:.2f}")
        
        reasons.append(
            f"Perfect scores detected ({', '.join(perf_details)}). "
            "This is statistically unlikely in real clinical data and may indicate data leakage."
        )
    elif overfit_suspected:
        reasons.append(f"High gap between training ({train_acc*100:.0f}%) and testing ({test_acc*100:.0f}%) accuracy.")
        
    if cv_std > 0.15:
        reasons.append(f"High variance in cross-validation (std: {cv_std:.2f}), suggesting instability.")
        
    if len(y_test) < 30:
        reasons.append(f"Warning: Test set has only {len(y_test)} samples. Results on small test sets are unreliable.")
        
    overfit_reason = " ".join(reasons) if reasons else "Model performance looks normal."
    
    return overfit_suspected, perfect_score, overfit_reason
