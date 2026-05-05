from typing import List, Dict, Any, Set
import pandas as pd
import numpy as np

def detect_outliers_iqr(df: pd.DataFrame, num_cols: List[str]) -> Dict[str, Any]:
    """
    Detect outliers using the IQR method (1.5 * IQR rule).
    """
    if df.empty:
        return {"total_count": 0, "outliers_found": 0, "percentage": 0, "details": []}
    
    # Ensure columns are numeric
    valid_cols = [c for c in num_cols if c in df.columns]
    for col in valid_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    outlier_indices = set()
    details = []
    total_rows = len(df)
    row_hit_counts: Dict[int, int] = {}
    # Use a slightly wider fence than classical 1.5*IQR to reduce false positives
    # on noisy clinical datasets.
    iqr_multiplier = 2.0
    # In wide datasets, a row should be extreme in at least 2 numeric columns
    # before being considered an outlier row.
    min_columns_for_row_outlier = 2 if len(valid_cols) >= 6 else 1
    
    for col in valid_cols:
        series = df[col].dropna()
        if series.empty: continue
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        if pd.isna(IQR) or IQR == 0:
            # Constant / near-constant columns should not mark outliers.
            continue
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
        col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        count = len(col_outliers)
        
        if count > 0:
            for idx in col_outliers.index.tolist():
                row_hit_counts[idx] = row_hit_counts.get(idx, 0) + 1
            details.append({
                "column": col,
                "count": count,
                "min": float(series.min()),
                "max": float(series.max()),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound)
            })

    outlier_indices = {
        idx for idx, hits in row_hit_counts.items()
        if hits >= min_columns_for_row_outlier
    }

    return {
        "total_count": total_rows,
        "outliers_found": len(outlier_indices),
        "percentage": round((len(outlier_indices) / total_rows) * 100, 1) if total_rows > 0 else 0,
        "details": details,
        "outlier_indices": list(outlier_indices)
    }

def filter_outliers(df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    """
    Remove rows containing outliers based on IQR.
    """
    result = detect_outliers_iqr(df, num_cols)
    indices_to_drop = result.get("outlier_indices", [])
    if indices_to_drop:
        return df.drop(index=indices_to_drop)
    return df
