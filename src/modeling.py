#modeling
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def multiple_linear_regression(
    df: pd.DataFrame, outcome: str, predictors: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    TODO (Student task): Fit a multiple linear regression model.

    Requirements:
    - Outcome must be numeric; raise ValueError otherwise
    - If predictors is None:
        use ALL numeric columns except outcome
    - Drop rows with missing values in outcome or predictors before fitting
    - Fit the model using least squares:
        y = intercept + b1*x1 + b2*x2 + ...
    - Return a JSON-safe dictionary containing:
        outcome, predictors, n_rows_used, r_squared, adj_r_squared,
        intercept, coefficients (dict)

    Hints: use statsmodels package:
    import statsmodels.api as sm
    X = df[predictors]
    X = sm.add_constant(X)
    y = df[outcome]
    model = sm.OLS(y, X).fit()

    IMPORTANT:
    - Convert any numpy/pandas scalars to Python floats/ints before returning.
    """
    import statsmodels.api as sm
    
    # Check if outcome is numeric
    if not pd.api.types.is_numeric_dtype(df[outcome]):
        raise ValueError(f"Outcome column '{outcome}' must be numeric")
    
    # If predictors is None, use all numeric columns except outcome
    if predictors is None:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        predictors = [col for col in numeric_cols if col != outcome]
    
    # Drop rows with missing values
    model_df = df[[outcome] + predictors].dropna()
    n_rows_used = len(model_df)
    
    # Prepare X and y
    X = model_df[predictors]
    X = sm.add_constant(X)
    y = model_df[outcome]
    
    # Fit the model
    model = sm.OLS(y, X).fit()
    
    # Create coefficients dictionary (excluding intercept)
    coefficients = {}
    for pred in predictors:
        # Convert to Python float
        coefficients[pred] = float(model.params[pred])
    
    # Return JSON-safe dictionary
    return {
        "outcome": outcome,
        "predictors": predictors,
        "n_rows_used": int(n_rows_used),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "intercept": float(model.params["const"]),
        "coefficients": coefficients
    }
