import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from pingouin import cronbach_alpha
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple, Union, Optional, Dict, Any
from pandas.api.types import is_numeric_dtype

# Configure logging
logging.basicConfig(
    filename='app.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@st.cache_data
def handle_missing_values(data: pd.DataFrame, method: str = 'drop') -> pd.DataFrame:
    """
    Handle missing values in the data
    
    Parameters:
        data: Input DataFrame
        method: Method for handling missing values ('drop', 'mean', 'median', 'mode')
        
    Returns:
        DataFrame with missing values handled
    """
    try:
        if data.empty:
            logging.warning("Empty DataFrame passed to handle_missing_values")
            return data
            
        if method == 'drop':
            return data.dropna()
        elif method == 'mean':
            return data.fillna(data.mean(numeric_only=True))
        elif method == 'median':
            return data.fillna(data.median(numeric_only=True))
        elif method == 'mode':
            mode_val = data.mode()
            if not mode_val.empty:
                return data.fillna(mode_val.iloc[0])
            return data.fillna(0)
        return data
    except Exception as e:
        logging.error(f"Error in handle_missing_values with method {method}: {str(e)}")
        raise

@st.cache_data
def remove_outliers(data: pd.DataFrame, method: str = 'zscore', threshold: float = 3) -> pd.DataFrame:
    """
    Remove outliers from the data
    
    Parameters:
        data: Input DataFrame
        method: Method for outlier detection ('zscore' or 'iqr')
        threshold: Threshold value for outlier detection
        
    Returns:
        DataFrame with outliers removed
    """
    try:
        if len(data) < 2:
            return data
            
        if data.nunique().eq(1).any():  # Check for constant columns
            return data
            
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
            return data[(z_scores < threshold).all(axis=1)]
        elif method == 'iqr':
            numeric_data = data.select_dtypes(include=['number'])
            q1 = numeric_data.quantile(0.25)
            q3 = numeric_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            return data[((numeric_data > lower_bound) & (numeric_data < upper_bound)).all(axis=1)]
        return data
    except Exception as e:
        logging.error(f"Error in remove_outliers with method {method}: {str(e)}")
        raise

@st.cache_data
def calculate_cronbach_alpha(data: pd.DataFrame) -> Optional[float]:
    """
    Calculate Cronbach's alpha for scale reliability
    
    Parameters:
        data: Input DataFrame (items as columns, observations as rows)
        
    Returns:
        Cronbach's alpha value or None if calculation fails
    """
    try:
        if data.empty or len(data.columns) < 2:
            logging.warning("Insufficient data for Cronbach's alpha calculation")
            return None
            
        result = cronbach_alpha(data)
        return result[0]
    except Exception as e:
        logging.error(f"Error calculating Cronbach's alpha: {str(e)}")
        return None

@st.cache_data
def parallel_analysis(
    data: pd.DataFrame, 
    method: str = 'PCA', 
    n_iter: int = 20, 
    random_state: Optional[int] = None
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Perform parallel analysis for factor retention
    
    Parameters:
        data: Input DataFrame
        method: Factor extraction method ('PCA' or 'minres')
        n_iter: Number of iterations for random data generation
        random_state: Random state for reproducibility
        
    Returns:
        Tuple containing:
        - Number of factors to retain
        - Original eigenvalues
        - Mean eigenvalues from random data
    """
    try:
        if random_state is not None:
            np.random.seed(random_state)
            
        n_rows, n_cols = data.shape
        if n_rows < 10 or n_cols < 2:
            raise ValueError("Insufficient data for parallel analysis")
            
        eig_orig = FactorAnalyzer(
            n_factors=n_cols, 
            method='principal' if method=='PCA' else 'minres'
        ).fit(data).get_eigenvalues()[0]
        
        eig_rand = np.zeros((n_iter, n_cols))
        for i in range(n_iter):
            rand_data = np.random.normal(size=(n_rows, n_cols))
            eig_rand[i] = FactorAnalyzer(
                n_factors=n_cols, 
                method='principal' if method=='PCA' else 'minres'
            ).fit(rand_data).get_eigenvalues()[0]
        
        eig_rand_mean = eig_rand.mean(axis=0)
        n_factors = sum(eig_orig > eig_rand_mean)
        return n_factors, eig_orig, eig_rand_mean
    except Exception as e:
        logging.error(f"Error in parallel_analysis with method {method}: {str(e)}")
        raise

def preprocess_data(
    df: pd.DataFrame, 
    allow_categorical: bool = False, 
    ui_callback: Optional[callable] = None
) -> Optional[pd.DataFrame]:
    """
    Shared data preprocessing steps
    
    Parameters:
        df: Input DataFrame
        allow_categorical: If True, preserves categorical columns
        ui_callback: Optional function to handle UI interactions
        
    Returns:
        Processed DataFrame or None if preprocessing fails
    """
    try:
        if df.empty:
            if ui_callback:
                ui_callback.error("Empty DataFrame provided")
            return None
            
        # Convert to numeric where possible
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            if ui_callback:
                ui_callback.warning(f"Non-numeric columns detected: {', '.join(non_numeric_cols)}")
                if ui_callback.checkbox("Attempt to convert non-numeric columns to numeric?"):
                    for col in non_numeric_cols:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except Exception as e:
                            logging.warning(f"Could not convert column {col} to numeric: {str(e)}")
                    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
                    if non_numeric_cols:
                        ui_callback.error(f"Could not convert: {', '.join(non_numeric_cols)}")
                        if not allow_categorical:
                            return None
        
        # Handle missing values
        if df.isnull().values.any():
            if ui_callback:
                ui_callback.warning("Dataset contains missing values")
                missing_method = ui_callback.radio(
                    "Handle missing values by:", 
                    ["Drop rows with missing values", "Fill with mean", "Fill with median", "Fill with mode"],
                    horizontal=True
                )
                
                if missing_method == "Drop rows with missing values":
                    df = df.dropna()
                elif missing_method == "Fill with mean":
                    df = df.fillna(df.mean(numeric_only=True))
                elif missing_method == "Fill with median":
                    df = df.fillna(df.median(numeric_only=True))
                elif missing_method == "Fill with mode":
                    df = df.fillna(df.mode().iloc[0])
        
        # Select appropriate columns for analysis
        if allow_categorical:
            return df
        else:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if not numeric_cols:
                if ui_callback:
                    ui_callback.error("No numeric columns available for analysis")
                return None
            return df[numeric_cols]
    except Exception as e:
        logging.error(f"Error in preprocess_data: {str(e)}")
        if ui_callback:
            ui_callback.error(f"Data preprocessing failed: {str(e)}")
        return None

def run_ttest(
    data1: Union[pd.Series, np.ndarray], 
    data2: Union[pd.Series, np.ndarray], 
    paired: bool = False
) -> Tuple[float, float]:
    """
    Run t-test (independent or paired)
    
    Parameters:
        data1: First sample data
        data2: Second sample data
        paired: Whether to perform paired t-test
        
    Returns:
        Tuple of (t-statistic, p-value)
    """
    try:
        if paired:
            t_stat, p_val = stats.ttest_rel(data1, data2)
        else:
            t_stat, p_val = stats.ttest_ind(data1, data2)
        return t_stat, p_val
    except Exception as e:
        logging.error(f"Error in run_ttest (paired={paired}): {str(e)}")
        raise

def run_anova(data: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """
    Run one-way ANOVA
    
    Parameters:
        data: Input DataFrame
        group_col: Column name for grouping variable
        value_col: Column name for outcome variable
        
    Returns:
        ANOVA results DataFrame
    """
    try:
        model = ols(f'{value_col} ~ C({group_col})', data=data).fit()
        return sm.stats.anova_lm(model, typ=2)
    except Exception as e:
        logging.error(f"Error in run_anova with groups {group_col} and values {value_col}: {str(e)}")
        raise

def run_chi2_test(
    data: pd.DataFrame, 
    var1: str, 
    var2: str
) -> Tuple[float, float, int, np.ndarray, pd.DataFrame]:
    """
    Run chi-square test of independence
    
    Parameters:
        data: Input DataFrame
        var1: First categorical variable
        var2: Second categorical variable
        
    Returns:
        Tuple containing:
        - chi2 statistic
        - p-value
        - degrees of freedom
        - expected frequencies
        - contingency table
    """
    try:
        contingency_table = pd.crosstab(data[var1], data[var2])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        return chi2, p, dof, expected, contingency_table
    except Exception as e:
        logging.error(f"Error in run_chi2_test with variables {var1} and {var2}: {str(e)}")
        raise

def run_regression(
    X: pd.DataFrame, 
    y: Union[pd.Series, np.ndarray], 
    regression_type: str = 'linear',
    **kwargs: Any
) -> Union[sm.OLS, Tuple[LogisticRegression, StandardScaler]]:
    """
    Run regression analysis
    
    Parameters:
        X: Predictor variables
        y: Outcome variable
        regression_type: Type of regression ('linear' or 'logistic')
        **kwargs: Additional arguments for regression models
        
    Returns:
        For linear regression: statsmodels OLS results object
        For logistic regression: tuple of (model, scaler)
    """
    try:
        if regression_type == 'linear':
            X = sm.add_constant(X)  # Add intercept
            model = sm.OLS(y, X, **kwargs).fit()
            return model
        elif regression_type == 'logistic':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = LogisticRegression(**kwargs)
            model.fit(X_scaled, y)
            return model, scaler
        else:
            raise ValueError(f"Unknown regression type: {regression_type}")
    except Exception as e:
        logging.error(f"Error in run_regression with type {regression_type}: {str(e)}")
        raise