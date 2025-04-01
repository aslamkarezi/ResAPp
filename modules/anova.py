import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

def load_and_preprocess_data(uploaded_file):
    """Load and preprocess Excel data, handling formulas"""
    try:
        # Read Excel file while evaluating formulas
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        # Clean column names
        df.columns = [str(col).strip().replace(' ', '_') for col in df.columns]
        
        # Convert formula results to proper values
        for col in df.columns:
            # Convert categorical columns
            if df[col].dtype == 'object':
                try:
                    # Try to convert to numeric first
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                    # If still object and low cardinality, treat as categorical
                    if df[col].dtype == 'object' and df[col].nunique() <= 10:
                        df[col] = df[col].astype('category')
                except Exception:
                    df[col] = df[col].astype('category')
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def detect_column_types(df):
    """Robust detection of numeric and categorical columns"""
    numeric_cols = []
    categorical_cols = []
    
    for col in df.columns:
        # Skip columns with all missing values
        if df[col].isna().all():
            continue
            
        # Check for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            # If numeric with few unique values, offer as both
            if 2 <= df[col].nunique() <= 5:
                categorical_cols.append(col)
            numeric_cols.append(col)
        # Check for categorical columns
        elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            if 2 <= df[col].nunique() <= 10:
                categorical_cols.append(col)
    
    return numeric_cols, categorical_cols

def check_assumptions(data, group_col, value_col):
    """Check ANOVA assumptions with error handling"""
    assumptions = {
        'normality': {'passed': False, 'test': 'Shapiro-Wilk', 'p_value': None, 'message': ''},
        'homogeneity': {'passed': False, 'test': "Levene's", 'p_value': None, 'message': ''}
    }
    
    try:
        # Ensure proper data types
        data = data[[group_col, value_col]].dropna()
        data[group_col] = data[group_col].astype('category')
        data[value_col] = pd.to_numeric(data[value_col], errors='raise')
        
        # Normality check
        model = ols(f'Q("{value_col}") ~ C(Q("{group_col}"))', data=data).fit()
        residuals = model.resid
        
        if len(residuals) >= 3:
            _, normality_p = stats.shapiro(residuals)
            assumptions['normality']['p_value'] = normality_p
            assumptions['normality']['passed'] = normality_p > 0.05
            assumptions['normality']['message'] = (
                f"✅ Normally distributed (p = {normality_p:.3f})" 
                if normality_p > 0.05 
                else f"❌ Non-normal (p = {normality_p:.3f})"
            )
        else:
            assumptions['normality']['message'] = "⚠️ Insufficient data for normality test"
        
        # Homogeneity check
        groups = data[group_col].unique()
        group_samples = [data[data[group_col] == g][value_col].dropna() for g in groups]
        
        if len(group_samples) >= 2:
            _, homo_p = stats.levene(*group_samples)
            assumptions['homogeneity']['p_value'] = homo_p
            assumptions['homogeneity']['passed'] = homo_p > 0.05
            assumptions['homogeneity']['message'] = (
                f"✅ Equal variances (p = {homo_p:.3f})" 
                if homo_p > 0.05 
                else f"❌ Unequal variances (p = {homo_p:.3f})"
            )
        else:
            assumptions['homogeneity']['message'] = "⚠️ Insufficient groups for variance test"
            
    except Exception as e:
        st.error(f"Assumption checking failed: {str(e)}")
    
    return assumptions

def perform_tukey_hsd(data, group_col, value_col):
    """Perform Tukey's HSD post-hoc test with robust formatting"""
    try:
        data = data[[group_col, value_col]].dropna()
        data[group_col] = data[group_col].astype('category')
        data[value_col] = pd.to_numeric(data[value_col], errors='raise')
        
        mc = MultiComparison(data[value_col], data[group_col])
        tukey_result = mc.tukeyhsd()
        
        # Convert to DataFrame with proper numeric formatting
        result_df = pd.DataFrame(
            data=tukey_result._results_table.data[1:],
            columns=tukey_result._results_table.data[0]
        )
        
        # Ensure numeric columns are properly formatted
        numeric_cols = ['meandiff', 'lower', 'upper', 'p-adj']
        for col in numeric_cols:
            if col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
        
        return result_df
    except Exception as e:
        st.error(f"Post-hoc analysis failed: {str(e)}")
        return pd.DataFrame()

def plot_results(data, group_col, value_col):
    """Create visualization of ANOVA results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    try:
        # Boxplot
        sns.boxplot(x=group_col, y=value_col, data=data, ax=ax1)
        ax1.set_title(f'Distribution of {value_col} by {group_col}')
        sns.stripplot(x=group_col, y=value_col, data=data, color='black', alpha=0.5, ax=ax1)
        
        # Q-Q plot
        model = ols(f'Q("{value_col}") ~ C(Q("{group_col}"))', data=data).fit()
        sm.qqplot(model.resid, line='s', ax=ax2)
        ax2.set_title('Q-Q Plot of Residuals')
    except Exception as e:
        for ax in [ax1, ax2]:
            ax.clear()
            ax.text(0.5, 0.5, f"Plot error:\n{str(e)}", ha='center', va='center')
    
    plt.tight_layout()
    return fig

def show():
    st.title("ANOVA Analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx'])
    
    if uploaded_file is not None:
        try:
            # Load and preprocess data
            df = load_and_preprocess_data(uploaded_file)
            
            if df is None or df.empty:
                st.error("No valid data loaded")
                return
            
            # Detect column types
            numeric_cols, categorical_cols = detect_column_types(df)
            
            # Debug panel
            with st.expander("🔍 Data Diagnostics"):
                st.write("### Raw Data Types")
                st.dataframe(df.dtypes.astype(str).to_frame('Data Type'))
                
                st.write("### Numeric Columns Detected", numeric_cols)
                if numeric_cols:
                    st.write("Sample values:")
                    st.dataframe(df[numeric_cols].head())
                else:
                    st.warning("No numeric columns detected!")
                
                st.write("### Categorical Columns Detected", categorical_cols)
                if categorical_cols:
                    st.write("Sample values:")
                    for col in categorical_cols:
                        st.write(f"**{col}**: {df[col].dropna().unique()[:10]}")
                else:
                    st.warning("No categorical columns detected!")
                
                st.write("### Full Data Sample")
                st.dataframe(df.head())
            
            # Check if we have the required columns
            if not numeric_cols or not categorical_cols:
                st.error("Column detection issues found!")
                
                if not numeric_cols:
                    st.write("""
                    **No numeric columns were detected. Possible solutions:**
                    1. Check if your numeric columns contain non-numeric values
                    2. Ensure columns with numbers are formatted as numbers in Excel
                    """)
                
                if not categorical_cols:
                    st.write("""
                    **No categorical columns were detected. Possible solutions:**
                    1. Ensure you have columns with text categories or limited unique values
                    2. Check if formula results are being properly evaluated
                    """)
                
                return
            
            # ANOVA configuration
            st.subheader("ANOVA Configuration")
            
            # Smart defaults for your specific data structure
            default_group = next((col for col in categorical_cols if 'cat' in col.lower()), None)
            default_group = default_group or categorical_cols[0] if categorical_cols else None
                
            default_value = next((col for col in numeric_cols if 'score' in col.lower()), None)
            default_value = default_value or numeric_cols[0] if numeric_cols else None
            
            col1, col2 = st.columns(2)
            with col1:
                group_col = st.selectbox(
                    "Select grouping (categorical) variable:",
                    categorical_cols,
                    index=categorical_cols.index(default_group) if default_group in categorical_cols else 0
                )
                
            with col2:
                value_col = st.selectbox(
                    "Select numeric variable to compare:",
                    numeric_cols,
                    index=numeric_cols.index(default_value) if default_value in numeric_cols else 0
                )
            
            if st.button("Run One-Way ANOVA"):
                try:
                    # Prepare analysis data
                    analysis_df = df[[group_col, value_col]].dropna()
                    
                    if analysis_df.empty:
                        st.error("No valid data remaining after dropping missing values")
                        return
                    
                    # Convert and validate
                    analysis_df[group_col] = analysis_df[group_col].astype('category')
                    analysis_df[value_col] = pd.to_numeric(analysis_df[value_col], errors='raise')
                    
                    # Check group sizes
                    group_counts = analysis_df[group_col].value_counts()
                    if len(group_counts) < 2:
                        st.error(f"Need at least 2 groups, found: {group_counts.index.tolist()}")
                        return
                        
                    if any(group_counts < 2):
                        st.error(f"Insufficient data in groups: {group_counts[group_counts < 2].index.tolist()}")
                        return
                    
                    # Run ANOVA
                    model = ols(f'Q("{value_col}") ~ C(Q("{group_col}"))', data=analysis_df).fit()
                    anova_result = sm.stats.anova_lm(model, typ=2)
                    
                    # Display results
                    st.subheader("ANOVA Results")
                    st.dataframe(anova_result.style.format("{:.4f}"))
                    
                    # Group statistics
                    st.subheader("Group Statistics")
                    stats_df = analysis_df.groupby(group_col)[value_col].agg(['mean', 'std', 'count', 'sem'])
                    st.dataframe(stats_df.style.format("{:.2f}"))
                    
                    # Assumptions
                    st.subheader("Assumption Checks")
                    assumptions = check_assumptions(analysis_df, group_col, value_col)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(assumptions['normality']['message'])
                    with col2:
                        st.write(assumptions['homogeneity']['message'])
                    
                    # Visualization
                    st.subheader("Visualization")
                    fig = plot_results(analysis_df, group_col, value_col)
                    st.pyplot(fig)
                    
                    # Post-hoc analysis
                    term = f'C(Q("{group_col}"))'
                    if term not in anova_result.index:
                        term = f'C({group_col})'
                    
                    if term in anova_result.index and anova_result.loc[term, 'PR(>F)'] < 0.05:
                        st.subheader("Post-hoc Analysis (Tukey's HSD)")
                        tukey_results = perform_tukey_hsd(analysis_df, group_col, value_col)
                        if not tukey_results.empty:
                            # Convert numeric columns to float for proper formatting
                            format_dict = {col: "{:.4f}" for col in tukey_results.columns 
                                         if pd.api.types.is_numeric_dtype(tukey_results[col])}
                            st.dataframe(tukey_results.style.format(format_dict))
                    else:
                        st.info("No significant differences found (p ≥ 0.05)")
                        
                except Exception as e:
                    st.error(f"ANOVA analysis failed: {str(e)}")
                    with st.expander("Technical Details"):
                        st.write("Error type:", type(e).__name__)
                        st.write("Full error:", str(e))
                        if 'analysis_df' in locals():
                            st.write("Group values:", analysis_df[group_col].unique())
                            st.write("Numeric values sample:", analysis_df[value_col].head())
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            with st.expander("Technical Details"):
                st.write("Error type:", type(e).__name__)
                st.write("Full error:", str(e))