import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns

def show():
    st.title("📊 Statistical Analysis Suite")
    st.markdown("""
    Welcome to the Statistical Analysis Suite! This interactive application provides tools for:
    
    - **Factor Analysis** (PCA, EFA with various rotations)
    - **Descriptive Statistics** (Mean, Median, Mode, etc.)
    - **Variability Analysis** (Standard Deviation, Variance, Outlier detection)
    - **Reliability Analysis** (Cronbach's Alpha)
    - **Correlation Analysis** (Pearson, Spearman, Kendall)
    - **Normality Testing** (Shapiro-Wilk, Anderson-Darling)
    
    ### Getting Started
    1. Select an analysis module from the sidebar
    2. Upload your data file (CSV or Excel)
    3. Configure your analysis options
    4. View and download results
    
    ### Sample Data
    Try the app with sample data before uploading your own:
    """)
    
    sample_options = {
        "Iris Dataset": sns.load_dataset('iris'),
        "Tips Dataset": sns.load_dataset('tips'),
        "Random Normal Data": pd.DataFrame(np.random.normal(size=(100, 5)), 
                        columns=[f"Var_{i}" for i in range(1,6)])
    }
    
    sample_choice = st.selectbox("Choose sample dataset", list(sample_options.keys()))
    
    if st.button("Load Sample Data"):
        st.session_state.df = sample_options[sample_choice]
        st.success(f"Loaded {sample_choice} successfully!")
        st.dataframe(st.session_state.df.head())
        
    st.markdown("---")
    st.markdown("""
    ### Documentation
    - **Factor Analysis**: For dimensionality reduction and latent variable identification
    - **Descriptive Stats**: Basic statistics for data understanding
    - **Variability Analysis**: Measures of spread and distribution
    """)