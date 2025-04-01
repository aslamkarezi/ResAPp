import streamlit as st
import pandas as pd
from utils import preprocess_data, run_chi2_test

def show():
    st.title("Chi-Square Test of Independence")
    
    st.header("Data Input Method")
    input_method = st.radio("Choose data input method:", 
                          ["Upload Data File", "Use Existing Data"],
                          horizontal=True)
    
    if input_method == "Upload Data File":
        uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", 
                                       type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:  # Excel files
                    df = pd.read_excel(uploaded_file)
                
                # Preprocess and store in session state
                df = preprocess_data(df)
                st.session_state.df = df
                st.success("Data uploaded successfully!")
                st.write("Preview of your data:")
                st.dataframe(df.head())
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return
        else:
            if 'df' not in st.session_state or st.session_state.df is None:
                st.info("Please upload a data file to continue")
                return
            else:
                df = st.session_state.df
                st.write("Using previously uploaded data:")
                st.dataframe(df.head())
    else:
        if 'df' not in st.session_state or st.session_state.df is None:
            st.warning("No existing data found. Please upload data first.")
            return
        df = st.session_state.df
        st.write("Using previously uploaded data:")
        st.dataframe(df.head())
    
    # Check for categorical variables
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if len(categorical_cols) < 2:
        st.error("Need at least 2 categorical variables for chi-square test")
        return
    
    st.header("Select Variables")
    col1, col2 = st.columns(2)
    
    with col1:
        var1 = st.selectbox("Select first categorical variable:", categorical_cols)
    
    with col2:
        remaining_cols = [col for col in categorical_cols if col != var1]
        var2 = st.selectbox("Select second categorical variable:", remaining_cols)
    
    if st.button("Run Chi-Square Test"):
        chi2, p_val, dof, expected, contingency_table = run_chi2_test(df, var1, var2)
        
        st.subheader("Contingency Table")
        st.write(contingency_table)
        
        st.subheader("Results")
        st.write(f"Chi-square statistic: {chi2:.4f}")
        st.write(f"P-value: {p_val:.4f}")
        st.write(f"Degrees of freedom: {dof}")
        
        if p_val < 0.05:
            st.success("Statistically significant association (p < 0.05)")
        else:
            st.info("No statistically significant association (p ≥ 0.05)")