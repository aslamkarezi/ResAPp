import streamlit as st
import pandas as pd
from utils import preprocess_data, run_regression

def show():
    st.title("Regression Analysis")
    
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
    
    st.header("Select Regression Type")
    reg_type = st.selectbox("Choose regression type:", 
                          ["Linear Regression", "Logistic Regression"])
    
    st.header("Select Variables")
    
    # Get numeric columns for predictors (can vary by regression type)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    outcome = st.selectbox("Select outcome variable:", df.columns)
    
    # For logistic regression, verify binary outcome if selected
    if reg_type == "Logistic Regression":
        if outcome not in df.columns:
            st.error("Selected outcome variable not found in data")
            return
        if len(df[outcome].unique()) != 2:
            st.error("Logistic regression requires a binary outcome variable")
            return
    
    # Select predictors - exclude outcome variable
    available_predictors = [col for col in df.columns if col != outcome]
    predictors = st.multiselect("Select predictor variables:", available_predictors)
    
    if not predictors:
        st.warning("Please select at least one predictor variable")
        return
    
    if st.button(f"Run {reg_type}"):
        try:
            if reg_type == "Linear Regression":
                X = df[predictors]
                y = df[outcome]
                
                # Check for missing values
                if X.isnull().any().any() or y.isnull().any():
                    st.warning("Your data contains missing values. Rows with missing values will be dropped.")
                    X = X.dropna()
                    y = y.loc[X.index]
                
                model = run_regression(X, y, 'linear')
                
                st.subheader("Regression Results")
                st.write(model.summary())
            
            else:  # Logistic Regression
                X = df[predictors]
                y = df[outcome]
                
                # Check for missing values
                if X.isnull().any().any() or y.isnull().any():
                    st.warning("Your data contains missing values. Rows with missing values will be dropped.")
                    X = X.dropna()
                    y = y.loc[X.index]
                
                model, scaler = run_regression(X, y, 'logistic')
                
                st.subheader("Logistic Regression Results")
                st.write("Coefficients:")
                coef_df = pd.DataFrame({
                    "Predictor": predictors,
                    "Coefficient": model.coef_[0]
                })
                st.write(coef_df)
                st.write(f"Intercept: {model.intercept_[0]}")
                st.write("\nNote: Coefficients are on the log-odds scale")
        
        except Exception as e:
            st.error(f"An error occurred during regression analysis: {str(e)}")