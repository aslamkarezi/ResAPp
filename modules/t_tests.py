import streamlit as st
import pandas as pd
from scipy import stats
from utils import preprocess_data
import numpy as np

def run_ttest(sample1, sample2=None, paired=False, pop_mean=0):
    """
    Perform t-test with flexible parameters for different test types
    """
    if sample2 is None:
        # One-sample t-test
        t_stat, p_val = stats.ttest_1samp(sample1, pop_mean)
    elif paired:
        # Paired samples t-test
        t_stat, p_val = stats.ttest_rel(sample1, sample2)
    else:
        # Independent samples t-test
        t_stat, p_val = stats.ttest_ind(sample1, sample2, equal_var=False)  # Welch's t-test
    return t_stat, p_val

def manual_data_entry(test_type):
    """Handle manual data entry based on test type"""
    st.subheader("Manual Data Entry")
    
    if test_type == "One Sample":
        data_input = st.text_area(
            "Enter your data (comma or space separated values):",
            help="Example: 12.5, 13.2, 14.1, 15.0, 12.8"
        )
        pop_mean = st.number_input("Enter population mean to compare against:", 
                                  value=0.0, step=0.1)
        
        if st.button("Analyze Manual Data"):
            if not data_input.strip():
                st.error("Please enter some data values")
                return None, None, pop_mean
            
            try:
                # Convert input to numeric array
                data = [float(x.strip()) for x in data_input.replace(",", " ").split() if x.strip()]
                if len(data) < 2:
                    st.error("You need at least 2 data points for analysis")
                    return None, None, pop_mean
                
                return data, None, pop_mean
            except ValueError:
                st.error("Invalid data format. Please enter numbers only.")
                return None, None, pop_mean
    
    elif test_type == "Independent Samples":
        col1, col2 = st.columns(2)
        
        with col1:
            group1_input = st.text_area(
                "Group 1 data (comma or space separated):",
                help="Example: 12.5, 13.2, 14.1, 15.0, 12.8"
            )
        
        with col2:
            group2_input = st.text_area(
                "Group 2 data (comma or space separated):",
                help="Example: 10.5, 11.2, 12.1, 10.0, 11.8"
            )
        
        if st.button("Analyze Manual Data"):
            if not group1_input.strip() or not group2_input.strip():
                st.error("Please enter data for both groups")
                return None, None, None
            
            try:
                group1 = [float(x.strip()) for x in group1_input.replace(",", " ").split() if x.strip()]
                group2 = [float(x.strip()) for x in group2_input.replace(",", " ").split() if x.strip()]
                
                if len(group1) < 2 or len(group2) < 2:
                    st.error("Each group needs at least 2 data points")
                    return None, None, None
                
                return group1, group2, None
            except ValueError:
                st.error("Invalid data format. Please enter numbers only.")
                return None, None, None
    
    else:  # Paired Samples
        col1, col2 = st.columns(2)
        
        with col1:
            time1_input = st.text_area(
                "First measurement (comma or space separated):",
                help="Example: 120, 125, 130, 118, 122"
            )
        
        with col2:
            time2_input = st.text_area(
                "Second measurement (comma or space separated):",
                help="Example: 115, 120, 125, 112, 118"
            )
        
        if st.button("Analyze Manual Data"):
            if not time1_input.strip() or not time2_input.strip():
                st.error("Please enter data for both measurements")
                return None, None, None
            
            try:
                time1 = [float(x.strip()) for x in time1_input.replace(",", " ").split() if x.strip()]
                time2 = [float(x.strip()) for x in time2_input.replace(",", " ").split() if x.strip()]
                
                if len(time1) != len(time2):
                    st.error("Paired measurements must have the same number of data points")
                    return None, None, None
                
                if len(time1) < 2:
                    st.error("You need at least 2 pairs of data points")
                    return None, None, None
                
                return time1, time2, None
            except ValueError:
                st.error("Invalid data format. Please enter numbers only.")
                return None, None, None
    
    return None, None, None

def show():
    st.title("T-Tests Analysis")
    
    st.header("Data Input Method")
    input_method = st.radio("Choose data input method:", 
                          ["Upload Data File", "Enter Data Manually"],
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
        df = None  # Will use manual data entry
    
    st.header("Select Test Type")
    test_type = st.radio("Choose t-test type:", 
                        ["One Sample", "Independent Samples", "Paired Samples"],
                        horizontal=True)
    
    # Handle manual data entry first
    if input_method == "Enter Data Manually":
        sample1, sample2, pop_mean = manual_data_entry(test_type)
        
        if sample1 is not None:
            if test_type == "One Sample":
                t_stat, p_val = run_ttest(sample1, pop_mean=pop_mean)
                
                st.subheader("Results")
                st.write(f"Sample size: {len(sample1)}")
                st.write(f"Sample mean: {np.mean(sample1):.4f}")
                st.write(f"Population mean: {pop_mean:.4f}")
                st.write(f"T-statistic: {t_stat:.4f}")
                st.write(f"P-value: {p_val:.4f}")
                
                if p_val < 0.05:
                    st.success("Statistically significant difference from population mean (p < 0.05)")
                else:
                    st.info("No statistically significant difference from population mean (p ≥ 0.05)")
            
            elif test_type == "Independent Samples":
                t_stat, p_val = run_ttest(sample1, sample2)
                
                st.subheader("Results")
                st.write(f"Group 1 size: {len(sample1)}, Mean: {np.mean(sample1):.4f}")
                st.write(f"Group 2 size: {len(sample2)}, Mean: {np.mean(sample2):.4f}")
                st.write(f"T-statistic: {t_stat:.4f}")
                st.write(f"P-value: {p_val:.4f}")
                
                if p_val < 0.05:
                    st.success("Statistically significant difference between groups (p < 0.05)")
                else:
                    st.info("No statistically significant difference between groups (p ≥ 0.05)")
            
            else:  # Paired Samples
                t_stat, p_val = run_ttest(sample1, sample2, paired=True)
                
                st.subheader("Results")
                st.write(f"Number of pairs: {len(sample1)}")
                st.write(f"Mean difference: {np.mean(np.array(sample1) - np.array(sample2)):.4f}")
                st.write(f"T-statistic: {t_stat:.4f}")
                st.write(f"P-value: {p_val:.4f}")
                
                if p_val < 0.05:
                    st.success("Statistically significant difference between measurements (p < 0.05)")
                else:
                    st.info("No statistically significant difference between measurements (p ≥ 0.05)")
        
        return  # Skip the file-based analysis if doing manual entry
    
    # Original file-based analysis (only if not doing manual entry)
    if 'df' not in st.session_state or st.session_state.df is None:
        return
    
    df = st.session_state.df
    
    # Get available columns of each type
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not numeric_cols:
        st.error("No numeric columns found in the data for analysis!")
        return
    
    if test_type == "One Sample":
        st.subheader("One Sample t-test")
        st.markdown("""
        **Purpose:** Tests whether the mean of a single variable is equal to a known value.
        *Example:* Checking if students' test scores differ from a passing score of 50.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            value_col = st.selectbox("Select numeric variable:", numeric_cols)
            
        with col2:
            pop_mean = st.number_input("Enter population mean to compare against:", 
                                     value=0.0, step=0.1)
        
        if st.button("Run One Sample t-test"):
            sample_data = df[value_col].dropna()
            if len(sample_data) < 2:
                st.error("Insufficient data points for analysis (need at least 2 values)")
                return
                
            t_stat, p_val = run_ttest(sample_data, pop_mean=pop_mean)
            
            st.subheader("Results")
            st.write(f"T-statistic: {t_stat:.4f}")
            st.write(f"P-value: {p_val:.4f}")
            
            if p_val < 0.05:
                st.success("Statistically significant difference from population mean (p < 0.05)")
            else:
                st.info("No statistically significant difference from population mean (p ≥ 0.05)")
    
    elif test_type == "Independent Samples":
        st.subheader("Independent Samples t-test")
        st.markdown("""
        **Purpose:** Compares means between two independent groups.
        *Example:* Comparing male vs. female salaries.
        """)
        
        if not categorical_cols:
            st.error("No categorical columns found for grouping variables!")
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            group_col = st.selectbox("Select grouping variable:", categorical_cols)
            if group_col in df.columns:
                groups = df[group_col].dropna().unique()
                if len(groups) < 2:
                    st.error("Grouping variable must have at least 2 unique values")
                    return
                group1 = st.selectbox("Select first group:", groups)
            else:
                st.error("Selected grouping column not found in data")
                return
            
        with col2:
            value_col = st.selectbox("Select numeric variable to compare:", numeric_cols)
            if group_col in df.columns:
                group2 = st.selectbox("Select second group:", 
                                    [g for g in groups if g != group1])
            else:
                st.error("Selected grouping column not found in data")
                return
        
        if st.button("Run Independent t-test"):
            if group_col not in df.columns or value_col not in df.columns:
                st.error("Selected columns not found in data")
                return
                
            group1_data = df[df[group_col] == group1][value_col].dropna()
            group2_data = df[df[group_col] == group2][value_col].dropna()
            
            if len(group1_data) < 2 or len(group2_data) < 2:
                st.error("Each group must have at least 2 data points")
                return
                
            t_stat, p_val = run_ttest(group1_data, group2_data)
            
            st.subheader("Results")
            st.write(f"Group 1 ({group1}) N: {len(group1_data)}")
            st.write(f"Group 2 ({group2}) N: {len(group2_data)}")
            st.write(f"T-statistic: {t_stat:.4f}")
            st.write(f"P-value: {p_val:.4f}")
            
            if p_val < 0.05:
                st.success("Statistically significant difference between groups (p < 0.05)")
            else:
                st.info("No statistically significant difference between groups (p ≥ 0.05)")
    
    else:  # Paired Samples
        st.subheader("Paired Samples t-test")
        st.markdown("""
        **Purpose:** Compares means of two related groups (before-after studies).
        *Example:* Comparing blood pressure before and after treatment.
        """)
        
        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns for paired t-test")
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            time1_col = st.selectbox("Select first measurement:", numeric_cols)
        
        with col2:
            remaining_cols = [col for col in numeric_cols if col != time1_col]
            time2_col = st.selectbox("Select second measurement:", remaining_cols)
        
        if st.button("Run Paired t-test"):
            paired_data = df[[time1_col, time2_col]].dropna()
            if len(paired_data) < 2:
                st.error("Insufficient paired data points (need at least 2 pairs)")
                return
                
            t_stat, p_val = run_ttest(paired_data[time1_col], paired_data[time2_col], paired=True)
            
            st.subheader("Results")
            st.write(f"Number of pairs: {len(paired_data)}")
            st.write(f"T-statistic: {t_stat:.4f}")
            st.write(f"P-value: {p_val:.4f}")
            
            if p_val < 0.05:
                st.success("Statistically significant difference between measurements (p < 0.05)")
            else:
                st.info("No statistically significant difference between measurements (p ≥ 0.05)")