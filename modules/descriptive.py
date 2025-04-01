import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats
from utils import preprocess_data

def show():
    st.title("📏 Descriptive Statistics Calculator")
    st.session_state.current_module = "Descriptive Statistics"
    
    # Initialize session state for plot type if it doesn't exist
    if 'plot_type' not in st.session_state:
        st.session_state.plot_type = "Histogram"
    
    # Data upload section
    with st.expander("📤 Data Upload", expanded=True):
        uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                # Read the file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.df = df
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Data preprocessing
        with st.expander("🛠 Data Preprocessing", expanded=True):
            df = preprocess_data(df)
            if df is None:
                return
            
            st.session_state.df = df
            
            # Data summary
            st.subheader("Data Preview")
            st.dataframe(df.head())
        
        # Analysis configuration
        with st.expander("⚙ Analysis Configuration", expanded=True):
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            selected_cols = st.multiselect(
                "Select columns to analyze", 
                numeric_cols, 
                default=numeric_cols[:min(3, len(numeric_cols))]
            )
            
            # Additional statistics options
            st.subheader("Additional Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                calc_trimmed = st.checkbox("Trimmed mean (10%)", value=True)
                calc_geometric = st.checkbox("Geometric mean", value=False)
                calc_harmonic = st.checkbox("Harmonic mean", value=False)
            
            with col2:
                calc_skewness = st.checkbox("Skewness", value=True)
                calc_kurtosis = st.checkbox("Kurtosis", value=True)
                calc_quantiles = st.checkbox("Quantiles (25%, 50%, 75%)", value=True)
        
        # Interactive plot selection - update session state when changed
        plot_type = st.selectbox(
            "Select plot type", 
            ["Histogram", "Box Plot", "Violin Plot", "ECDF"],
            key='plot_type_selector'
        )
        st.session_state.plot_type = plot_type
        
        # Run analysis
        if st.button("🚀 Calculate Statistics") and selected_cols:
            try:
                results = []
                
                for col in selected_cols:
                    col_data = df[col].dropna()
                    if len(col_data) == 0:
                        continue
                    
                    # Basic statistics
                    stats_dict = {
                        'Column': col,
                        'Mean': round(col_data.mean(), 4),
                        'Median': round(col_data.median(), 4),
                        'Mode': ", ".join(map(str, col_data.mode().values)) if not col_data.mode().empty else "N/A",
                        'Count': len(col_data),
                        'Missing': df[col].isna().sum()
                    }
                    
                    # Additional statistics
                    if calc_trimmed:
                        stats_dict['Trimmed Mean (10%)'] = round(stats.trim_mean(col_data, 0.1), 4)
                    
                    if calc_geometric:
                        stats_dict['Geometric Mean'] = round(stats.gmean(col_data), 4) if all(col_data > 0) else "N/A"
                    
                    if calc_harmonic:
                        stats_dict['Harmonic Mean'] = round(stats.hmean(col_data), 4) if all(col_data > 0) else "N/A"
                    
                    if calc_quantiles:
                        q1, q2, q3 = np.percentile(col_data, [25, 50, 75])
                        stats_dict['25th Percentile'] = round(q1, 4)
                        stats_dict['50th Percentile'] = round(q2, 4)
                        stats_dict['75th Percentile'] = round(q3, 4)
                    
                    if calc_skewness:
                        stats_dict['Skewness'] = round(stats.skew(col_data), 4) if len(col_data) > 2 else "N/A"
                    
                    if calc_kurtosis:
                        stats_dict['Kurtosis'] = round(stats.kurtosis(col_data), 4) if len(col_data) > 3 else "N/A"
                    
                    results.append(stats_dict)
                
                # Display results
                if results:
                    results_df = pd.DataFrame(results)
                    st.subheader("📊 Results")
                    
                    # Generate plot based on selected type
                    if st.session_state.plot_type == "Histogram":
                        fig = px.histogram(
                            df[selected_cols], 
                            nbins=30, 
                            marginal="rug",
                            title="Distribution of Selected Variables"
                        )
                    elif st.session_state.plot_type == "Box Plot":
                        fig = px.box(
                            df[selected_cols], 
                            title="Box Plot of Selected Variables"
                        )
                    elif st.session_state.plot_type == "Violin Plot":
                        fig = px.violin(
                            df[selected_cols], 
                            title="Violin Plot of Selected Variables"
                        )
                    elif st.session_state.plot_type == "ECDF":
                        fig = px.ecdf(
                            df[selected_cols], 
                            title="ECDF of Selected Variables"
                        )
                    
                    st.plotly_chart(fig)
                    
                    # Display table
                    st.dataframe(results_df)
                    
                    # Download button
                    st.subheader("📥 Download Results")
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        results_df.to_excel(writer, sheet_name='Statistics', index=False)
                        
                        # Add plots to Excel
                        if 'fig' in locals():
                            plot_bytes = fig.to_image(format="png")
                            worksheet = writer.book.add_worksheet('Plots')
                            worksheet.insert_image('B2', '', {'image_data': io.BytesIO(plot_bytes)})
                    
                    st.download_button(
                        label="📊 Download Results as Excel",
                        data=output.getvalue(),
                        file_name='descriptive_stats_results.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")