import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    preprocess_data, 
    calculate_cronbach_alpha, 
    parallel_analysis,
    handle_missing_values,
    remove_outliers
)
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

def show():
    st.title("🔍 Factor Analysis & Reliability")
    st.session_state.current_module = "Factor Analysis"
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Data input section - expanded with manual input option
    with st.expander("📤 Data Input", expanded=True):
        input_method = st.radio("Data input method:", 
                              ["Upload CSV/Excel file", "Enter data manually"], 
                              horizontal=True)
        
        if input_method == "Upload CSV/Excel file":
            uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
            
            if uploaded_file is not None:
                try:
                    status_text.text("Reading file...")
                    progress_bar.progress(10)
                    
                    # Read the file
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.session_state.df = df
                    progress_bar.progress(30)
                    
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    return
        else:
            # Manual data input
            st.write("Enter your data below (comma-separated values):")
            
            # Get number of variables and samples
            col1, col2 = st.columns(2)
            with col1:
                num_vars = st.number_input("Number of variables", min_value=2, max_value=50, value=5)
            with col2:
                num_samples = st.number_input("Number of samples", min_value=3, max_value=1000, value=100)
            
            # Create input grid
            data_input = []
            headers = []
            
            # Get variable names
            st.write("**Variable Names:**")
            var_cols = st.columns(min(5, num_vars))
            for i in range(num_vars):
                with var_cols[i % len(var_cols)]:
                    headers.append(st.text_input(f"Var {i+1} name", value=f"Var{i+1}", key=f"var_name_{i}"))
            
            # Get data rows
            st.write("**Enter Data:**")
            for row in range(num_samples):
                cols = st.columns(min(5, num_vars))
                row_data = []
                for i in range(num_vars):
                    with cols[i % len(cols)]:
                        val = st.number_input(
                            f"Sample {row+1}, {headers[i]}",
                            value=0.0,
                            key=f"sample_{row}_var_{i}"
                        )
                        row_data.append(val)
                data_input.append(row_data)
            
            if st.button("Create DataFrame"):
                try:
                    df = pd.DataFrame(data_input, columns=headers)
                    st.session_state.df = df
                    progress_bar.progress(30)
                    st.success("DataFrame created successfully!")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error creating DataFrame: {str(e)}")
    
    if 'df' in st.session_state and st.session_state.df is not None:
        df = st.session_state.df
        
        # Data preprocessing
        with st.expander("🛠 Data Preprocessing", expanded=True):
            df = preprocess_data(df)
            if df is None:
                return
            
            st.session_state.df = df
            
            # Data summary
            st.subheader("Data Summary")
            st.dataframe(df.describe().T)
            
            progress_bar.progress(50)
        
        # Analysis configuration
        with st.expander("⚙ Analysis Configuration", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                method = st.radio("Analysis method:", ["PCA", "EFA"], horizontal=True)
                
                # Parallel analysis for factor retention
                if st.checkbox("Use parallel analysis to determine number of factors"):
                    try:
                        status_text.text("Running parallel analysis...")
                        n_factors, eig_orig, eig_rand_mean = parallel_analysis(df, method=method)
                        st.info(f"Parallel analysis suggests {n_factors} factors/components")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=list(range(1, len(eig_orig)+1)),
                            y=eig_orig,
                            mode='lines+markers', 
                            name='Actual Data'
                        ))
                        fig.add_trace(go.Scatter(
                            x=list(range(1, len(eig_rand_mean)+1)), 
                            y=eig_rand_mean,
                            mode='lines+markers', 
                            name='Random Data'
                        ))
                        fig.update_layout(
                            title='Parallel Analysis Scree Plot',
                            xaxis_title='Factor Number',
                            yaxis_title='Eigenvalue'
                        )
                        st.plotly_chart(fig)
                        
                    except Exception as e:
                        st.warning(f"Parallel analysis failed: {str(e)}")
                        n_factors = min(3, len(df.columns))
                
                else:
                    n_factors = st.number_input(
                        "Number of factors/components", 
                        min_value=1, 
                        max_value=len(df.columns), 
                        value=min(3, len(df.columns))
                    )
            
            with col2:
                if method == 'EFA':
                    rotation = st.selectbox("Rotation method:", 
                                          ["varimax", "promax", "oblimin", "none"])
                else:
                    rotation = None
                
                # Additional options
                st.checkbox("Calculate factor scores", value=True, key='calc_scores')
                st.checkbox("Calculate Cronbach's alpha", value=True, key='calculate_cronbach_alpha')
            
            # Test assumptions
            if method == 'EFA' and st.checkbox("Check factorability of data"):
                try:
                    bartlett, p_value = calculate_bartlett_sphericity(df)
                    kmo_all, kmo_model = calculate_kmo(df)
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Bartlett's Test (p-value)", f"{p_value:.4f}", 
                               "Assumption met" if p_value < 0.05 else "Assumption not met")
                    col2.metric("KMO Measure", f"{kmo_model:.3f}", 
                               "Meritorious" if kmo_model > 0.7 else "May be problematic")
                    
                    st.caption("Bartlett's test should be significant (p < .05) and KMO > 0.6 for good factorability")
                except Exception as e:
                    st.warning(f"Could not calculate factorability tests: {str(e)}")
            
            progress_bar.progress(70)
        
        # Run analysis
        if st.button("🚀 Run Analysis"):
            try:
                status_text.text("Running analysis...")
                
                # Perform factor analysis
                if method == 'PCA':
                    fa = FactorAnalyzer(n_factors=n_factors, method='principal', rotation=None)
                    results_type = "Principal Components"
                    analysis_note = "PCA extracts components that explain maximum variance in the data."
                else:
                    rotation_method = None if rotation == 'none' else rotation
                    fa = FactorAnalyzer(n_factors=n_factors, method='minres', rotation=rotation_method)
                    results_type = "Factors"
                    analysis_note = "EFA identifies latent factors explaining correlations among variables."
                
                fa.fit(df)
                
                # Display results
                st.subheader(f"📊 {results_type} Analysis Results")
                st.caption(analysis_note)
                
                # Get loadings
                loadings = pd.DataFrame(
                    fa.loadings_, 
                    index=df.columns, 
                    columns=[f'{results_type[:-1]} {i+1}' for i in range(n_factors)]
                )
                
                # Highlight significant loadings
                def highlight_loadings(val):
                    color = 'white'
                    if abs(val) >= 0.5:
                        color = 'lightgreen'
                    elif abs(val) >= 0.3:
                        color = 'lightyellow'
                    return f'background-color: {color}'
                
                st.dataframe(loadings.style.format("{:.3f}").applymap(highlight_loadings))
                
                # Get eigenvalues
                ev, v = fa.get_eigenvalues()
                
                # Create interactive scree plot
                st.subheader("Scree Plot")
                fig = px.line(
                    x=range(1, len(ev)+1), 
                    y=ev, 
                    markers=True,
                    labels={'x': results_type, 'y': 'Eigenvalue'},
                    title=f'Scree Plot ({results_type})'
                )
                fig.add_hline(y=1, line_dash="dash", line_color="red")
                st.plotly_chart(fig)
                
                # Additional results
                st.subheader("Additional Results")
                
                if method == 'EFA':
                    # Communalities
                    communalities = pd.DataFrame(
                        fa.get_communalities(),
                        index=df.columns,
                        columns=['Communality']
                    )
                    st.write("Communalities:")
                    st.dataframe(communalities.style.format("{:.3f}").background_gradient(cmap='Blues'))
                    
                    # Cronbach's alpha
                    if st.session_state.get('calculate_cronbach_alpha', True):
                        alpha = calculate_cronbach_alpha(df)
                        if alpha is not None:
                            st.metric("Cronbach's Alpha", f"{alpha:.3f}",
                                    "Excellent" if alpha > 0.8 else 
                                    "Good" if alpha > 0.7 else 
                                    "Acceptable" if alpha > 0.6 else "Questionable")
                
                # Factor scores
                if st.session_state.get('calc_scores', True):
                    factor_scores = pd.DataFrame(
                        fa.transform(df),
                        columns=[f'{results_type[:-1]}_Score_{i+1}' for i in range(n_factors)]
                    )
                    st.write("Factor Scores:")
                    st.dataframe(factor_scores.head())
                
                # Download buttons
                st.subheader("📥 Download Results")
                
                # Create Excel writer with multiple sheets
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    loadings.to_excel(writer, sheet_name='Loadings')
                    if method == 'EFA':
                        communalities.to_excel(writer, sheet_name='Communalities')
                        if 'factor_scores' in locals():
                            factor_scores.to_excel(writer, sheet_name='Factor_Scores')
                    
                    # Add eigenvalues sheet
                    pd.DataFrame({'Eigenvalues': ev}).to_excel(writer, sheet_name='Eigenvalues')
                
                st.download_button(
                    label="📊 Download All Results as Excel",
                    data=output.getvalue(),
                    file_name='factor_analysis_results.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                st.balloons()
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                progress_bar.progress(0)
                status_text.text("")