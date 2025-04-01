import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from utils import preprocess_data
import io
from statsmodels.stats.multitest import multipletests  # Add this import

def show():
    st.title("🔗 Correlation Analysis")
    st.session_state.current_module = "Correlation Analysis"
    
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
            col1, col2 = st.columns(2)
            
            with col1:
                method = st.radio(
                    "Correlation method:", 
                    ["Pearson", "Spearman", "Kendall"], 
                    horizontal=True
                )
                
            with col2:
                significance_test = st.checkbox("Calculate significance (p-values)", value=True)
                adjust_p_values = st.checkbox("Adjust p-values (FDR correction)", value=False)
            
            # Variable selection
            selected_cols = st.multiselect(
                "Select columns to include in correlation", 
                numeric_cols, 
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
        
        # Run analysis
        if st.button("🚀 Calculate Correlations") and len(selected_cols) >= 2:
            try:
                # Calculate correlations
                if method == "Pearson":
                    corr_matrix = df[selected_cols].corr(method='pearson')
                elif method == "Spearman":
                    corr_matrix = df[selected_cols].corr(method='spearman')
                elif method == "Kendall":
                    corr_matrix = df[selected_cols].corr(method='kendall')
                
                # Calculate p-values if requested
                p_matrix = None
                if significance_test:
                    p_matrix = np.zeros((len(selected_cols), len(selected_cols)))
                    for i in range(len(selected_cols)):
                        for j in range(len(selected_cols)):
                            if i == j:
                                p_matrix[i, j] = 0
                            else:
                                if method == "Pearson":
                                    _, p_val = stats.pearsonr(
                                        df[selected_cols[i]].dropna(), 
                                        df[selected_cols[j]].dropna()
                                    )
                                elif method == "Spearman":
                                    _, p_val = stats.spearmanr(
                                        df[selected_cols[i]].dropna(), 
                                        df[selected_cols[j]].dropna()
                                    )
                                elif method == "Kendall":
                                    _, p_val = stats.kendalltau(
                                        df[selected_cols[i]].dropna(), 
                                        df[selected_cols[j]].dropna()
                                    )
                                p_matrix[i, j] = p_val
                    
                    # Adjust p-values if requested
                    if adjust_p_values:
                        st.caption("P-values adjusted using Benjamini/Hochberg FDR correction")
                        mask = np.triu(np.ones(p_matrix.shape)).astype(bool)
                        p_values = p_matrix[mask]
                        p_values[p_values == 0] = 1  # Replace diagonal with 1 for adjustment
                        p_adjusted = np.zeros_like(p_values)
                        _, p_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
                        p_matrix[mask] = p_adjusted
                        p_matrix.T[mask] = p_adjusted  # Make symmetric
                
                # Display results
                st.subheader("📊 Correlation Matrix")
                
                # Interactive heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=selected_cols,
                    y=selected_cols,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1,
                    colorbar=dict(title='Correlation'),
                    text=np.round(corr_matrix.values, 2),
                    hoverinfo='text'
                ))
                
                fig.update_layout(
                    title=f'{method} Correlation Matrix',
                    xaxis_title="Variables",
                    yaxis_title="Variables",
                    width=800,
                    height=700
                )
                
                st.plotly_chart(fig)
                
                # Correlation table
                st.write("Numerical Values:")
                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1))
                
                # P-value table if calculated
                if p_matrix is not None:
                    st.subheader("Significance (p-values)")
                    p_df = pd.DataFrame(p_matrix, index=selected_cols, columns=selected_cols)
                    
                    # Highlight significant correlations
                    def highlight_p(val):
                        color = 'lightgreen' if val < 0.05 else ''
                        return f'background-color: {color}'
                    
                    st.dataframe(p_df.style.applymap(highlight_p).format("{:.4f}"))
                
                # Scatter plot matrix
                st.subheader("Scatter Plot Matrix")
                scatter_fig = px.scatter_matrix(
                    df[selected_cols], 
                    title="Scatter Plot Matrix of Selected Variables"
                )
                st.plotly_chart(scatter_fig)
                
                # Download button
                st.subheader("📥 Download Results")
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    corr_matrix.to_excel(writer, sheet_name='Correlation')
                    
                    if p_matrix is not None:
                        p_df.to_excel(writer, sheet_name='P_Values')
                    
                    # Add plots to Excel
                    heatmap_bytes = fig.to_image(format="png")
                    scatter_bytes = scatter_fig.to_image(format="png")
                    
                    worksheet = writer.book.add_worksheet('Plots')
                    worksheet.insert_image('B2', 'heatmap', {'image_data': io.BytesIO(heatmap_bytes)})
                    worksheet.insert_image('B30', 'scatter_matrix', {'image_data': io.BytesIO(scatter_bytes)})
                
                st.download_button(
                    label="📊 Download Results as Excel",
                    data=output.getvalue(),
                    file_name='correlation_analysis_results.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")