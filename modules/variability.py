import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from utils import preprocess_data, remove_outliers

def show():
    st.title("📈 Variability & Distribution Analysis")
    st.session_state.current_module = "Variability Analysis"

    # Data input section
    data_source = st.radio("Data Input Method:", ["File Upload", "Manual Entry"], horizontal=True)

    if data_source == "File Upload":
        with st.expander("📤 Data Upload", expanded=True):
            uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                    st.session_state.df = df
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    return
    else:
        with st.expander("✍️ Manual Data Entry", expanded=True):
            manual_data = st.text_area("Enter data (CSV format):",
                                     "A,B,C\n1.1,2.2,3.3\n4.4,5.5,6.6\n7.7,8.8,9.9",
                                     height=200)
            if st.button("Load Data"):
                try:
                    df = pd.read_csv(io.StringIO(manual_data))
                    st.session_state.df = df
                except Exception as e:
                    st.error(f"Error parsing data: {str(e)}")
                    return

    if 'df' not in st.session_state or st.session_state.df is None:
        return

    df = preprocess_data(st.session_state.df)
    if df is None:
        return
    st.session_state.df = df

    # Data Preview
    with st.expander("🔍 Data Preview", expanded=True):
        st.dataframe(df.head())
        st.write("Data Shape:", df.shape)

    # Analysis Configuration
    with st.expander("⚙ Analysis Configuration", expanded=True):
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            st.error("No numeric columns found for analysis!")
            return

        selected_cols = st.multiselect(
            "Select columns to analyze",
            numeric_cols,
            default=numeric_cols[:min(3, len(numeric_cols))]
        )

        # Outlier Detection
        st.subheader("Outlier Detection")
        outlier_method = st.radio("Method:", ["None", "Z-score", "IQR"], horizontal=True)
        threshold = st.slider("Threshold", 1.0, 5.0, 3.0, 0.1) if outlier_method != "None" else None

        # Normality Tests
        st.subheader("Normality Tests")
        col1, col2 = st.columns(2)
        with col1:
            test_shapiro = st.checkbox("Shapiro-Wilk Test", True)
            test_anderson = st.checkbox("Anderson-Darling Test", False)
        with col2:
            test_ks = st.checkbox("Kolmogorov-Smirnov Test", False)
            test_dagostino = st.checkbox("D'Agostino K² Test", False)

    if st.button("🚀 Run Analysis") and selected_cols:
        try:
            results = []
            outlier_reports = []
            normality_reports = []

            for col in selected_cols:
                col_data = df[col].dropna()
                if len(col_data) == 0:
                    continue

                # Outlier Handling
                original_count = len(col_data)
                if outlier_method == "Z-score":
                    col_data = remove_outliers(col_data, method='zscore', threshold=threshold)
                elif outlier_method == "IQR":
                    col_data = remove_outliers(col_data, method='iqr', threshold=threshold)
                removed_outliers = original_count - len(col_data)

                # Variability Statistics
                stats_dict = {
                    'Column': col,
                    'Count': len(col_data),
                    'Mean': float(col_data.mean()),
                    'Std Dev': float(col_data.std()),
                    'Variance': float(col_data.var()),
                    'Range': float(col_data.max() - col_data.min()),
                    'IQR': float(col_data.quantile(0.75) - col_data.quantile(0.25)),
                    'MAD': float(stats.median_abs_deviation(col_data)),
                    'CV (%)': float((col_data.std()/col_data.mean())*100) if col_data.mean() != 0 else "N/A",
                    'Outliers Removed': removed_outliers
                }
                results.append(stats_dict)

                # Outlier Report
                if outlier_method != "None" and removed_outliers > 0:
                    outlier_reports.append({
                        'Column': col,
                        'Method': outlier_method,
                        'Threshold': float(threshold),
                        'Removed': removed_outliers,
                        'Remaining': len(col_data)
                    })

                # Normality Tests
                test_results = {'Column': col}

                if test_shapiro and 3 <= len(col_data) <= 5000:
                    try:
                        _, p = stats.shapiro(col_data)
                        test_results['Shapiro-Wilk p-value'] = f"{p:.4f}"
                        test_results['Shapiro Normal'] = p > 0.05
                    except ValueError as e:
                        test_results['Shapiro-Wilk p-value'] = "N/A (Too few data points)"
                        test_results['Shapiro Normal'] = "N/A"

                if test_anderson and len(col_data) >= 8:
                    try:
                        res = stats.anderson(col_data)
                        test_results['Anderson-Darling Stat'] = f"{res.statistic:.4f}"
                        test_results['Anderson Normal'] = res.statistic < res.critical_values[2]
                    except ValueError as e:
                        test_results['Anderson-Darling Stat'] = "N/A (Too few data points)"
                        test_results['Anderson Normal'] = "N/A"

                if test_ks and len(col_data) >= 5:
                    try:
                        _, p = stats.kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
                        test_results['KS p-value'] = f"{p:.4f}"
                        test_results['KS Normal'] = p > 0.05
                    except ValueError as e:
                        test_results['KS p-value'] = "N/A (Standard deviation is zero)"
                        test_results['KS Normal'] = "N/A"

                if test_dagostino and len(col_data) >= 20:
                    try:
                        _, p = stats.normaltest(col_data)
                        test_results['D\'Agostino p-value'] = f"{p:.4f}"
                        test_results['D\'Agostino Normal'] = p > 0.05
                    except ValueError as e:
                        test_results['D\'Agostino p-value'] = "N/A (Too few data points)"
                        test_results['D\'Agostino Normal'] = "N/A"

                if len(test_results) > 1:
                    normality_reports.append(test_results)

            # Display Results
            st.subheader("📊 Analysis Results")

            # Plot Selection
            plot_type = st.selectbox("Visualization Type",
                                    ["Box Plot", "Violin Plot", "Histogram", "QQ Plot"])

            if plot_type in ["Box Plot", "Violin Plot", "Histogram"]:
                fig = px.box(df[selected_cols], y=selected_cols, title=f"{plot_type} of Selected Columns") if plot_type == "Box Plot" else \
                      px.violin(df[selected_cols], y=selected_cols, title=f"{plot_type} of Selected Columns") if plot_type == "Violin Plot" else \
                      px.histogram(df.melt(value_vars=selected_cols, var_name='Column', value_name='Value'),
                                   x='Value', color='Column', nbins=30, marginal="rug",
                                   title="Histogram of Selected Columns")
                st.plotly_chart(fig)
            elif plot_type == "QQ Plot":
                selected_q = st.selectbox("Select column for QQ Plot", selected_cols)
                fig = go.Figure()
                data = df[selected_q].dropna()
                if len(data) > 1:
                    (osm, osr), (slope, intercept, _) = stats.probplot(data, fit=True)
                    fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Data'))
                    fig.add_trace(go.Scatter(x=osm, y=slope*osm+intercept, mode='lines', name='Normal'))
                    fig.update_layout(title=f'QQ Plot for {selected_q}', xaxis_title="Theoretical Quantiles", yaxis_title="Ordered Values")
                    st.plotly_chart(fig)
                else:
                    st.warning(f"Not enough data points to generate QQ Plot for {selected_q}.")

            # Results Tables
            st.subheader("Summary Statistics")
            st.dataframe(pd.DataFrame(results))

            if outlier_reports:
                st.subheader("Outlier Report")
                st.dataframe(pd.DataFrame(outlier_reports))

            if normality_reports:
                st.subheader("Normality Tests")
                norm_df = pd.DataFrame(normality_reports)

                # Safe styling implementation
                bool_cols = [c for c in norm_df.columns if 'Normal' in c]
                for col in bool_cols:
                    norm_df[col] = norm_df[col].map({True: 'Normal', False: 'Not Normal', 'N/A': 'N/A'})

                def highlight_normality(val):
                    return 'background-color: lightgreen' if val == 'Normal' else \
                           'background-color: lightcoral' if val == 'Not Normal' else ''

                st.dataframe(
                    norm_df.style.applymap(highlight_normality, subset=bool_cols)
                )

        except Exception as e:
            st.error(f"Analysis error: {str(e)}")

# Helper functions (assuming these are in your utils.py)
# utils.py
def preprocess_data(df):
    if df is not None:
        return df.copy()
    return None

def remove_outliers(series, method='zscore', threshold=3):
    if method == 'zscore':
        z = np.abs(stats.zscore(series))
        return series[z < threshold]
    elif method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return series[(series >= lower_bound) & (series <= upper_bound)]
    return series