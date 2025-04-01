import streamlit as st
from modules import (
    welcome, 
    factor_analysis, 
    descriptive, 
    variability, 
    correlation,
    t_tests,
    anova,
    chi_square,
    regression
)

# Set page config
st.set_page_config(
    page_title="Statistical Analysis Suite", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'df' not in st.session_state:
    st.session_state.df = None
if 'current_module' not in st.session_state:
    st.session_state.current_module = None

def main():
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    modules = {
        "Welcome": welcome.show,
        "Factor Analysis & Reliability": factor_analysis.show,
        "Descriptive Statistics": descriptive.show,
        "Variability & Distribution": variability.show,
        "Correlation Analysis": correlation.show,
        "T-Tests": t_tests.show,
        "ANOVA": anova.show,
        "Chi-Square Test": chi_square.show,
        "Regression Analysis": regression.show
    }
    
    module_choice = st.sidebar.radio("Select Module", list(modules.keys()))
    st.session_state.current_module = module_choice
    modules[module_choice]()
    
    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Statistical Analysis Suite**  
    Version 2.0  
    Created by Aslam  
    """)

if __name__ == "__main__":
    main()