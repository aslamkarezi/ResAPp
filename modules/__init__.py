from .welcome import show as welcome_show
from .factor_analysis import show as factor_analysis_show
from .descriptive import show as descriptive_show
from .variability import show as variability_show
from .correlation import show as correlation_show
from .t_tests import show as t_tests_show
from .anova import show as anova_show
from .chi_square import show as chi_square_show
from .regression import show as regression_show

# Define exports
__all__ = [
    'welcome',
    'factor_analysis',
    'descriptive',
    'variability',
    'correlation',
    't_tests',
    'anova',
    'chi_square',
    'regression'
]

# Create module objects with show functions
welcome = type('welcome', (), {'show': welcome_show})
factor_analysis = type('factor_analysis', (), {'show': factor_analysis_show})
descriptive = type('descriptive', (), {'show': descriptive_show})
variability = type('variability', (), {'show': variability_show})
correlation = type('correlation', (), {'show': correlation_show})
t_tests = type('t_tests', (), {'show': t_tests_show})
anova = type('anova', (), {'show': anova_show})
chi_square = type('chi_square', (), {'show': chi_square_show})
regression = type('regression', (), {'show': regression_show})