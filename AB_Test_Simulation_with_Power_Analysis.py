"""
A/B test simulation

Simulates an A/B test to compare the conversion rates of two versions of an app feature. It generates synthetic data for each version, 
performs a t-test to check if the difference in conversion rates is statistically significant, 
and visualises the results. A power analysis is included to determine the 
required sample size for statistical power.

"""

##################################################################################################################################################################################################################
# Importing libraries
##################################################################################################################################################################################################################
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(123)

# Define the sample size for A/B test
sample_size = 1500

# Simulating conversion rates for two versions (A and B) of the app feature
# Version A has a conversion rate of 30%
# Version B has a higher conversion rate of 35% (hypothetical improvement)
conversion_rate_A = 0.30
conversion_rate_B = 0.35

##################################################################################################################################################################################################################
# Power analysis
##################################################################################################################################################################################################################
effect_size = abs(conversion_rate_A - conversion_rate_B) / np.sqrt((conversion_rate_A * (1 - conversion_rate_A) + conversion_rate_B * (1 - conversion_rate_B)) / 2)
alpha = 0.05  # significance level
power = 0.8   # desired power (80%)

# Calculate the required sample size
from statsmodels.stats.power import TTestIndPower
power_analysis = TTestIndPower()
required_sample_size = power_analysis.solve_power(effect_size, power=power, alpha=alpha)

print(f"Required sample size for sufficient power: {required_sample_size:.0f}")

##################################################################################################################################################################################################################
# Generate synthetic data
##################################################################################################################################################################################################################
# For version A, we generate random 0s (no conversion) and 1s (conversion) based on the conversion rate
data_A = np.random.binomial(1, conversion_rate_A, sample_size)
# For version B, we generate similar data with a slightly higher conversion rate
data_B = np.random.binomial(1, conversion_rate_B, sample_size)

# Creating a DataFrame for easier data handling
df = pd.DataFrame({
    'Version_A': data_A,
    'Version_B': data_B
})

# Calculate the conversion rates for both versions
conversion_rate_A_observed = df['Version_A'].mean()
conversion_rate_B_observed = df['Version_B'].mean()

print(f"Observed Conversion Rate - Version A: {conversion_rate_A_observed:.2%}")
print(f"Observed Conversion Rate - Version B: {conversion_rate_B_observed:.2%}")

##################################################################################################################################################################################################################
# Perform a t-test to check if the difference in conversion rates is statistically significant
##################################################################################################################################################################################################################
# t-test as comparing the mean conversion rates between two groups. t-test is appropriate for continuous or binary data where we want to see if the average outcome differs. 
# chi-square test would be used if comparing categorical counts
t_stat, p_value = stats.ttest_ind(df['Version_A'], df['Version_B'])

# Output the results of the A/B test
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

##################################################################################################################################################################################################################
# Calculate CIs
##################################################################################################################################################################################################################
def calculate_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    stderr = stats.sem(data)
    margin_of_error = stderr * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean - margin_of_error, mean + margin_of_error

# Calculate confidence intervals for both versions
ci_A = calculate_confidence_interval(data_A)
ci_B = calculate_confidence_interval(data_B)

print(f"95% Confidence Interval for Version A: {ci_A}")
print(f"95% Confidence Interval for Version B: {ci_B}")

# Calculate the error margins for plot (difference between upper and lower CI)
error_A = (ci_A[1] - ci_A[0]) / 2  # Half of the CI range
error_B = (ci_B[1] - ci_B[0]) / 2  # Half of the CI range

##################################################################################################################################################################################################################
# Plots
##################################################################################################################################################################################################################
plt.figure(figsize=(8, 8))
fs=18
plt.rcParams['font.family'] = 'Calibri'
plt.bar(['Version A', 'Version B'], 
        [conversion_rate_A_observed, conversion_rate_B_observed], 
        yerr=[error_A, error_B],  # Add error bars
        color=['#76c7c0', '#ff6f61'],
        capsize=13)
plt.title('Conversion Rates for A/B Test', fontweight='bold', fontsize=fs)
plt.ylabel('Conversion Rate', fontweight='bold', fontsize=fs)
plt.ylim(0, 1)
plt.xticks(fontsize=fs-1)
plt.yticks(fontsize=fs-1)
plt.grid(axis='y', alpha=0.25)
t_test_text = f"t = {t_stat:.4f}\np = {p_value:.4f}"
# Display the t-test result in the top-right corner
plt.text(0.5, 0.5, t_test_text, fontsize=fs-1, verticalalignment='center', horizontalalignment='center', transform=plt.gca().transAxes)
# Remove axes borders
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
# Remove x and y tick lines
plt.tick_params(axis='both', which='both', length=0)
folder = 'C:/Users/bcm9/OneDrive/Documents/code/AB_test_simulation/'
plt.savefig(folder+'conversion_rate.png', dpi=300, bbox_inches='tight')
plt.show()