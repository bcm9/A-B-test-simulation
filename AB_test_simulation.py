"""
A/B test simulation with Bayesian analysis

Simulates an A/B test to compare conversion rates of versions of an app feature. 
Generates synthetic data for each version, performs both a t-test and Bayesian analysis to check if the difference in conversion rates is significant, and visualises the results. 
Bayesian inference used to estimate the posterior distributions of conversion rates and calculate the probability that one version performs better than the other. 
Multivariate logistic regression used to examine the effect of age and device type on conversion.
Power analysis included to determine the required sample size.

"""

##################################################################################################################################################################################################################
# Import libraries
##################################################################################################################################################################################################################
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import beta

# Set seed
np.random.seed(123)

# Define the sample size for A/B test
sample_size = 1500

# Simulating conversion rates for two versions (A and B)
conversion_rate_A = 0.30
conversion_rate_B = 0.35

##################################################################################################################################################################################################################
# Power analysis
##################################################################################################################################################################################################################
effect_size = abs(conversion_rate_A - conversion_rate_B) / np.sqrt((conversion_rate_A * (1 - conversion_rate_A) + conversion_rate_B * (1 - conversion_rate_B)) / 2)
alpha = 0.05  # significance level
power = 0.8   # desired power (80%)

# Calculate required sample size
from statsmodels.stats.power import TTestIndPower
power_analysis = TTestIndPower()
required_sample_size = power_analysis.solve_power(effect_size, power=power, alpha=alpha)

print(f"Required sample size for sufficient power: {required_sample_size:.0f}")

##################################################################################################################################################################################################################
# Plot power vs. sample size
##################################################################################################################################################################################################################
# Array of different sample sizes
sample_sizes = np.arange(50, 2000, 50)

# Calculate power for each sample size
powers = [power_analysis.solve_power(effect_size, nobs1=n, alpha=alpha) for n in sample_sizes]

# Plot power vs sample size
plt.figure(figsize=(8, 6))
plt.plot(sample_sizes, powers, label='Power curve', color='lightblue', lw=4)
plt.axhline(y=0.8, color='red', linestyle='--', label='80% power')  # Add horizontal line for 80% power
fs=22
plt.title('T-Test Power vs. Sample Size', fontsize=fs, fontweight='bold')
plt.xlabel('Sample Size (n)', fontsize=fs-2, fontweight='bold')
plt.ylabel('Power (1-\u03B2)', fontsize=fs-2, fontweight='bold')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=fs-2)
plt.yticks(fontsize=fs-2)
# Remove axes borders
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
# Remove x and y tick lines
plt.tick_params(axis='both', which='both', length=0)
plt.legend(loc='lower right', fontsize=fs-4, frameon=False)

# Save plot
folder = 'C:/Users/bc22/OneDrive/Documents/code/AB_test_simulation/'
plt.savefig(folder + 'power_vs_sample_size.png', dpi=300, bbox_inches='tight')
plt.show()

##################################################################################################################################################################################################################
# Generate synthetic data
##################################################################################################################################################################################################################
# Generate random 0s (no conversion) and 1s (conversion) based on the conversion rate
data_A = np.random.binomial(1, conversion_rate_A, sample_size)
# Generate similar data with a slightly higher conversion rate
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
# T-test to check if the difference in conversion rates is statistically significant
# Outputs p-value to determine if the difference between two versions is statistically significant
##################################################################################################################################################################################################################
# t-test as comparing the mean conversion rates between two groups. t-test is appropriate for continuous or binary data where we want to see if the average outcome differs. 
# chi-square test would be used if comparing categorical counts
t_stat, p_value = stats.ttest_ind(df['Version_A'], df['Version_B'])

# Output the results of the A/B test
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

##################################################################################################################################################################################################################
# Bayesian analysis to estimate conversion rates and calculate the probability that B is better than A
# Bayesian analysis gives probability of one being better than the other, giving more insight into likelihood of which performs better while accounting for uncertainty
##################################################################################################################################################################################################################
# Observed data: count conversions and trials for each version
conversions_A = data_A.sum()  # N conversions in A
conversions_B = data_B.sum()  # N conversions in B
trials_A = len(data_A)        # N users in A
trials_B = len(data_B)        # N users in B

# Beta prior (non-informative prior: flat, no assumptions)
alpha_prior = 1
beta_prior = 1

# Beta posterior update based on data
# Posterior parameters for Version A
alpha_post_A = alpha_prior + conversions_A  # Update alpha with the number of successes (conversions)
beta_post_A = beta_prior + (trials_A - conversions_A)  # Update beta with the number of failures (non-conversions)

# Posterior parameters for Version B
alpha_post_B = alpha_prior + conversions_B  # Update alpha with the number of successes
beta_post_B = beta_prior + (trials_B - conversions_B)  # Update beta with the number of failures

# Generate random samples from posterior distribution, see how likely each conversion rate is
posterior_A = beta.rvs(alpha_post_A, beta_post_A, size=10000)
posterior_B = beta.rvs(alpha_post_B, beta_post_B, size=10000)

# Plot posterior distributions
plt.figure(figsize=(10, 6))
plt.rcParams['font.family'] = 'Calibri'
plt.hist(posterior_A, bins=50, alpha=0.5, label='A', density=True,color=['#76c7c0'])
plt.hist(posterior_B, bins=50, alpha=0.5, label='B', density=True,color=['#ff6f61'])
plt.title('Posterior Distributions of Conversion Rates', fontweight='bold', fontsize=fs)
plt.xlabel('Conversion Rate', fontweight='bold', fontsize=fs)
plt.ylabel('Density', fontweight='bold', fontsize=fs)
plt.xticks(fontsize=fs-1)
plt.yticks(fontsize=fs-1)
plt.legend(loc='upper right', fontsize=fs-4, frameon=False)
plt.grid(axis='y', alpha=0.25)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.tick_params(axis='both', which='both', length=0)
folder = 'C:/Users/bc22/OneDrive/Documents/code/AB_test_simulation/'
plt.savefig(folder+'post_dist.png', dpi=300, bbox_inches='tight')
plt.show()

# Print comparison
prob_B_better_than_A = np.mean(posterior_B > posterior_A)
print(f"Probability that Version B is better than A: {prob_B_better_than_A:.2%}")

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
# Plot conversion rate
##################################################################################################################################################################################################################
plt.figure(figsize=(8, 7))
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
folder = 'C:/Users/bc22/OneDrive/Documents/code/AB_test_simulation/'
plt.savefig(folder+'conversion_rate.png', dpi=300, bbox_inches='tight')
plt.show()

##################################################################################################################################################################################################################
# Multivariate data simulation: effect of age and device type
##################################################################################################################################################################################################################
# Factors (age group and device type)
age_group = np.random.choice([0, 1], size=sample_size, p=[0.5, 0.5])  # 0 = younger, 1 = older
device_type = np.random.choice([0, 1], size=sample_size, p=[0.7, 0.3])  # 0 = mobile, 1 = desktop

# Combine into a df for regression
df = pd.DataFrame({
    'conversion': np.concatenate([data_A, data_B]),
    'version': np.concatenate([np.zeros(sample_size), np.ones(sample_size)]),  # 0 = Version A, 1 = Version B
    'age_group': np.concatenate([age_group, age_group]),  # Repeat age group for both versions
    'device_type': np.concatenate([device_type, device_type])  # Repeat device type for both versions
})

##################################################################################################################################################################################################################
# Logistic regression (multivariate analysis)
##################################################################################################################################################################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from statsmodels.stats.power import TTestIndPower
# Split data into training and test sets
X = df[['version', 'age_group', 'device_type']]
y = df['conversion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Make predictions, assess accuracy
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy:.2%}")

# Print coefficients
coefficients = pd.DataFrame({
    'Variable': ['Version (A vs B)', 'Age Group (Young vs Old)', 'Device Type (Mobile vs Desktop)'],
    'Coefficient': log_reg.coef_[0]
})

print("\nCoefficients from logistic regression:")
print(coefficients)

##################################################################################################################################################################################################################
# Plot coefficients
##################################################################################################################################################################################################################
plt.figure(figsize=(8, 7))
plt.rcParams['font.family'] = 'Calibri'
plt.bar(coefficients['Variable'], coefficients['Coefficient'], color=['#50C878'])
plt.title('Regression Coefficients', fontweight='bold', fontsize=fs)
plt.ylabel('Coefficient Value', fontweight='bold', fontsize=fs)
plt.xticks(fontsize=fs-3, rotation=45)
plt.yticks(fontsize=fs-1)
plt.grid(axis='y', alpha=0.25)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.tick_params(axis='both', which='both', length=0)
plt.savefig(folder+'regression_coeffs.png', dpi=300, bbox_inches='tight')
plt.show()