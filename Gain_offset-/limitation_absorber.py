import numpy as np
from scipy.constants import physical_constants
from scipy.integrate import quad
import pandas as pd
import os

# Constants
r0 = physical_constants["classical electron radius"][0]  # meters
E_keV = 356
alpha = E_keV / 511  # using mc² in keV

# Klein-Nishina differential cross-section
def d_sigma_dOmega(theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    E_ratio = 1 / (1 + alpha * (1 - cos_theta))
    return 0.5 * r0**2 * E_ratio**2 * (E_ratio + 1/E_ratio - sin_theta**2)

# Total cross-section
def sigma_KN(alpha):
    term1 = (1 + alpha) / alpha**2
    term2 = (2 * (1 + alpha)) / (1 + 2 * alpha)
    term3 = np.log(1 + 2 * alpha) / alpha
    term4 = np.log(1 + 2 * alpha) / (2 * alpha)
    term5 = (1 + 3 * alpha) / ((1 + 2 * alpha)**2)
    return 2 * np.pi * r0**2 * (term1 * (term2 - term3) + term4 - term5)

# Probability calculation
def calculate_probability(theta_min_deg, theta_max_deg, W_const):
    theta_min = np.radians(theta_min_deg)
    theta_max = np.radians(theta_max_deg)

    def integrand(theta):
        return d_sigma_dOmega(theta) * np.sin(theta) * W_const

    numerator, _ = quad(integrand, theta_min, theta_max)
    denominator = sigma_KN(alpha)
    return numerator / denominator

# Region definitions
regions = {
    "Top": {
        "angles": [(70, 115), (52, 97), (90, 135), (79.5, 124.5)],
        "W_const": np.pi / 6.8
    },
    "Center": {
        "angles": [(70, 133), (52, 115), (90, 153), (79.5, 142.5)],
        "W_const": np.pi / 2
    },
    "Bottom": {
        "angles": [(70, 160), (52, 142), (90, 180), (79.5, 169.5)],
        "W_const": np.pi
    }
}

# Compile results
results = []
for region, data in regions.items():
    for i, (theta_min, theta_max) in enumerate(data["angles"], 1):
        prob = calculate_probability(theta_min, theta_max, data["W_const"])
        results.append({
            "Region": region,
            "Case #": i,
            "Theta Min (°)": theta_min,
            "Theta Max (°)": theta_max,
            "W_const": round(data["W_const"], 4),
            "Probability (%)": round(prob * 100, 5)
        })

# Save to CSV
csv_file = "/home/arya/research_work/Compton_cam/ouput/klein_nishina_probabilities.csv"
df = pd.DataFrame(results)

if os.path.exists(csv_file):
    df_existing = pd.read_csv(csv_file)
    df_combined = pd.concat([df_existing, df], ignore_index=True)
    df_combined.to_csv(csv_file, index=False)
else:
    df.to_csv(csv_file, index=False)

df
