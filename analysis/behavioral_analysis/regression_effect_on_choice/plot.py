import pandas as pd
import matplotlib.pyplot as plt

# Load data
file_path = 'combined_estimate.csv'
df = pd.read_csv(file_path)

# Mapping for relabeling
label_map = {
    'PR_minus_0': "partner's reward",
    'PC_minus_0': "partner's action",
    'is_partner_high_expTRUE': "partner's exploration level",
    'PR_minus_0:is_partner_high_expTRUE': "partner's reward\nx partner selection",
    'PC_minus_0:is_partner_high_expTRUE': "partner's action\nx partner selection"
}

# Apply relabeling
df['name'] = df['name'].replace(label_map)

# Keep the same order as in your screenshot (not sorted by estimate)
order = [
    "partner's action",
    "partner's reward",
    "partner's action\nx partner selection",
    "partner's reward\nx partner selection"
]

df_sorted = df.set_index('name').loc[order].reset_index()

# Plot
plt.figure(figsize=(7, 6))
import numpy as np
bars = plt.bar(np.arange(4), df_sorted['estimate'],
               yerr=df_sorted['s.e.'],
               capsize=4, color='gray')

# plt.ylabel('Effects on the behavior', fontsize=14)
# plt.xticks(rotation=45, ha='right', fontsize=14)
# hidden x-axis labels
plt.xticks(np.arange(4), ['']*4)
# remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.yticks([-1.0, 0.0, 1.0, 2.0], fontsize=18)
plt.axhline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig("plot.png", dpi=300)
plt.show()
