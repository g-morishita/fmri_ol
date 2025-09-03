import pandas as pd
import matplotlib.pyplot as plt

# Load data
file_path = 'combined_estimate.csv'
df = pd.read_csv(file_path)

# Mapping for relabeling
label_map = {
    'PR_minus_0': 'observed reward',
    'PC_minus_0': 'observed action',
    'is_partner_high_expTRUE': "partner's exploration level",
    'PR_minus_0:is_partner_high_expTRUE': 'reward x exploration level',
    'PC_minus_0:is_partner_high_expTRUE': 'action x exploration level'
}

# Apply relabeling
df['name'] = df['name'].replace(label_map)

# Sort by estimate
df_sorted = df.sort_values('estimate')

# Plot
plt.figure(figsize=(10, 6))
plt.barh(df_sorted['name'], df_sorted['estimate'], xerr=df_sorted['s.e.'] * 1.96)
plt.xlabel('Effect Size')
plt.title('Effects on choice behavior')
plt.tight_layout()
plt.savefig("plot.png")
