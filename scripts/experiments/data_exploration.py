import pandas as pd

df = pd.read_csv('./scripts/experiments/model_stats.csv')
print(df.shape)

# Remove columns with only 1 unique value and print removed columns
# removed_columns = df.columns[df.nunique() <= 1]
# df = df.loc[:, df.nunique() > 1]
# print(f"\nRemoved columns: {list(removed_columns)}")
# print(df.shape)

# Print percentage of "is_better" overall and for each dataset
print("\nPercentage of 'is_better' for each dataset:")
print((df.groupby(['dataset_name', 'dataset_group'])['scores_is_better']
    .value_counts(normalize=True) * 100))

print("\nPercentage of 'is_better' overall:")
print((df['scores_is_better'].value_counts(normalize=True) * 100))

# -----

import re

# Define the layer steps you want to compare
steps = ['step_10', 'step_25', 'step_50', 'step_100', 'step_200', 'step_300', 'step_400', 'step_500']
layers = ['mlp.0', 'mlp.1', 'mlp.2']  # Adapt if you have more
stats = ['weight_mean', 'weight_std', 'weight_var', 'weight_frobenius_norm', 'weight_spectral_norm', 'weight_alpha_hat']  # etc.

# Find all relevant columns
pattern = re.compile(r'weights_step_(\d+)_mlp\.(\d+)\.(\w+)')
relevant_cols = [col for col in df.columns if pattern.match(col)]

# Compute diffs and store them in a dictionary first
diff_features = {}

for stat in stats:
    for layer in layers:
        for i in range(1, len(steps)):
            prev_step = steps[i-1]
            curr_step = steps[i]
            col_prev = f'weights_{prev_step}_{layer}.{stat}'
            col_curr = f'weights_{curr_step}_{layer}.{stat}'
            if col_prev in df.columns and col_curr in df.columns:
                new_col = f'diff_{curr_step}_{prev_step}_{layer}_{stat}'
                diff_features[new_col] = df[col_curr] - df[col_prev]

# Concatenate all new features at once
diff_df = pd.DataFrame(diff_features)
df = pd.concat([df, diff_df], axis=1)

# Save or return the enhanced dataframe
df.to_csv('./scripts/experiments/new_model_stats.csv', index=False)

