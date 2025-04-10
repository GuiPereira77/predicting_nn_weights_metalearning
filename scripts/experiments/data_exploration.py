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
