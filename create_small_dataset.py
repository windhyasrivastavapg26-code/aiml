import pandas as pd

CSV_PATH = "dataset/koniq10k_distributions_sets.csv"

OUTPUT_PATH = "koniq_small.csv"

df = pd.read_csv(CSV_PATH, on_bad_lines="skip")

# Select a small sample from each split
train = df[df["set"] == "training"].sample(
    n=2000, random_state=42
)

validation = df[df["set"] == "validation"].sample(
    n=500, random_state=42
)

test = df[df["set"] == "test"].sample(
    n=300, random_state=42
)

# Combine datasets
small_df = pd.concat(
    [train, validation, test],
    ignore_index=True
)

small_df.to_csv(OUTPUT_PATH, index=False)

print("Small dataset created!")
print(small_df["set"].value_counts())
print("Total images:", len(small_df))