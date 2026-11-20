import pandas as pd

df = pd.read_csv("model_dataset_manifest.csv")

print("\nCLASS DISTRIBUTION")
print(df["class"].value_counts().sort_index())

print("\nTOTAL IMAGES:", len(df))
print("UNIQUE GROUPS:", df["group_id"].nunique())

print("\nDUPLICATE GROUP IDs:")
print(df[df["group_id"].duplicated(keep=False)])

print("\nCLASS-WISE GROUP COUNTS:")
print(df.groupby("class")["group_id"].nunique())