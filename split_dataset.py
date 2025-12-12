from datasets import load_dataset

# 1. Download the dataset
print("Downloading dataset...")
dataset = load_dataset("victorzarzu/interior-design-prompt-editing-dataset-train")

# 2. Split it: 80% Train, 20% Test
# The seed=42 ensures the split is the same every time you run this
print("Splitting dataset 80/20...")
split_ds = dataset["train"].train_test_split(test_size=0.2, seed=42)

# 3. Save the split dataset to a folder on your Mac
print("Saving to 'interior_design_split' folder...")
split_ds.save_to_disk("interior_design_split")

print("Done! You now have a 'train' and 'test' set in the 'interior_design_split' folder.")