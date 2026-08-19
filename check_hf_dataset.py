from datasets import load_dataset

ds = load_dataset("ai4bharat/MSMARCO-XI", "en", split="train", streaming=True)
for i, item in enumerate(ds):
    print(item)
    if i >= 1:
        break
