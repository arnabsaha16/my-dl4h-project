import glob
import json

data = []
for fname in sorted(glob.glob("ecgqa/ptbxl/paraphrased/train/*.json")):
    with open(fname, "r") as f:
        data.extend(json.load(f))

print("Total samples:", len(data))
print("First sample:", data[0])