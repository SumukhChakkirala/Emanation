# import pickle
# import numpy as np

# PKL_PATH = r"C:\Users\Sumukh\Desktop\projects\Emanation-standalone\programs\experiments\IQData\iq_dict_crepe_dirac_comb.pkl"

# with open(PKL_PATH, "rb") as f:
#     data = pickle.load(f)

# print("Type:", type(data))
# print("Total samples:", len(data))

# print("\n--- First 3 entries ---")
# for i, (k, v) in enumerate(data.items()):
#     print(f"\n[{i}] Key:", k)
#     print("  Type:", type(v))
#     print("  Shape:", v.shape)
#     print("  Dtype:", v.dtype)
#     print("  First 10 samples:")
#     print(" ", v[:10])
#     if i == 10:
#         break


import pickle

PKL_PATH = r"C:\Users\Sumukh\Desktop\projects\Emanation-standalone\programs\experiments\IQData\iq_dict_crepe_dirac_comb.pkl"

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)

# Extract unique bin indices
bins = sorted({k.split("_")[1] for k in data.keys()})

print("Number of unique bins:", len(bins))
print("First 10 bins:", bins[:10])
print("Last 10 bins:", bins[-10:])
