import pickle
import numpy as np

# Load both files
with open('./test_results/iq_dict_case1_unified.pkl', 'rb') as f:
    data_orig = pickle.load(f)

with open('./test_results/iq_dict_case1_fast.pkl', 'rb') as f:
    data_fast = pickle.load(f)

# Check if keys are the same
keys_orig = set(data_orig.keys())
keys_fast = set(data_fast.keys())

if keys_orig == keys_fast:
    print("✓ Keys match")
else:
    print("✗ Keys differ")
    print("Extra in orig:", keys_orig - keys_fast)
    print("Extra in fast:", keys_fast - keys_orig)

# Check if values are the same
all_match = True
for key in keys_orig:
    if key in data_fast:
        val_orig = data_orig[key]
        val_fast = data_fast[key]
        if not np.allclose(val_orig, val_fast, rtol=1e-10, atol=1e-15):
            print(f"✗ Values differ for key {key}")
            all_match = False
            break

if all_match:
    print("✓ All values match")
else:
    print("✗ Some values differ")

print(f"Original samples: {len(data_orig)}")
print(f"Fast samples: {len(data_fast)}")