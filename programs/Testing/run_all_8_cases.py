"""
Run all 8 cases: Cases 1-4 with decimation, and Cases 1-4 without decimation.
Each case runs for 10 frames only.
"""

import sys
import os
import importlib.util

# Add parent directory and core directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(os.path.dirname(current_dir), 'core'))

# Load the module using importlib
spec = importlib.util.spec_from_file_location(
    "UnifiedGenerate_Case_A_Decimation",
    os.path.join(current_dir, "UnifiedGenerate_Case_A_Decimation.py")
)
unified_module = importlib.util.module_from_spec(spec)
sys.modules["UnifiedGenerate_Case_A_Decimation"] = unified_module
spec.loader.exec_module(unified_module)

generate_unified_case_dataset = unified_module.generate_unified_case_dataset
save_dataset_info = unified_module.save_dataset_info

# Output directory
OUTPUT_DIR = r"C:\Users\veidp\Desktop\Reserch Papers\Pitich Estimation\Emanation\programs\Testing"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration for all 8 cases
cases_config = [
    # (case_idx, no_decimation, label)
    (1, False, "case1_withdec"),
    (2, False, "case2_withdec"),
    (3, False, "case3_withdec"),
    (4, False, "case4_withdec"),
    (1, True,  "case1_nodec"),
    (2, True,  "case2_nodec"),
    (3, True,  "case3_nodec"),
    (4, True,  "case4_nodec"),
]

def main():
    print("=" * 80)
    print("RUNNING ALL 8 CASES (10 frames each)")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    for idx, (case_num, no_decimation, label) in enumerate(cases_config, 1):
        mode_str = "no-decimation" if no_decimation else "with decimation"
        print(f"\n[{idx}/8] Running Case {case_num} ({mode_str})...")
        print("-" * 80)
        
        output_file = os.path.join(OUTPUT_DIR, f"iq_dict_{label}_10frames.pkl")
        
        try:
            iq_dict = generate_unified_case_dataset(
                output_path=output_file,
                case_idx=case_num,
                snr_list=list(range(-10, 21)),  # Default SNR range
                n_input_frames=10,  # Only 10 frames
                no_decimation=no_decimation,
                use_gpu=False,  # CPU only for stability
                n_workers=None,  # Use default (CPU count)
            )
            
            # Save metadata info
            save_dataset_info(output_file, iq_dict, case_num, no_decimation=no_decimation)
            
            print(f"✓ Case {case_num} ({mode_str}) completed successfully")
            
        except Exception as e:
            print(f"✗ Error in Case {case_num} ({mode_str}): {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("ALL CASES COMPLETED")
    print("=" * 80)
    print(f"\nOutput files saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for case_num, no_decimation, label in cases_config:
        pkl_file = f"iq_dict_{label}_10frames.pkl"
        info_file = f"iq_dict_{label}_10frames_info.txt"
        print(f"  - {pkl_file}")
        print(f"  - {info_file}")


if __name__ == "__main__":
    main()
