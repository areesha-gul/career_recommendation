#!/usr/bin/env python
"""Quick import test"""

try:
    from modules.module2_dt import _fuse_dimension_predictions
    print("SUCCESS: Fusion function imported")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
