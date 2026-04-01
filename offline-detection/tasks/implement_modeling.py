"""
Task file for Issue #33, #34, #35: Implement Isolation Forest Modeling
========================================================================

This script loads the data, cleans it, preprocesses it, and runs the 
Isolation Forest model. Finally, it outputs the evaluation metrics.
"""

import sys
from pathlib import Path

# Allow importing from the parent offline-detection directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import pipeline

if __name__ == "__main__":
    print("-" * 60)
    print("Testing Pipeline with Isolation Forest Modeling")
    print("-" * 60)
    
    # Run the full pipeline logic from main.py
    pipeline()
