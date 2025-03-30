# run_dashboard.py (in root folder)
import os
import sys
from app.dashboard import main

if __name__ == "__main__":
    # Add the project root to Python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()