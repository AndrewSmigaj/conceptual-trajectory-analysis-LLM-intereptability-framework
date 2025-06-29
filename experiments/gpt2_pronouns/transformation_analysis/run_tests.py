#!/usr/bin/env python
"""
Test runner for transformation analysis tests.
Run from the transformation_analysis directory.
"""

import sys
import subprocess
from pathlib import Path


def run_tests(args=None):
    """Run pytest with appropriate configuration."""
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add any additional arguments
    if args:
        cmd.extend(args)
    else:
        # Default arguments
        cmd.extend([
            "-v",  # Verbose
            "--tb=short",  # Short traceback
            "--cov=.",  # Coverage for current directory
            "--cov-report=term-missing",  # Show missing lines
            "--cov-report=html",  # Generate HTML report
            "--cov-config=pytest.ini",  # Use our config
        ])
    
    # Run pytest
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


if __name__ == "__main__":
    # Pass any command line arguments to pytest
    exit_code = run_tests(sys.argv[1:])
    sys.exit(exit_code)