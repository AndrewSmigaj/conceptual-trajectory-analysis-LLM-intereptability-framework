#!/usr/bin/env python3
"""
Alternative export script using Selenium for D3 Sankey visualization.
Falls back to this if Playwright is not available.
"""

import os
import sys
import time
from pathlib import Path

# Try to import selenium, install if needed
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
except ImportError:
    print("Selenium not installed. Please install with:")
    print(f"{sys.executable} -m pip install selenium pillow")
    sys.exit(1)

def export_sankey_with_selenium(html_path, output_dir, base_name="apple_sankey_full_network"):
    """Export D3 sankey using Selenium WebDriver."""
    
    html_path = Path(html_path).absolute()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")
    
    # Configure Chrome options
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--window-size=1600,800')
    
    # Create driver
    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        # Load the HTML file
        file_url = f"file://{html_path}"
        print(f"Loading {file_url}")
        driver.get(file_url)
        
        # Wait for D3 to render
        time.sleep(3)
        
        # Take screenshot
        png_path = output_dir / f"{base_name}.png"
        driver.save_screenshot(str(png_path))
        print(f"Exported PNG: {png_path}")
        
        # Note: Selenium doesn't have built-in PDF export
        # Would need to use print dialog or additional tools
        
    finally:
        driver.quit()
    
    return png_path

def create_manual_export_instructions():
    """Create instructions for manual export if automated tools fail."""
    
    instructions = """
# Manual Export Instructions for D3 Sankey Diagram

If automated export fails, follow these steps:

1. Open the HTML file in Chrome/Firefox:
   experiments/apple_variety/results/apple_realistic/d3_sankey_full_network.html

2. For PNG export:
   - Press F12 to open Developer Tools
   - Press Ctrl+Shift+P (Cmd+Shift+P on Mac)
   - Type "screenshot" and select "Capture full size screenshot"
   - Save as "apple_sankey_full_network.png"

3. For PDF export:
   - Press Ctrl+P (Cmd+P on Mac) to open print dialog
   - Change destination to "Save as PDF"
   - Set Layout to "Landscape"
   - Set Margins to "None" or "Minimum"
   - Enable "Background graphics"
   - Save as "apple_sankey_full_network.pdf"

4. Move files to: arxiv_apple/figures/

Alternative: Use online HTML to Image converters:
- https://html2canvas.hertzen.com/
- https://www.converthtml.net/html-to-image/
"""
    
    instructions_path = Path("experiments/apple_variety/MANUAL_EXPORT_INSTRUCTIONS.md")
    instructions_path.write_text(instructions)
    print(f"\nManual export instructions saved to: {instructions_path}")

def main():
    """Try to export sankey, provide fallback instructions if needed."""
    
    project_root = Path(__file__).parent.parent.parent
    html_file = project_root / "experiments/apple_variety/results/apple_realistic/d3_sankey_full_network.html"
    output_dir = project_root / "arxiv_apple/figures"
    
    try:
        png_path = export_sankey_with_selenium(html_file, output_dir)
        print(f"\nExport successful: {png_path}")
    except Exception as e:
        print(f"\nAutomated export failed: {e}")
        create_manual_export_instructions()
        print("\nPlease follow manual export instructions.")

if __name__ == "__main__":
    main()