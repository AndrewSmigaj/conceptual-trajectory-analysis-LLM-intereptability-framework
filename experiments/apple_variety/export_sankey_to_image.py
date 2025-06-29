#!/usr/bin/env python3
"""
Export D3 Sankey HTML to PNG/PDF for paper publication.
Uses Playwright to capture the D3 visualization at high quality.
"""

import os
import sys
import asyncio
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Playwright not installed. Installing...")
    os.system(f"{sys.executable} -m pip install playwright")
    os.system("playwright install chromium")
    from playwright.async_api import async_playwright

async def export_sankey_to_images(html_path, output_dir, base_name="apple_sankey_full_network"):
    """Export D3 sankey HTML to PNG and PDF formats."""
    
    html_path = Path(html_path).absolute()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")
    
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Set viewport for high quality capture
        await page.set_viewport_size({"width": 1600, "height": 800})
        
        # Load the HTML file
        file_url = f"file://{html_path}"
        print(f"Loading {file_url}")
        await page.goto(file_url)
        
        # Wait for D3 to render
        await page.wait_for_timeout(2000)  # Give D3 time to render
        
        # Export as PNG (high resolution)
        png_path = output_dir / f"{base_name}.png"
        await page.screenshot(
            path=str(png_path),
            full_page=True,
            scale="device"  # Use device pixel ratio for high quality
        )
        print(f"Exported PNG: {png_path}")
        
        # Export as PDF
        pdf_path = output_dir / f"{base_name}.pdf"
        await page.pdf(
            path=str(pdf_path),
            format="A4",
            landscape=True,
            print_background=True,
            margin={"top": "20px", "right": "20px", "bottom": "20px", "left": "20px"}
        )
        print(f"Exported PDF: {pdf_path}")
        
        await browser.close()
        
    return png_path, pdf_path

def main():
    """Export the realistic apple experiment sankey diagram."""
    
    # Paths
    project_root = Path(__file__).parent.parent.parent
    html_file = project_root / "experiments/apple_variety/results/apple_realistic/d3_sankey_full_network.html"
    output_dir = project_root / "arxiv_apple/figures"
    
    # Run export
    png_path, pdf_path = asyncio.run(
        export_sankey_to_images(html_file, output_dir)
    )
    
    print(f"\nExport complete!")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    
    # Also export to experiment results directory for convenience
    exp_output_dir = project_root / "experiments/apple_variety/results/apple_realistic"
    asyncio.run(
        export_sankey_to_images(html_file, exp_output_dir)
    )

if __name__ == "__main__":
    main()