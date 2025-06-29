# Manual Export Instructions for D3 Sankey Diagram

Since automated browser export requires additional setup, here are manual steps:

## Quick Export Steps

### For PNG Export:
1. Open in your browser:
   ```
   experiments/apple_variety/results/apple_realistic/d3_sankey_full_network.html
   ```

2. In Chrome/Edge:
   - Press `F12` to open Developer Tools
   - Press `Ctrl+Shift+P` (Windows) or `Cmd+Shift+P` (Mac)
   - Type "screenshot" and select "Capture full size screenshot"
   - Save as `apple_sankey_full_network.png` in `arxiv_apple/figures/`

3. In Firefox:
   - Press `Shift+F2` to open developer toolbar
   - Type: `screenshot --fullpage`
   - Save to `arxiv_apple/figures/`

### For PDF Export:
1. Open the HTML file in your browser
2. Press `Ctrl+P` (Windows) or `Cmd+P` (Mac)
3. Configure print settings:
   - Destination: "Save as PDF"
   - Layout: **Landscape**
   - Margins: "None" or "Minimum"
   - Options: ✓ "Background graphics"
4. Save as `apple_sankey_full_network.pdf` in `arxiv_apple/figures/`

## Alternative: Online Converters

If browser tools don't work well:

1. **For high-quality PNG**: 
   - Open https://html2canvas.hertzen.com/
   - Upload the HTML file
   - Download the resulting image

2. **For vector PDF**:
   - Open https://www.web2pdfconvert.com/
   - Enter the file:// URL of your HTML
   - Convert and download

## Expected Output

The sankey diagram should show:
- 4 layers (L0 → L1 → L2 → L3)
- Apple routing paths with colors
- Title: "Apple Routing Network - Full Network View"
- Subtitle with accuracy (92.8%) and economic loss ($186.41)

## File Locations

- **Source HTML**: `experiments/apple_variety/results/apple_realistic/d3_sankey_full_network.html`
- **Output PNG**: `arxiv_apple/figures/apple_sankey_full_network.png`
- **Output PDF**: `arxiv_apple/figures/apple_sankey_full_network.pdf`