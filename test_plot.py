#!/usr/bin/env python
"""
Script to test the plotting functionality for nightlights h5 files.
"""
import os
import matplotlib.pyplot as plt
from nightlights.plotting import plot_nightlights

# Path to the h5 file
H5_FILE = "data/testing/raw/VNP46A3_1/VNP46A3.A2024336.h12v12.001.2025014172348.h5"
# Variable to plot - common variables in VIIRS Black Marble data include:
# "DNB_BRDF-Corrected_NTL", "Gap_Filled_DNB_BRDF-Corrected_NTL", etc.
VARIABLE_NAME = "AllAngle_Composite_Snow_Free"
# Output directory for the plot
OUTPUT_DIR = "output/plots"

def main():
    """Run the plotting function and display the result."""
    # Ensure the file exists
    if not os.path.exists(H5_FILE):
        print(f"Error: File not found: {H5_FILE}")
        return
    
    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Plotting {VARIABLE_NAME} from {H5_FILE}...")
    
    # Plot the data
    fig, ax = plot_nightlights(
        file_path=H5_FILE,
        variable_name=VARIABLE_NAME,
        output_dir=OUTPUT_DIR,
        # cmap="cividis", # You can try other colormaps like 'plasma', 'inferno', 'magma', etc.
    )
    
    # Show the plot
    plt.show()
    
    print("Done!")

if __name__ == "__main__":
    main()
