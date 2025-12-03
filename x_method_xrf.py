import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt

# X-ray line energies database (keV)
XRAY_LINES = {
    'Si': {'Ka': 1.740, 'Kb': 1.836},
    'Cs': {'La': 4.287, 'Lb': 4.619, 'Ka': 30.973},
    'I': {'La': 3.938, 'Lb': 4.221, 'Ka': 28.612},
    'Ba': {'La': 4.466, 'Lb': 4.828, 'Ka': 32.194},
    'Cu': {'Ka': 8.048, 'Kb': 8.905},
    'Se': {'Ka': 11.222, 'Kb': 12.496},
    'Mo': {'Ka': 17.479, 'Kb': 19.608},
    'Rh': {'Ka': 20.216, 'Kb': 22.724},
}

ATOMIC_WEIGHTS = {
    'Si': 28.09, 'Cs': 132.91, 'I': 126.90,
    'Ba': 137.33, 'Cu': 63.55, 'Se': 78.96,
    'Mo': 95.96, 'Rh': 102.91
}

def read_spx_file(file_path):
    """Read SPX file and extract spectrum"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    root = ET.fromstring(content)
    
    # Get sample name
    spectrum_elem = root.find('.//ClassInstance[@Type="TRTSpectrum"]')
    sample_name = spectrum_elem.get('Name', 'Unknown')
    
    # Get calibration
    spec_header = root.find('.//ClassInstance[@Type="TRTSpectrumHeader"]')
    calib_abs = float(spec_header.findtext('CalibAbs', '-1'))
    calib_lin = float(spec_header.findtext('CalibLin', '0.01'))
    
    # Get spectrum data
    channels_elem = root.find('.//Channels')
    spectrum = np.array([int(x) for x in channels_elem.text.split(',')])
    
    # Calculate energy
    channels = np.arange(len(spectrum))
    energy = calib_abs + calib_lin * channels
    
    return {
        'sample_name': sample_name,
        'energy': energy,
        'spectrum': spectrum
    }

def identify_elements(energy, spectrum, energy_tolerance=0.15):
    """Identify elements by matching peaks to known X-ray lines"""
    
    # Smooth spectrum
    spectrum_smooth = savgol_filter(spectrum, 21, 3)
    
    # Find peaks
    peaks, _ = find_peaks(spectrum_smooth, prominence=50, distance=10)
    
    peak_energies = energy[peaks]
    peak_intensities = spectrum[peaks]
    
    # Match peaks to elements
    identified = {}
    
    for peak_energy, intensity in zip(peak_energies, peak_intensities):
        if peak_energy < 0.5:  # Skip noise
            continue
        
        # Match with known lines
        for element, lines in XRAY_LINES.items():
            for line_name, line_energy in lines.items():
                if abs(peak_energy - line_energy) < energy_tolerance:
                    if element not in identified:
                        identified[element] = {}
                    
                    identified[element][line_name] = {
                        'energy': peak_energy,
                        'intensity': intensity
                    }
    
    return identified

def calculate_composition(identified_elements, exclude_substrate=True):
    """Calculate composition from peak intensities"""
    
    # Get intensities for each element (use strongest line)
    intensities = {}
    
    for element, lines in identified_elements.items():
        # Skip substrate elements if requested
        if exclude_substrate and element in ['Si', 'Rh']:
            continue
        
        # Use La line if available, otherwise Ka
        if 'La' in lines:
            intensities[element] = lines['La']['intensity']
        elif 'Ka' in lines:
            intensities[element] = lines['Ka']['intensity']
    
    if not intensities:
        return None, None
    
    # Calculate weight percent (simplified - proportional to intensity)
    total_intensity = sum(intensities.values())
    composition_wt = {el: (inten / total_intensity) * 100 
                      for el, inten in intensities.items()}
    
    # Convert to atomic percent
    moles = {el: wt / ATOMIC_WEIGHTS[el] 
             for el, wt in composition_wt.items()}
    
    total_moles = sum(moles.values())
    composition_at = {el: (mol / total_moles) * 100 
                      for el, mol in moles.items()}
    
    return composition_wt, composition_at

def estimate_thickness(total_intensity, density=4.5):
    """Estimate thickness from total intensity (very simplified)"""
    # This is a rough empirical formula
    # Real calculation requires fundamental parameters
    thickness_nm = (total_intensity / 10000) * 100
    return thickness_nm

def plot_spectrum(energy, spectrum, identified_elements, save_path=None):
    """Plot spectrum with identified peaks"""
    plt.figure(figsize=(12, 6))
    plt.plot(energy, spectrum, 'b-', linewidth=0.5, alpha=0.7)
    
    # Mark identified peaks
    colors = {'Si': 'green', 'Cs': 'red', 'I': 'orange', 
              'Ba': 'purple', 'Cu': 'brown', 'Se': 'pink',
              'Mo': 'gray', 'Rh': 'black'}
    
    for element, lines in identified_elements.items():
        color = colors.get(element, 'blue')
        for line_name, data in lines.items():
            plt.axvline(data['energy'], color=color, 
                       linestyle='--', alpha=0.5, linewidth=1)
            plt.text(data['energy'], data['intensity'], 
                    f"{element}-{line_name}", 
                    rotation=90, fontsize=8, 
                    verticalalignment='bottom')
    
    plt.xlabel('Energy (keV)')
    plt.ylabel('Intensity (counts)')
    plt.title('XRF Spectrum with Identified Elements')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 35)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def analyze_spx(file_path):
    """Complete analysis workflow"""
    
    print("="*70)
    print("XRF SPX FILE ANALYSIS")
    print("="*70)
    
    # Read file
    data = read_spx_file(file_path)
    print(f"\nSample: {data['sample_name']}")
    print(f"Total counts: {np.sum(data['spectrum']):,}")
    
    # Identify elements
    identified = identify_elements(data['energy'], data['spectrum'])
    
    print("\nIdentified Elements:")
    print("-"*70)
    for element, lines in identified.items():
        print(f"\n{element}:")
        for line_name, line_data in lines.items():
            print(f"  {line_name}: {line_data['energy']:.3f} keV, "
                  f"Intensity: {line_data['intensity']:.0f} counts")
    
    # Calculate composition
    comp_wt, comp_at = calculate_composition(identified)
    
    if comp_wt:
        print("\n" + "="*70)
        print("COMPOSITION (excluding substrate)")
        print("="*70)
        
        print("\nWeight Percent:")
        for element, wt in comp_wt.items():
            print(f"  {element}: {wt:.2f} wt%")
        
        print("\nAtomic Percent:")
        for element, at in comp_at.items():
            print(f"  {element}: {at:.2f} at%")
        
        # Calculate stoichiometry
        if len(comp_at) == 2:
            elements = list(comp_at.keys())
            ratio = comp_at[elements[0]] / comp_at[elements[1]]
            print(f"\nStoichiometry: {elements[0]}:{elements[1]} = {ratio:.2f}:1")
        
        # Estimate thickness
        film_elements = {k: v for k, v in identified.items() 
                        if k not in ['Si', 'Rh']}
        total_film_intensity = sum(
            max(line['intensity'] for line in lines.values())
            for lines in film_elements.values()
        )
        
        thickness = estimate_thickness(total_film_intensity)
        print(f"\nEstimated thickness: {thickness:.1f} nm")
        print("(Note: This is a rough estimate)")
    
    # Plot
    plot_spectrum(
        data['energy'],
        data['spectrum'],
        identified,
        r'D:\High-throughput program\Mongo DB\New Versions\XRF Deployment\Extractred atomic percent\spx_analysis.png'
    )

    
    # Create results dataframe
    results = {
        'Sample': [data['sample_name']],
    }
    
    if comp_wt:
        for element in comp_wt.keys():
            results[f'{element}_wt%'] = [comp_wt[element]]
            results[f'{element}_at%'] = [comp_at[element]]
        results['Thickness_nm'] = [thickness]
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*70)
    print("RESULTS TABLE")
    print("="*70)
    print(df.to_string(index=False))
    
    return df

# Run analysis
df = analyze_spx(r'D:\High-throughput program\Mongo DB\New Versions\XRF Deployment\Example\Thomas data\4039-1\26.spx')

# Save to CSV
df.to_csv(r'D:\High-throughput program\Mongo DB\New Versions\XRF Deployment\Extractred atomic percent/spx_results.csv', index=False)
print("\nResults saved to: spx_results.csv")