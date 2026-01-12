import streamlit as st
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path
import base64
import struct
from typing import Dict, List, Optional


# ============================================================================
# CONSTANTS
# ============================================================================

ELEMENT_SYMBOLS = {
    13: 'Al', 22: 'Ti', 24: 'Cr', 26: 'Fe', 27: 'Co', 28: 'Ni',
    29: 'Cu', 42: 'Mo', 45: 'Rh', 46: 'Pd', 47: 'Ag', 74: 'W', 79: 'Au'
}



# ============================================================================
# ATOMIC MASS DATA
# ============================================================================

ATOMIC_MASSES = {
    'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.81,
    'C': 12.01, 'N': 14.01, 'O': 16.00, 'F': 19.00, 'Ne': 20.18,
    'Na': 22.99, 'Mg': 24.31, 'Al': 26.98, 'Si': 28.09, 'P': 30.97,
    'S': 32.07, 'Cl': 35.45, 'Ar': 39.95, 'K': 39.10, 'Ca': 40.08,
    'Sc': 44.96, 'Ti': 47.87, 'V': 50.94, 'Cr': 52.00, 'Mn': 54.94,
    'Fe': 55.85, 'Co': 58.93, 'Ni': 58.69, 'Cu': 63.55, 'Zn': 65.38,
    'Ga': 69.72, 'Ge': 72.63, 'As': 74.92, 'Se': 78.96, 'Br': 79.90,
    'Kr': 83.80, 'Rb': 85.47, 'Sr': 87.62, 'Y': 88.91, 'Zr': 91.22,
    'Nb': 92.91, 'Mo': 95.96, 'Tc': 98.00, 'Ru': 101.1, 'Rh': 102.9,
    'Pd': 106.4, 'Ag': 107.9, 'Cd': 112.4, 'In': 114.8, 'Sn': 118.7,
    'Sb': 121.8, 'Te': 127.6, 'I': 126.9, 'Xe': 131.3, 'Cs': 132.9,
    'Ba': 137.3, 'La': 138.9, 'Ce': 140.1, 'Pr': 140.9, 'Nd': 144.2,
    'Pm': 145.0, 'Sm': 150.4, 'Eu': 152.0, 'Gd': 157.3, 'Tb': 158.9,
    'Dy': 163.5, 'Ho': 164.9, 'Er': 167.3, 'Tm': 168.9, 'Yb': 173.0,
    'Lu': 175.0, 'Hf': 178.5, 'Ta': 180.9, 'W': 183.8, 'Re': 186.2,
    'Os': 190.2, 'Ir': 192.2, 'Pt': 195.1, 'Au': 197.0, 'Hg': 200.6,
    'Tl': 204.4, 'Pb': 207.2, 'Bi': 209.0, 'Po': 209.0, 'At': 210.0,
    'Rn': 222.0, 'Fr': 223.0, 'Ra': 226.0, 'Ac': 227.0, 'Th': 232.0,
    'Pa': 231.0, 'U': 238.0, 'Np': 237.0, 'Pu': 244.0
}


# ============================================================================
# WEIGHT % TO ATOMIC % CONVERSION
# ============================================================================

def convert_wt_to_at_percent(composition_wt: Dict[str, float]) -> Dict[str, float]:
    """
    Convert weight percentage to atomic percentage
    
    Args:
        composition_wt: Dictionary of {element: weight%}
        
    Returns:
        Dictionary of {element: atomic%}
    """
    if not composition_wt:
        return {}
    
    # Calculate molar ratios (wt% / atomic_mass)
    molar_ratios = {}
    for element, wt_pct in composition_wt.items():
        # Clean element symbol (remove any spaces, brackets)
        element_clean = element.strip()
        
        # Get atomic mass
        atomic_mass = ATOMIC_MASSES.get(element_clean)
        if atomic_mass is None:
            # Try common variations
            element_clean = element_clean.replace('[]', '').replace('[', '').replace(']', '').strip()
            atomic_mass = ATOMIC_MASSES.get(element_clean)
        
        if atomic_mass is None:
            st.warning(f"Unknown element: {element}. Skipping in atomic % calculation.")
            continue
        
        if wt_pct > 0:
            molar_ratios[element] = wt_pct / atomic_mass
    
    # Calculate sum of molar ratios
    total_molar = sum(molar_ratios.values())
    
    if total_molar == 0:
        return {}
    
    # Calculate atomic percentages
    composition_at = {}
    for element, molar_ratio in molar_ratios.items():
        composition_at[element] = (molar_ratio / total_molar) * 100.0
    
    return composition_at

#####################################################


# ============================================================================
# XML PARSING UTILITIES
# ============================================================================

def _localname(tag: str) -> str:
    """Extract local name from XML tag (remove namespace)"""
    return tag.split('}', 1)[-1] if '}' in tag else tag


def get_text(element, path: str, default=''):
    """Safely extract text from XML element"""
    if element is None:
        return default
    found = element.find(path)
    return found.text.strip() if found is not None and found.text else default


def get_float(element, path: str, default=0.0):
    """Safely extract float from XML element"""
    try:
        text = get_text(element, path, str(default))
        return float(text)
    except (ValueError, TypeError):
        return default


def get_int(element, path: str, default=0):
    """Safely extract integer from XML element"""
    try:
        text = get_text(element, path, str(default))
        return int(float(text))
    except (ValueError, TypeError):
        return default


# ============================================================================
# COORDINATE EXTRACTION
# ============================================================================

def _length_to_mm(val: float, unit: str = None) -> float:
    """Convert length value to millimeters"""
    if unit is None:
        return float(val)
    u = unit.strip().lower()
    conversion = {
        'mm': 1.0, 'µm': 0.001, 'μm': 0.001, 'um': 0.001,
        'cm': 10.0, 'm': 1000.0
    }
    return float(val) * conversion.get(u, 1.0)


def find_xyz_from_blob(blob: bytes):
    """Extract XYZ coordinates from binary blob"""
    best = None
    for off in range(0, len(blob) - 24 + 1):
        try:
            x, y, z = struct.unpack("<ddd", blob[off:off + 24])
        except struct.error:
            continue
        if all(np.isfinite([x, y, z])) and all(0 <= v <= 1000 for v in (x, y, z)):
            if all(v > 0 for v in (x, y, z)):
                best = (x, y, z)
    return best


def extract_xyz_positions_from_spx(root: ET.Element):
    """Extract X, Y, Z positions from SPX file in millimeters"""
    positions = {}

    # Method 1: Look for Axis elements
    for elem in root.iter():
        tag_local = _localname(elem.tag)
        if tag_local.startswith("Axis"):
            name = elem.attrib.get("AxisName", tag_local)
            unit = elem.attrib.get("AxisUnit", "")
            pos = elem.attrib.get("AxisPosition")
            if pos is not None:
                try:
                    positions[name] = {"position": float(pos), "unit": unit}
                except ValueError:
                    pass

    if positions:
        x_data = positions.get("X") or next((v for k, v in positions.items() if k.upper().startswith("X")), None)
        y_data = positions.get("Y") or next((v for k, v in positions.items() if k.upper().startswith("Y")), None)
        z_data = positions.get("Z") or next((v for k, v in positions.items() if k.upper().startswith("Z")), None)

        x_mm = _length_to_mm(x_data["position"], x_data.get("unit")) if x_data else None
        y_mm = _length_to_mm(y_data["position"], y_data.get("unit")) if y_data else None
        z_mm = _length_to_mm(z_data["position"], z_data.get("unit")) if z_data else None

        return x_mm, y_mm, z_mm

    # Method 2: Look for Data blob
    for node in root.iter():
        if _localname(node.tag) != "Data":
            continue
        b64 = (node.text or "").strip()
        if len(b64) < 8:
            continue
        try:
            blob = base64.b64decode(b64, validate=True)
            xyz = find_xyz_from_blob(blob)
            if xyz:
                return float(xyz[0]), float(xyz[1]), float(xyz[2])
        except Exception:
            continue

    return None, None, None


# ============================================================================
# SPECTRUM DATA PARSING
# ============================================================================

def parse_date_time(date_str: str, time_str: str):
    """Parse date and time strings to datetime object"""
    try:
        day, month, year = date_str.split('.')
        hour, minute, second = time_str.split(':')
        return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    except (ValueError, AttributeError):
        return datetime.now()


def parse_spectrum_data(channels_text: str):
    """Parse spectrum channel data from comma-separated string"""
    try:
        return np.array([int(x) for x in channels_text.split(',')])
    except (ValueError, TypeError):
        return np.array([])


def parse_spx_file(file_path: Path) -> Dict:
    """Parse SPX file and extract all relevant information"""
    with open(file_path, 'rb') as f:
        file_content = f.read()

    root = ET.fromstring(file_content)
    spectrum = root.find(".//ClassInstance[@Type='TRTSpectrum']")

    if spectrum is None:
        raise ValueError("Invalid SPX file")

    header = spectrum.find(".//ClassInstance[@Type='TRTSpectrumHeader']")
    hardware = spectrum.find(".//ClassInstance[@Type='TRTSpectrumHardwareHeader']")
    xrf_header = spectrum.find(".//ClassInstance[@Type='TRTXrfHeader']")

    # Date and time
    date_str = get_text(header, 'Date')
    time_str = get_text(header, 'Time')
    measurement_date = parse_date_time(date_str, time_str)

    # Spectrum data
    channels_element = spectrum.find('Channels')
    spectrum_data = parse_spectrum_data(channels_element.text if channels_element is not None else '')

    # Calibration
    calib_abs = get_float(header, 'CalibAbs', -0.956)
    calib_lin = get_float(header, 'CalibLin', 0.01)
    channel_count = get_int(header, 'ChannelCount', 4096)

    # X-ray tube parameters
    tube_voltage = get_float(xrf_header, 'Voltage', 0) if xrf_header is not None else 0
    tube_current = get_float(xrf_header, 'Current', 0) if xrf_header is not None else 0
    tube_anode = get_int(xrf_header, 'Anode', 45) if xrf_header is not None else 45

    xray_tube_target = ELEMENT_SYMBOLS.get(tube_anode, 'Unknown')
    chassis_type = get_text(xrf_header, 'ChassisType') if xrf_header is not None else ''

    # Position
    x_position, y_position, z_position = extract_xyz_positions_from_spx(root)

    # Ensure spectrum data matches channel count
    if len(spectrum_data) > channel_count:
        spectrum_data = spectrum_data[:channel_count]
    elif len(spectrum_data) < channel_count:
        padded = np.zeros(channel_count, dtype=int)
        padded[:len(spectrum_data)] = spectrum_data
        spectrum_data = padded

    result = {
        'file_name': file_path.name,
        'date': date_str,
        'time': time_str,
        'measurement_date': measurement_date,
        'real_time_ms': int(get_int(hardware, 'RealTime', 0)),
        'live_time_ms': int(get_int(hardware, 'LifeTime', 0)),
        'dead_time_percent': float(get_float(hardware, 'DeadTime', 0)),
        'xray_tube_target': xray_tube_target,
        'voltage_kV': float(tube_voltage),
        'current_uA': float(tube_current),
        'calib_abs_kev': float(calib_abs),
        'calib_lin_kev_per_channel': float(calib_lin),
        'channel_count': int(channel_count),
        'detector_count': int(get_int(hardware, 'DetectorCount', 0)),
        'selected_detectors': str(get_text(hardware, 'SelectedDetectors')).replace('\t', '&').replace(' ', '&').replace('\n', '&'),
        'total_counts': int(spectrum_data.sum()),
        'max_counts': int(spectrum_data.max()),
        'spectrum_data': spectrum_data,
        'chassis_type': chassis_type,
        'x_position_mm': x_position,
        'y_position_mm': y_position,
        'z_position_mm': z_position,
    }

    # Calculate energy at max counts
    energy_axis = calib_abs + calib_lin * np.arange(channel_count)
    result['max_counts_energy_kev'] = float(energy_axis[spectrum_data.argmax()])

    return result


# ============================================================================
# XRF XLS FILE PARSING
# ============================================================================

def read_xrf_xls(uploaded_file) -> pd.DataFrame:
    """
    Read XRF XLS file with multi-layer structure (Bruker M4 format)
    Returns DataFrame with one row per spectrum per layer
    """
    try:
        # Read XLS file
        try:
            df_raw = pd.read_excel(uploaded_file, engine='openpyxl', header=None)
        except:
            df_raw = pd.read_excel(uploaded_file, engine='xlrd', header=None)
        
        # Row 0: Layer labels (Layer 1, Layer 2, etc.)
        layer_row = df_raw.iloc[0].values
        # Row 1: Column headers (Spectrum, Si [%], Thickn. [nm], etc.)
        headers = df_raw.iloc[1].values
        # Data starts from row 2
        data = df_raw.iloc[2:].reset_index(drop=True)
        
        # Find layer column indices
        layer_indices = [i for i, layer_name in enumerate(layer_row) 
                        if pd.notna(layer_name) and 'Layer' in str(layer_name)]
        
        # Parse all rows
        all_rows = []
        
        for row_idx in range(len(data)):
            spectrum = data.iloc[row_idx, 0]
            
            # Skip empty or summary rows
            if pd.isna(spectrum):
                continue
            spectrum = str(spectrum).strip()
            if not spectrum or any(x in spectrum.lower() for x in ['mean', 'average', 'std', 'spectrum']):
                continue
            
            # Process each layer
            for layer_num, layer_idx in enumerate(layer_indices, 1):
                # Determine column range for this layer
                end_idx = layer_indices[layer_num] if layer_num < len(layer_indices) else len(headers)
                
                thickness = None
                composition = {}
                
                # Extract data for this layer
                for col_idx in range(layer_idx, end_idx):
                    header = headers[col_idx]
                    if pd.isna(header):
                        continue
                    
                    header_str = str(header)
                    value = data.iloc[row_idx, col_idx]
                    
                    if pd.isna(value):
                        continue
                    
                    # Convert to float, skip if not numeric
                    try:
                        value_float = float(value)
                    except (ValueError, TypeError):
                        continue
                    
                    # Check if thickness
                    if 'Thickn' in header_str or 'thickness' in header_str.lower():
                        thickness = value_float
                    # Check if composition
                    elif '[%]' in header_str or '(%)' in header_str:
                        element = header_str.replace('[%]', '').replace('(%)', '').strip()
                        composition[element] = value_float
                
                # Add row
                row_data = {
                    'spectrum': spectrum,
                    'layer': f'layer{layer_num}',
                    'thickness_nm': thickness
                }
                
                for elem, val in composition.items():
                    row_data[f'{elem} (%)'] = val
                
                all_rows.append(row_data)
        
        if not all_rows:
            raise ValueError("No valid data extracted from XLS file")
        
        return pd.DataFrame(all_rows)
        
    except Exception as e:
        import traceback
        raise ValueError(f"Error reading XRF XLS: {str(e)}\n{traceback.format_exc()}")


def parse_xrf_xls_to_dict(xrf_df: pd.DataFrame) -> List[Dict]:
    """
    Convert XRF DataFrame to list of dicts
    Groups layers by spectrum name
    """
    xrf_data_list = []
    composition_cols = [col for col in xrf_df.columns if '(%)' in col]
    
    for _, row in xrf_df.iterrows():
        spectrum_name = str(row['spectrum'])
        layer_name = str(row['layer'])
        thickness_nm = row['thickness_nm'] if pd.notna(row['thickness_nm']) else None
        
        composition = {col.replace('(%)', '').strip(): float(row[col]) 
                      for col in composition_cols if pd.notna(row[col])}
        
        xrf_data_list.append({
            'spectrum_name': spectrum_name,
            'thickness_nm': thickness_nm,
            'composition': composition,
            'layer_name': layer_name
        })
    
    return xrf_data_list


# ============================================================================
# SPECTRUM MATCHING
# ============================================================================

def match_spx_with_xrf_csv(spx_data_list: List[Dict], xrf_data_list: List[Dict]) -> List[Dict]:
    """
    Match SPX files with XRF data
    Returns one entry per SPX per layer
    """
    import re
    combined_data = []
    
    # Group XRF data by spectrum name
    xrf_by_spectrum = {}
    for xrf in xrf_data_list:
        spectrum = xrf['spectrum_name']
        if spectrum not in xrf_by_spectrum:
            xrf_by_spectrum[spectrum] = []
        xrf_by_spectrum[spectrum].append(xrf)
    
    for spx_data in spx_data_list:
        spx_name = spx_data['file_name']
        spx_base = Path(spx_name).stem
        
        # Try to find match
        xrf_matches = None
        match_method = None
        
        # Method 1: Direct name match
        for xrf_name in xrf_by_spectrum.keys():
            xrf_base = Path(xrf_name).stem if '.spx' in xrf_name else xrf_name
            if spx_base == xrf_base:
                xrf_matches = xrf_by_spectrum[xrf_name]
                match_method = "exact_name"
                break
        
        # Method 2: Partial name match
        if xrf_matches is None:
            spx_clean = spx_base.replace('-', '').replace('_', '').replace(' ', '').lower()
            for xrf_name in xrf_by_spectrum.keys():
                xrf_clean = xrf_name.replace('-', '').replace('_', '').replace(' ', '').replace('.spx', '').lower()
                if spx_clean in xrf_clean or xrf_clean in spx_clean:
                    xrf_matches = xrf_by_spectrum[xrf_name]
                    match_method = "partial_name"
                    break
        
        # Method 3: Extract Grid number
        if xrf_matches is None:
            grid_match = re.search(r'Grid[_\s]*(\d+)', spx_base, re.IGNORECASE)
            if grid_match:
                grid_num = grid_match.group(0)
                for xrf_name in xrf_by_spectrum.keys():
                    if grid_num in xrf_name:
                        xrf_matches = xrf_by_spectrum[xrf_name]
                        match_method = "grid_number"
                        break
        
        # Create combined data - one entry per layer
        if xrf_matches:
            for xrf_match in xrf_matches:
                combined = {
                    'spx_name': spx_name,
                    'xrf_spectrum_name': xrf_match['spectrum_name'],
                    'layer_name': xrf_match['layer_name'],
                    'date': spx_data['date'],
                    'time': spx_data['time'],
                    'x_position_mm': spx_data['x_position_mm'],
                    'y_position_mm': spx_data['y_position_mm'],
                    'z_position_mm': spx_data['z_position_mm'],
                    'thickness_nm': xrf_match['thickness_nm'],
                    'composition': xrf_match['composition'],
                    'spectrum_data_spx': spx_data['spectrum_data'],
                    'calib_abs_kev': spx_data['calib_abs_kev'],
                    'calib_lin_kev_per_channel': spx_data['calib_lin_kev_per_channel'],
                    'matched': True,
                    'match_method': match_method,
                    'correlation': 1.0,
                    'real_time_ms': spx_data['real_time_ms'],
                    'live_time_ms': spx_data['live_time_ms'],
                    'dead_time_percent': spx_data['dead_time_percent'],
                    'xray_tube_target': spx_data['xray_tube_target'],
                    'voltage_kV': spx_data['voltage_kV'],
                    'current_uA': spx_data['current_uA'],
                    'total_counts': spx_data['total_counts'],
                    'max_counts': spx_data['max_counts'],
                }
                combined_data.append(combined)
        else:
            # No match found
            combined = {
                'spx_name': spx_name,
                'xrf_spectrum_name': None,
                'layer_name': 'unknown',
                'date': spx_data['date'],
                'time': spx_data['time'],
                'x_position_mm': spx_data['x_position_mm'],
                'y_position_mm': spx_data['y_position_mm'],
                'z_position_mm': spx_data['z_position_mm'],
                'thickness_nm': None,
                'composition': {},
                'spectrum_data_spx': spx_data['spectrum_data'],
                'calib_abs_kev': spx_data['calib_abs_kev'],
                'calib_lin_kev_per_channel': spx_data['calib_lin_kev_per_channel'],
                'matched': False,
                'match_method': None,
                'correlation': 0.0,
                'real_time_ms': spx_data['real_time_ms'],
                'live_time_ms': spx_data['live_time_ms'],
                'dead_time_percent': spx_data['dead_time_percent'],
                'xray_tube_target': spx_data['xray_tube_target'],
                'voltage_kV': spx_data['voltage_kV'],
                'current_uA': spx_data['current_uA'],
                'total_counts': spx_data['total_counts'],
                'max_counts': spx_data['max_counts'],
            }
            combined_data.append(combined)
    
    return combined_data


# ============================================================================
# COORDINATE CONVERSION
# ============================================================================

def extract_xrf_bounds(combined_data: List[Dict]) -> Optional[Dict]:
    """Extract first and last XRF measurement coordinates"""
    if not combined_data:
        return None
    
    first_data = combined_data[0]
    last_data = combined_data[-1]
    
    x_1, y_1 = first_data['x_position_mm'], first_data['y_position_mm']
    x_2, y_2 = last_data['x_position_mm'], last_data['y_position_mm']
    
    if all(v is not None for v in [x_1, y_1, x_2, y_2]):
        return {'x_1': x_1, 'y_1': y_1, 'x_2': x_2, 'y_2': y_2}
    return None


def convert_xrf_to_optical(combined_data: List[Dict], xrf_bounds: Dict, optical_bounds: Dict) -> List[Dict]:
    """Convert XRF coordinates to optical coordinates"""
    A_x_1, A_y_1 = xrf_bounds['x_1'], xrf_bounds['y_1']
    A_x_2, A_y_2 = xrf_bounds['x_2'], xrf_bounds['y_2']
    B_x_1, B_y_1 = optical_bounds['x_1'], optical_bounds['y_1']
    B_x_2, B_y_2 = optical_bounds['x_2'], optical_bounds['y_2']
    
    converted_data = []
    
    for data in combined_data:
        x_xrf, y_xrf = data['x_position_mm'], data['y_position_mm']
        
        if x_xrf is None or y_xrf is None:
            converted_data.append({
                'spx_name': data['spx_name'],
                'x_xrf': 0, 'y_xrf': 0, 'x_optical': 0, 'y_optical': 0
            })
            continue
        
        # Linear mapping
        t_x = (x_xrf - A_x_1) / (A_x_2 - A_x_1) if A_x_2 != A_x_1 else 0
        t_y = (y_xrf - A_y_1) / (A_y_2 - A_y_1) if A_y_2 != A_y_1 else 0
        
        x_optical = B_x_1 + t_x * (B_x_2 - B_x_1)
        y_optical = B_y_1 + t_y * (B_y_2 - B_y_1)
        
        converted_data.append({
            'spx_name': data['spx_name'],
            'x_xrf': x_xrf, 'y_xrf': y_xrf,
            'x_optical': x_optical, 'y_optical': y_optical
        })
    
    return converted_data


# ============================================================================
# CSV EXPORT - COMBINED ALL LAYERS
# ============================================================================

def create_combined_csv_horizontal_layers(combined_data: List[Dict], metadata: Dict,
                                          converted_data: Optional[List[Dict]] = None) -> str:
    """Create CSV with one row per SPX file, all layers as horizontal columns with both wt% and at%"""
    csv_lines = []
    
    # Metadata section
    csv_lines.extend([
        "Substrate Number," + metadata['substrate_number'],
        "Substrate," + metadata['substrate'],
        "Sample Description," + metadata['sample_description'],
        "Substrate Size (mm)," + metadata['substrate_size'],
        "Fabrication Method," + metadata['fabrication_method'],
        "Treatment Method," + metadata['treatment_method'],
        "Treatment Sequence," + metadata['treatment_sequence'],
        "Air exposure Duration (min)," + metadata['air_exposure_duration'],
        "Operator," + metadata['operator'],
        "Institution," + metadata['institution'],
        "Measurement Type," + metadata['measurement_type'],
        "Spectrometer," + metadata.get('spectrometer', 'Bruker M4 Tornado'),
        "XRF Fitting Method," + metadata.get('xrf_fitting_method', 'Series')
    ])
    
    if combined_data:
        # Calculate medians
        medians = {
            'real_time': int(np.median([d['real_time_ms'] for d in combined_data])),
            'live_time': int(np.median([d['live_time_ms'] for d in combined_data])),
            'dead_time': float(np.median([d['dead_time_percent'] for d in combined_data])),
            'voltage': float(np.median([d['voltage_kV'] for d in combined_data])),
            'current': float(np.median([d['current_uA'] for d in combined_data])),
            'calib_abs': float(np.median([d['calib_abs_kev'] for d in combined_data])),
            'calib_lin': float(np.median([d['calib_lin_kev_per_channel'] for d in combined_data])),
        }
        
        csv_lines.extend([
            "real_time_ms," + str(medians['real_time']),
            "live_time_ms," + str(medians['live_time']),
            "dead_time_percent," + str(medians['dead_time']),
            "voltage_kV," + str(medians['voltage']),
            "current_micro_A," + str(medians['current']),
            "calib_abs_kev," + str(medians['calib_abs']),
            "calib_lin_kev_per_channel," + str(medians['calib_lin'])
        ])
        
        # Group by SPX file
        spx_groups = {}
        for data in combined_data:
            spx_name = data['spx_name']
            if spx_name not in spx_groups:
                spx_groups[spx_name] = []
            spx_groups[spx_name].append(data)
        
        # Determine layers and elements per layer
        unique_layers = set(data['layer_name'] for data in combined_data)
        layer_elements = {}
        
        for data in combined_data:
            layer = data['layer_name']
            if layer not in layer_elements:
                layer_elements[layer] = set()
            if data['composition']:
                layer_elements[layer].update(data['composition'].keys())
        
        sorted_layers = sorted(list(unique_layers))
        layer_elements_sorted = {layer: sorted(list(layer_elements.get(layer, []))) 
                                for layer in sorted_layers}
        
        csv_lines.append("total_material_layer," + str(len(unique_layers)))
        
        # Build Layer header row
        layer_header = ["", "", "", "", ""]
        for layer in sorted_layers:
            layer_num = layer.replace('layer', '')
            layer_header.append(f"Layer {layer_num}")
            # Now we need columns for: thickness + (wt% + at%) per element
            num_columns = 1 + len(layer_elements_sorted[layer])
            layer_header.extend([""] * (num_columns - 1))
        
        csv_lines.append(",".join(layer_header))
        
        # Build column headers
        column_headers = ["x position (mm)", "y position (mm)", "spectrum", "date", "time"]
        
        for layer in sorted_layers:
            column_headers.append("thickness (nm)")
            for elem in layer_elements_sorted[layer]:
                clean_elem = elem.replace('[]', '').replace('[', '').replace(']', '').strip()
                column_headers.append(f"{clean_elem} (%)")
        
        # Add energy columns
        channel_count = len(combined_data[0]['spectrum_data_spx'])
        energy_axis = medians['calib_abs'] + medians['calib_lin'] * np.arange(channel_count)
        energy_headers = [f"{e:.6f}" for e in energy_axis]
        
        header_row = ",".join(column_headers) + "," + ",".join(energy_headers)
        csv_lines.append(header_row)
        
        # Create coordinate lookup
        converted_lookup = {c['spx_name']: c for c in (converted_data or [])}
        
        # Data rows - one per SPX file
        for spx_name in sorted(spx_groups.keys()):
            layers_data = spx_groups[spx_name]
            first_entry = layers_data[0]
            
            # Get coordinates
            if spx_name in converted_lookup:
                x_pos = converted_lookup[spx_name]['x_optical']
                y_pos = converted_lookup[spx_name]['y_optical']
            else:
                x_pos = first_entry['x_position_mm'] if first_entry['x_position_mm'] is not None else 0
                y_pos = first_entry['y_position_mm'] if first_entry['y_position_mm'] is not None else 0
            
            row_data = [
                f"{x_pos:.3f}", f"{y_pos:.3f}", spx_name,
                first_entry['date'], first_entry['time']
            ]
            
            # Create layer lookup
            layer_lookup = {ld['layer_name']: ld for ld in layers_data}
            
            # Add data for each layer
            for layer in sorted_layers:
                if layer in layer_lookup:
                    ld = layer_lookup[layer]
                    row_data.append(f"{ld['thickness_nm']:.2f}" if ld['thickness_nm'] is not None else "")
                    
                    # Convert wt% to at%
                    composition_at = convert_wt_to_at_percent(ld['composition'])
                    
                    for elem in layer_elements_sorted[layer]:
                        # Atomic percentage only
                        at_value = composition_at.get(elem, 0)
                        row_data.append(f"{at_value:.2f}" if at_value > 0 else "0")
                else:
                    # Empty layer: empty thickness + empty wt% and at% for all elements
                    row_data.extend([""] * (1 + len(layer_elements_sorted[layer])))
            
            # Add spectrum data
            spectrum_data = first_entry['spectrum_data_spx'][:channel_count]
            counts_str = ",".join([str(int(c)) for c in spectrum_data])
            
            data_row = ",".join(row_data) + "," + counts_str
            csv_lines.append(data_row)
    
    return "\n".join(csv_lines)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_coordinate_comparison(xrf_bounds: Dict, optical_bounds: Dict, converted_data: List[Dict]):
    """Create visualization comparing XRF and optical coordinates"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("XRF Coordinates", "Optical Coordinates"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # XRF plot
    x_1_xrf, y_1_xrf = xrf_bounds['x_1'], xrf_bounds['y_1']
    x_2_xrf, y_2_xrf = xrf_bounds['x_2'], xrf_bounds['y_2']
    x_min_xrf, x_max_xrf = min(x_1_xrf, x_2_xrf), max(x_1_xrf, x_2_xrf)
    y_min_xrf, y_max_xrf = min(y_1_xrf, y_2_xrf), max(y_1_xrf, y_2_xrf)
    
    x_rect_xrf = [x_min_xrf, x_max_xrf, x_max_xrf, x_min_xrf, x_min_xrf]
    y_rect_xrf = [y_min_xrf, y_min_xrf, y_max_xrf, y_max_xrf, y_min_xrf]
    
    fig.add_trace(go.Scatter(
        x=x_rect_xrf, y=y_rect_xrf, mode="lines",
        line=dict(color="black", width=2), name="XRF Bounds", showlegend=True
    ), row=1, col=1)
    
    x_points_xrf = [d['x_xrf'] for d in converted_data]
    y_points_xrf = [d['y_xrf'] for d in converted_data]
    
    fig.add_trace(go.Scatter(
        x=x_points_xrf, y=y_points_xrf, mode="markers+lines",
        marker=dict(size=8, color=np.arange(len(converted_data)), colorscale="Viridis"),
        line=dict(width=1.5), name="XRF Points", showlegend=True
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=[x_1_xrf], y=[y_1_xrf], mode="markers+text",
        marker=dict(size=15, color="red", symbol="circle"),
        text=["First Point"], textposition="top center",
        name="First Point (XRF)", showlegend=True
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=[x_2_xrf], y=[y_2_xrf], mode="markers+text",
        marker=dict(size=15, color="blue", symbol="circle"),
        text=["Last Point"], textposition="top center",
        name="Last Point (XRF)", showlegend=True
    ), row=1, col=1)
    
    # Optical plot
    x_1_opt, y_1_opt = optical_bounds['x_1'], optical_bounds['y_1']
    x_2_opt, y_2_opt = optical_bounds['x_2'], optical_bounds['y_2']
    x_min_opt, x_max_opt = min(x_1_opt, x_2_opt), max(x_1_opt, x_2_opt)
    y_min_opt, y_max_opt = min(y_1_opt, y_2_opt), max(y_1_opt, y_2_opt)
    
    x_rect_opt = [x_min_opt, x_max_opt, x_max_opt, x_min_opt, x_min_opt]
    y_rect_opt = [y_min_opt, y_min_opt, y_max_opt, y_max_opt, y_min_opt]
    
    fig.add_trace(go.Scatter(
        x=x_rect_opt, y=y_rect_opt, mode="lines",
        line=dict(color="black", width=2), name="Optical Bounds", showlegend=True
    ), row=1, col=2)
    
    x_points_opt = [d['x_optical'] for d in converted_data]
    y_points_opt = [d['y_optical'] for d in converted_data]
    
    fig.add_trace(go.Scatter(
        x=x_points_opt, y=y_points_opt, mode="markers+lines",
        marker=dict(size=8, color=np.arange(len(converted_data)), colorscale="Viridis"),
        line=dict(width=1.5), name="Optical Points", showlegend=True
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=[x_1_opt], y=[y_1_opt], mode="markers+text",
        marker=dict(size=15, color="red", symbol="circle"),
        text=["First Point"], textposition="top center",
        name="First Point (Optical)", showlegend=True
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=[x_2_opt], y=[y_2_opt], mode="markers+text",
        marker=dict(size=15, color="blue", symbol="circle"),
        text=["Last Point"], textposition="top center",
        name="Last Point (Optical)", showlegend=True
    ), row=1, col=2)
    
    # XRF axes: reversed
    fig.update_xaxes(title_text="X (mm)", autorange="reversed", scaleanchor="y", scaleratio=1, row=1, col=1)
    fig.update_yaxes(title_text="Y (mm)", autorange="reversed", row=1, col=1)
    
    # Optical axes: normal
    fig.update_xaxes(title_text="X (mm)", scaleanchor="y2", scaleratio=1, row=1, col=2)
    fig.update_yaxes(title_text="Y (mm)", row=1, col=2)
    
    fig.update_layout(height=500, template="plotly_white")
    return fig


def plot_spectrum(data: Dict, title: str, yscale: str = "linear"):
    """Plot SPX spectrum"""
    calib_abs = data['calib_abs_kev']
    calib_lin = data['calib_lin_kev_per_channel']
    spectrum_spx = data['spectrum_data_spx']
    
    energy_axis = calib_abs + calib_lin * np.arange(len(spectrum_spx))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=energy_axis, y=spectrum_spx, mode='lines',
        name='SPX', line=dict(color='royalblue', width=1.5)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Energy (keV)",
        yaxis_title="Counts",
        height=400,
        template='plotly_white',
        showlegend=True,
        yaxis=dict(type=yscale)
    )
    
    return fig



# ============================================================================
# METADATA UI
# ============================================================================

def render_metadata_section():
    """Render metadata input section"""
    
    # Initialize session state defaults
    defaults = {
        'substrate_number': "3716-15",
        'operator': "",
        'operator_valid': False,
        'institution': "HZB",
        'xrf_fitting_method': "Series"
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        substrate_number = st.text_input("Substrate Number", value=st.session_state.substrate_number)
        substrate = st.selectbox("Substrate", 
            ["Silicon", "Glass", "Quartz", "Plastic", "Metal", "Other"], index=0)
        sample_description = st.text_area(
            "Sample Description",
            value="Perovskite solar cell with full stacks",
            max_chars=500,
            placeholder="Describe components, materials, layers of your samples"
        )
        substrate_size = st.text_input("Substrate Size (mm)", value="50x50")
        
        st.session_state.update({
            'substrate_number': substrate_number, 'substrate': substrate,
            'sample_description': sample_description, 'substrate_size': substrate_size
        })
    
    with col2:
        fabrication_method = st.selectbox("Fabrication Method", 
            ["PVD-J", "Sputtering", "Tube Furnace", "RTP", "PLD", "PVD-P"], 
            index=1)
        treatment_method = st.selectbox("Treatment Method", 
            ["As-deposited", "Annealing", "UV-Ozone", "Irradiation", "Plasma"], index=0)
        
        if treatment_method == "As-deposited":
            treatment_sequence = "0"
            st.info(f"Treatment sequence: {treatment_sequence}")
        else:
            treatment_sequence = str(st.number_input(
                "Treatment Sequence", min_value=1, max_value=10, value=1, step=1
            ))
        
        air_exposure_duration = st.text_input("Air exposure Duration (min)", value="30")
        
        st.session_state.update({
            'fabrication_method': fabrication_method, 'treatment_method': treatment_method,
            'treatment_sequence': treatment_sequence, 'air_exposure_duration': air_exposure_duration
        })
    
    with col3:
        operator = st.text_input(
            "Operator (First Name Last Name)",
            value=st.session_state.operator,
            placeholder="e.g., Dong Nguyen"
        )
        
        operator_valid = bool(operator and len(operator.split()) >= 2)
        if operator and not operator_valid:
            st.error("Please enter both first name and last name for Operator")
        elif not operator:
            st.warning("Operator name is required")
        
        institution = st.text_input("Institution", value=st.session_state.institution)
        measurement_type = st.text_input("Measurement Type", value="Mapping XRF")
        xrf_fitting_method = st.selectbox("XRF Fitting Method", ["Series", "Bayesian"], index=0)
        
        st.session_state.update({
            'operator': operator, 'operator_valid': operator_valid,
            'institution': institution, 'measurement_type': measurement_type,
            'xrf_fitting_method': xrf_fitting_method
        })
    
    st.markdown("---")
    
    return {
        'substrate_number': substrate_number, 'substrate': substrate,
        'sample_description': sample_description, 'substrate_size': substrate_size,
        'fabrication_method': fabrication_method, 'treatment_method': treatment_method,
        'treatment_sequence': treatment_sequence, 'air_exposure_duration': air_exposure_duration,
        'operator': operator, 'operator_valid': operator_valid,
        'institution': institution, 'measurement_type': measurement_type,
        'xrf_fitting_method': xrf_fitting_method
    }

