import streamlit as st
from datetime import datetime
import re


from spx_processing_func import *


import zipfile
import tempfile
import shutil

def extract_spx_files_from_zip(zip_file):
    """Extract SPX files from uploaded ZIP file to temporary directory"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find all SPX files recursively
        temp_path = Path(temp_dir)
        spx_files = list(temp_path.rglob('*.spx'))
        
        return temp_dir, sorted(spx_files)
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e
    

    

st.set_page_config(layout="wide")

st.title("CSV File Generator for Sample Database")


# ==================== HELPER FUNCTIONS ====================

def validate_operator_name(operator):
    """Validate operator has first and last name"""
    operator_parts = operator.strip().split()
    return len(operator_parts) >= 2


def convert_time_to_12hour(time_str):
    """Convert 24-hour time format to 12-hour format with AM/PM"""
    time_parts = time_str.split(':')
    if len(time_parts) != 3:
        return None
    
    try:
        hour = int(time_parts[0])
        minute = int(time_parts[1])
        second = int(time_parts[2])
        
        period = "AM" if hour < 12 else "PM"
        hour_12 = hour if hour <= 12 else hour - 12
        hour_12 = 12 if hour_12 == 0 else hour_12
        
        return f"{hour_12}:{minute:02d}:{second:02d} {period}"
    except:
        return None


def format_date(date_obj):
    """Format date object to 'Weekday, Month Day, Year'"""
    return date_obj.strftime("%A, %B %d, %Y")


def generate_substrate_filename(substrate_number, institution, operator, substrate_type):
    """Generate substrate CSV filename"""
    operator_formatted = operator
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{substrate_number}_{institution}_{operator_formatted}_substrate_{substrate_type}_{current_datetime}.csv"


def generate_fabrication_filename(substrate_number, institution, operator, sequence, method):
    """Generate fabrication CSV filename"""
    operator_formatted = operator
    method_map = {
        'PVD-J': 'PVDJ',
        'Sputtering': 'Sputtering',
        'Tube Furnace': 'TubeFurnace',
        'RTP': 'RTP',
        'PLD': 'PLD',
        'PVD-P': 'PVDP'
    }
    method_formatted = method_map.get(method, method.upper())
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{substrate_number}_{institution}_{operator_formatted}_fab{sequence}_{method_formatted}_{current_datetime}.csv"


def parse_pvdp_csv(uploaded_file):
    """Parse PVD-P CSV file and extract metadata"""
    content = uploaded_file.getvalue().decode('utf-8')
    lines = content.split('\n')
    
    metadata = {
        'substrate_number': None,
        'process_id': None,
        'operator_code': None,
        'date': None,
        'time': None
    }
    
    for line in lines:
        if line.startswith('# Date:'):
            date_str = line.replace('# Date:', '').strip()
            metadata['date'] = date_str
        elif line.startswith('# Time:'):
            time_str = line.replace('# Time:', '').strip()
            metadata['time'] = time_str
        elif line.startswith('# Substrate Number:'):
            sample_num = line.replace('# Substrate Number:', '').strip()
            metadata['substrate_number'] = sample_num
        elif line.startswith('# process ID:'):
            process_id = line.replace('# process ID:', '').strip()
            metadata['process_id'] = process_id
        elif line.startswith('# operator:'):
            operator_code = line.replace('# operator:', '').strip()
            metadata['operator_code'] = operator_code
    
    return metadata, content


def replace_operator_in_csv(csv_content, old_operator, new_operator):
    """Replace operator code with actual operator name in CSV content"""
    return csv_content.replace(f"# operator: {old_operator}", f"# operator: {new_operator}")


def generate_pvdp_filename_from_metadata(substrate_number, institution, operator, sequence, date_str, time_str):
    """Generate PVD-P filename from extracted metadata"""
    operator_formatted = operator
    
    date_obj = datetime.strptime(date_str, "%Y/%m/%d")
    date_formatted = date_obj.strftime("%Y%m%d")
    
    time_formatted = time_str.replace(':', '')
    
    return f"{substrate_number}_{institution}_{operator_formatted}_fab{sequence}_PVDP_{date_formatted}_{time_formatted}.csv"


def generate_substrate_csv_content(data):
    """Generate substrate CSV content"""
    return f"""substrate_number,{data['substrate_number']}
sub_substrate_type,{data['substrate_type']}
sub_production_batch,{data['production_batch']}
sub_vendor,{data['vendor']}
sub_manufacture,{data['manufacture']}
sub_softing_point_celsius,{data['softing_point']}
sub_expansion_coefficient,{data['expansion_coefficient']}
sub_temp_celsius,{data['temp_celsius']}
sub_thickness_mm,{data['thickness']}
sub_size_mm_x_y,{data['size']}
sub_materials,{data['materials']}
sub_program,{data['program']}
sub_operator,{data['operator']}
sub_institution,{data['institution']}
sub_clean_method,{data['clean_method']}
sub_clean_description,{data['clean_description']}
sub_clean_duration_Min,{data['clean_duration']}
sub_clean_temperature_C,{data['clean_temperature']}
sub_clean_pressure_mbar,{data['clean_pressure']}
sub_clean_date,"{data['clean_date']}"
sub_clean_time,"{data['clean_time']}"
"""


def generate_pvdj_csv_content(common_data, specific_data):
    """Generate PVD-J CSV content"""
    csv_content = f"""substrate_number,{common_data['substrate_number']}
fab_method,{common_data['method']}
fab_sequence,{common_data['sequence']}
fab_process_number,{specific_data['process_number']}
fab_operator,{common_data['operator']}
fab_institution,{common_data['institution']}
fab_recipe_name,{specific_data['recipe_name']}
fab_box_type,{specific_data['box_type']}
fab_duration_minutes,{specific_data['duration_minutes']}
fab_substrate_temperature_celsius,{specific_data['substrate_temperature_celsius']}
fab_cooling_temperature_celsius,{specific_data['cooling_temperature_celsius']}
fab_holding_time_seconds,{specific_data['holding_time_seconds']}
fab_rate_nmol_per_cm2_per_sec,{specific_data['rate_nmol_per_cm2_per_sec']}
fab_power_W,{specific_data['power_W']}
fab_tooling_factor,{specific_data['tooling_factor']}
fab_xtal,{specific_data['xtal']}
fab_sample_orientation,{specific_data['sample_orientation']}
fab_sample_mass_before_mg,{specific_data['sample_mass_before_mg']}
fab_sample_mass_after_mg,{specific_data['sample_mass_after_mg']}
fab_date,"{common_data['date']}"
fab_time,"{common_data['time']}"
"""
    return csv_content


def generate_sputtering_csv_content(common_data, specific_data):
    """Generate Sputtering CSV content"""
    csv_content = f"""substrate_number,{common_data['substrate_number']}
fab_method,{common_data['method']}
fab_sequence,{common_data['sequence']}
fab_operator,{common_data['operator']}
fab_institution,{common_data['institution']}
fab_pre_fab_pressure,{specific_data['pre_fab_pressure']}
fab_program,{specific_data['program']}
fab_power_W,{specific_data['power_W']}
fab_current_A,{specific_data['current_A']}
fab_voltage_V,{specific_data['voltage_V']}
fab_gas_mix,{specific_data['gas_mix']}
fab_duration_minutes,{specific_data['duration_minutes']}
fab_process_pressure_mbar,{specific_data['process_pressure_mbar']}
fab_note,{specific_data['note']}
fab_date,"{common_data['date']}"
fab_time,"{common_data['time']}"
"""
    return csv_content


def generate_tubefurnace_csv_content(common_data, specific_data):
    """Generate Tube Furnace CSV content"""
    csv_content = f"""substrate_number,{common_data['substrate_number']}
fab_method,{common_data['method']}
fab_sequence,{common_data['sequence']}
fab_operator,{common_data['operator']}
fab_institution,{common_data['institution']}
fab_temperature_celsius,{specific_data['temperature_celsius']}
fab_rample_celsius_per_min,{specific_data['rample_celsius_per_min']}
fab_amount_selenium_g,{specific_data['amount_selenium_g']}
fab_amount_sulfur_g,{specific_data['amount_sulfur_g']}
fab_pressure_mbar,{specific_data['pressure_mbar']}
fab_sample_orientation_in_box,{specific_data['sample_orientation_in_box']}
fab_humidity_percent,{specific_data['humidity_percent']}
fab_storage_days,{specific_data['storage_days']}
fab_duration_minutes,{specific_data['duration_minutes']}
fab_cooling_time_minutes,{specific_data['cooling_time_minutes']}
fab_sample_weight_before_mg,{specific_data['sample_weight_before_mg']}
fab_sample_weight_after_mg,{specific_data['sample_weight_after_mg']}
fab_position_in_oven,{specific_data['position_in_oven']}
fab_date,"{common_data['date']}"
fab_time,"{common_data['time']}"
"""
    return csv_content


def generate_rtp_csv_content(common_data, specific_data):
    """Generate RTP CSV content"""
    csv_content = f"""substrate_number,{common_data['substrate_number']}
fab_method,{common_data['method']}
fab_sequence,{common_data['sequence']}
fab_operator,{common_data['operator']}
fab_institution,{common_data['institution']}
fab_pressure_mbar,{specific_data['pressure_mbar']}
fab_box_type,{specific_data['box_type']}
fab_amount_selenium_g,{specific_data['amount_selenium_g']}
fab_amount_sulfur_g,{specific_data['amount_sulfur_g']}
fab_steps,{specific_data['steps']}
fab_recipe,{specific_data['recipe']}
fab_rampe_K_per_second,{specific_data['rampe_K_per_second']}
fab_holding_time_minutes,{specific_data['holding_time_minutes']}
fab_sample_weight_before_mg,{specific_data['sample_weight_before_mg']}
fab_sample_weight_after_mg,{specific_data['sample_weight_after_mg']}
fab_orientation,{specific_data['orientation']}
fab_date,"{common_data['date']}"
fab_time,"{common_data['time']}"
"""
    return csv_content


def generate_pld_csv_content(common_data, specific_data):
    """Generate PLD CSV content"""
    csv_content = f"""substrate_number,{common_data['substrate_number']}
fab_method,{common_data['method']}
fab_sequence,{common_data['sequence']}
fab_operator,{common_data['operator']}
fab_institution,{common_data['institution']}
pre_ablation,Pre-Ablation Parameters
shots,{specific_data['pre_shots']}
laser_frequency_hz,{specific_data['pre_laser_frequency_hz']}
laser_fluence,{specific_data['pre_laser_fluence']}
gas_pressure_mbar,{specific_data['pre_gas_pressure_mbar']}
gas_type,{specific_data['pre_gas_type']}
duration_minutes,{specific_data['pre_duration_minutes']}
fab_deposition,Deposition Parameters
fab_temperature_C,{specific_data['temperature_C']}
fab_shots,{specific_data['shots']}
fab_laser_frequency_hz,{specific_data['laser_frequency_hz']}
fab_laser_fluence,{specific_data['laser_fluence']}
fab_gas_pressure_mbar,{specific_data['gas_pressure_mbar']}
fab_gas_type,{specific_data['gas_type']}
fab_duration_minutes,{specific_data['duration_minutes']}
fab_date,"{common_data['date']}"
fab_time,"{common_data['time']}"
"""
    return csv_content


# ==================== TREATMENT ====================

def generate_treatment_filename(substrate_number, institution, operator, sequence, method):
    """Generate treatment CSV filename"""
    operator_formatted = operator
    method_map = {
        'Annealing': 'Annealing',
        'As-deposited': 'As-deposited',
        'Storing-in-Glovebox': 'Storing-in-Glovebox',
        'Storing-out-Glovebox': 'Storing-out-Glovebox'
    }
    method_formatted = method_map.get(method, method)
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{substrate_number}_{institution}_{operator_formatted}_treat{sequence}_{method_formatted}_{current_datetime}.csv"


def generate_treatment_csv_content(common_data, specific_data):
    """Generate Treatment CSV content (works for both Annealing and As-deposited)"""
    csv_content = f"""substrate_number,{common_data['substrate_number']}
treat_method,{common_data['method']}
treat_sequence,{common_data['sequence']}
treat_operator,{common_data['operator']}
treat_institution,{common_data['institution']}
treat_place,{specific_data['place']}
treat_temperature_celsius,{specific_data['temperature_celsius']}
treat_duration_second,{specific_data['duration_second']}
treat_humidity_ppm,{specific_data['humidity_ppm']}
treat_oxygen_concentration_ppm,{specific_data['oxygen_concentration_ppm']}
treat_gas,{specific_data['gas']}
treat_pressure_mbar,{specific_data['pressure_mbar']}
treat_date,"{common_data['date']}"
treat_time,"{common_data['time']}"
"""
    return csv_content


# ==================== TAB 1: SUBSTRATE GENERATION ====================

tab1, tab2, tab3, tab4 = st.tabs(["Substrate Generation", "Fabrication Generation", "Treatment Generation", "XRF&SPX Generation"])


with tab1:
    st.header("Substrate CSV File Generator")
    
    # Set default values
    st.session_state.sub_substrate_number = st.session_state.get('sub_substrate_number', "3716-15")
    st.session_state.sub_institution = st.session_state.get('sub_institution', "HZB")
    st.session_state.sub_operator = st.session_state.get('sub_operator', "Steinkopf Lars")
    st.session_state.sub_substrate_type = st.session_state.get('sub_substrate_type', "quartz")
    st.session_state.sub_thickness = st.session_state.get('sub_thickness', "1.1")
    st.session_state.sub_size = st.session_state.get('sub_size', "50x50")
    st.session_state.sub_materials = st.session_state.get('sub_materials', "SiO2")
    st.session_state.sub_production_batch = st.session_state.get('sub_production_batch', "B123")
    st.session_state.sub_vendor = st.session_state.get('sub_vendor', "Vendor Name")
    st.session_state.sub_manufacture = st.session_state.get('sub_manufacture', "Manufacturer Name")
    st.session_state.sub_softing_point = st.session_state.get('sub_softing_point', "821")
    st.session_state.sub_expansion_coefficient = st.session_state.get('sub_expansion_coefficient', "0.55")
    st.session_state.sub_temp_celsius = st.session_state.get('sub_temp_celsius', "20-300")
    st.session_state.sub_program = st.session_state.get('sub_program', "Standard")
    st.session_state.sub_clean_method = st.session_state.get('sub_clean_method', "Ultrasonic")
    st.session_state.sub_clean_description = st.session_state.get('sub_clean_description', "Cleaning process description")
    st.session_state.sub_clean_duration = st.session_state.get('sub_clean_duration', "15")
    st.session_state.sub_clean_temperature = st.session_state.get('sub_clean_temperature', "50")
    st.session_state.sub_clean_pressure = st.session_state.get('sub_clean_pressure', "1013")
    st.session_state.sub_clean_date = st.session_state.get('sub_clean_date', datetime.now().date())
    st.session_state.sub_clean_time = st.session_state.get('sub_clean_time', "14:01:15")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Sample Information")
        st.session_state.sub_substrate_number = st.text_input("Substrate Number", value=st.session_state.sub_substrate_number, key="sub_sn")
        st.session_state.sub_institution = st.text_input("Institution", value=st.session_state.sub_institution, key="sub_inst")
        st.session_state.sub_operator = st.text_input("Operator (First and Last Name)", value=st.session_state.sub_operator, help="Must include both first and last name", key="sub_op")
        st.session_state.sub_substrate_type = st.text_input("Substrate Type", value=st.session_state.sub_substrate_type, key="sub_type")
        
        st.subheader("Substrate Properties")
        st.session_state.sub_thickness = st.text_input("Thickness (mm)", value=st.session_state.sub_thickness, key="sub_thick")
        st.session_state.sub_size = st.text_input("Size mm (x, y)", value=st.session_state.sub_size, key="sub_size_val")
        st.session_state.sub_materials = st.text_input("Materials", value=st.session_state.sub_materials, key="sub_mat")
    
    with col2:
        st.subheader("Substrate Type Details")
        st.session_state.sub_production_batch = st.text_input("Production Batch", value=st.session_state.sub_production_batch, key="sub_pb")
        st.session_state.sub_vendor = st.text_input("Vendor", value=st.session_state.sub_vendor, key="sub_vend")
        st.session_state.sub_manufacture = st.text_input("Manufacture", value=st.session_state.sub_manufacture, key="sub_manuf")
        st.session_state.sub_softing_point = st.text_input("Softing Point (Celsius)", value=st.session_state.sub_softing_point, key="sub_soft")
        st.session_state.sub_expansion_coefficient = st.text_input("Expansion Coefficient", value=st.session_state.sub_expansion_coefficient, key="sub_exp")
        st.session_state.sub_temp_celsius = st.text_input("Temperature (Celsius)", value=st.session_state.sub_temp_celsius, key="sub_temp")
        st.session_state.sub_program = st.text_input("Program", value=st.session_state.sub_program, key="sub_prog")
    
    with col3:
        st.subheader("Cleaning Information")
        st.session_state.sub_clean_method = st.text_input("Clean Method", value=st.session_state.sub_clean_method, key="sub_cm")
        st.session_state.sub_clean_description = st.text_area("Clean Description", value=st.session_state.sub_clean_description, key="sub_cd")
        st.session_state.sub_clean_duration = st.text_input("Clean Duration (Min)", value=st.session_state.sub_clean_duration, key="sub_cdur")
        st.session_state.sub_clean_temperature = st.text_input("Clean Temperature (C)", value=st.session_state.sub_clean_temperature, key="sub_ctemp")
        st.session_state.sub_clean_pressure = st.text_input("Clean Pressure (mbar)", value=st.session_state.sub_clean_pressure, key="sub_cp")
        
        st.subheader("Date and Time")
        st.session_state.sub_clean_date = st.date_input("Clean Date", value=st.session_state.sub_clean_date, key="sub_cdate")
        st.session_state.sub_clean_time = st.text_input("Clean Time", value=st.session_state.sub_clean_time, help="Format: HH:MM:SS (24-hour)", key="sub_ctime")
    
    st.divider()
    
    if st.button("Generate Substrate CSV File", type="primary"):
        if not st.session_state.sub_substrate_number or not st.session_state.sub_institution or not st.session_state.sub_operator or not st.session_state.sub_substrate_type:
            st.error("Please fill in all required fields: Substrate Number, Institution, Operator, and Substrate Type")
        elif not validate_operator_name(st.session_state.sub_operator):
            st.error("Operator must include both First Name and Last Name (e.g., Steinkopf Lars)")
        else:
            date_formatted = format_date(st.session_state.sub_clean_date)
            time_formatted = convert_time_to_12hour(st.session_state.sub_clean_time)
            
            if not time_formatted:
                st.error("Invalid time format. Please use HH:MM:SS format")
            else:
                substrate_data = {
                    'substrate_number': st.session_state.sub_substrate_number,
                    'substrate_type': st.session_state.sub_substrate_type,
                    'production_batch': st.session_state.sub_production_batch,
                    'vendor': st.session_state.sub_vendor,
                    'manufacture': st.session_state.sub_manufacture,
                    'softing_point': st.session_state.sub_softing_point,
                    'expansion_coefficient': st.session_state.sub_expansion_coefficient,
                    'temp_celsius': st.session_state.sub_temp_celsius,
                    'thickness': st.session_state.sub_thickness,
                    'size': st.session_state.sub_size,
                    'materials': st.session_state.sub_materials,
                    'program': st.session_state.sub_program,
                    'operator': st.session_state.sub_operator,
                    'institution': st.session_state.sub_institution,
                    'clean_method': st.session_state.sub_clean_method,
                    'clean_description': st.session_state.sub_clean_description,
                    'clean_duration': st.session_state.sub_clean_duration,
                    'clean_temperature': st.session_state.sub_clean_temperature,
                    'clean_pressure': st.session_state.sub_clean_pressure,
                    'clean_date': date_formatted,
                    'clean_time': time_formatted
                }
                
                csv_content = generate_substrate_csv_content(substrate_data)
                filename = generate_substrate_filename(
                    st.session_state.sub_substrate_number,
                    st.session_state.sub_institution,
                    st.session_state.sub_operator,
                    st.session_state.sub_substrate_type
                )
                
                st.success("CSV file generated successfully")
                
                st.download_button(
                    label="Download CSV File",
                    data=csv_content,
                    file_name=filename,
                    mime="text/csv"
                )
                
                with st.expander("Preview CSV Content"):
                    st.text(csv_content)


# ==================== TAB 2: FABRICATION GENERATION ====================

with tab2:
    st.header("Fabrication CSV File Generator")
    
    fab_method = st.selectbox(
        "Select Fabrication Method",
        ["PVD-J", "Sputtering", "Tube Furnace", "RTP", "PLD", "PVD-P"]
    )
    
    # Set common default values
    st.session_state.fab_substrate_number = st.session_state.get('fab_substrate_number', "3716-15")
    st.session_state.fab_institution = st.session_state.get('fab_institution', "HZB")
    st.session_state.fab_operator = st.session_state.get('fab_operator', "Steinkopf Lars")
    st.session_state.fab_sequence = st.session_state.get('fab_sequence', "1")
    st.session_state.fab_date = st.session_state.get('fab_date', datetime.now().date())
    st.session_state.fab_time = st.session_state.get('fab_time', "14:01:15")
    
    if fab_method != "PVD-P":
        st.subheader("Common Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state.fab_substrate_number = st.text_input("Substrate Number", value=st.session_state.fab_substrate_number, key="fab_sn")
            st.session_state.fab_institution = st.text_input("Institution", value=st.session_state.fab_institution, key="fab_inst")
        
        with col2:
            st.session_state.fab_operator = st.text_input("Operator (First and Last Name)", value=st.session_state.fab_operator, key="fab_op")
            st.session_state.fab_sequence = st.text_input("Fabrication Sequence", value=st.session_state.fab_sequence, key="fab_seq")
        
        with col3:
            st.session_state.fab_date = st.date_input("Fabrication Date", value=st.session_state.fab_date, key="fab_date_input")
            st.session_state.fab_time = st.text_input("Fabrication Time", value=st.session_state.fab_time, help="Format: HH:MM:SS (24-hour)", key="fab_time_input")
        
        st.divider()
    
    # Method-specific parameters with default values
    if fab_method == "PVD-P":
        st.subheader("PVD-P CSV Upload")
      
        uploaded_file = st.file_uploader("Upload PVD-P CSV File", type=['csv'], key="pvdp_upload")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pvdp_institution = st.text_input("Institution", value="HZB", key="pvdp_inst")
        
        with col2:
            pvdp_operator = st.text_input("Operator (First and Last Name)", value="Henry Gos", key="pvdp_op")
        
        pvdp_sequence = st.text_input("Fabrication Sequence", value="1", key="pvdp_seq")
        
        if uploaded_file is not None:
            metadata, csv_content = parse_pvdp_csv(uploaded_file)
            
            st.success("File uploaded successfully!")
            
            with st.expander("Extracted Metadata from CSV"):
                st.write(f"Substrate Number: {metadata['substrate_number']}")
                st.write(f"Process ID: {metadata['process_id']}")
                st.write(f"Operator Code: {metadata['operator_code']}")
                st.write(f"Date: {metadata['date']}")
                st.write(f"Time: {metadata['time']}")
            
            st.divider()
            
            if st.button("Generate PVD-P CSV File", type="primary"):
                if not pvdp_operator or not pvdp_institution or not pvdp_sequence:
                    st.error("Please fill in Institution, Operator, and Sequence")
                elif not validate_operator_name(pvdp_operator):
                    st.error("Operator must include both First Name and Last Name")
                else:
                    updated_csv = replace_operator_in_csv(csv_content, metadata['operator_code'], pvdp_operator)
                    
                    filename = generate_pvdp_filename_from_metadata(
                        metadata['substrate_number'],
                        pvdp_institution,
                        pvdp_operator,
                        pvdp_sequence,
                        metadata['date'],
                        metadata['time']
                    )
                    
                    st.success("CSV file generated successfully!")
                    st.info(f"Filename: {filename}")
                    
                    st.download_button(
                        label="Download PVD-P CSV File",
                        data=updated_csv,
                        file_name=filename,
                        mime="text/csv"
                    )
    
    elif fab_method == "PVD-J":
        st.subheader("PVD-J Parameters")
        
        st.session_state.fab_process_number = st.session_state.get('fab_process_number', "P001")
        st.session_state.fab_recipe_name = st.session_state.get('fab_recipe_name', "Recipe1")
        st.session_state.fab_box_type = st.session_state.get('fab_box_type', "Standard Box")
        st.session_state.fab_duration_minutes = st.session_state.get('fab_duration_minutes', "30")
        st.session_state.fab_substrate_temperature_celsius = st.session_state.get('fab_substrate_temperature_celsius', "150")
        st.session_state.fab_cooling_temperature_celsius = st.session_state.get('fab_cooling_temperature_celsius', "25")
        st.session_state.fab_holding_time_seconds = st.session_state.get('fab_holding_time_seconds', "60")
        st.session_state.fab_rate_nmol_per_cm2_per_sec = st.session_state.get('fab_rate_nmol_per_cm2_per_sec', "0.5")
        st.session_state.fab_power_W = st.session_state.get('fab_power_W', "100")
        st.session_state.fab_tooling_factor = st.session_state.get('fab_tooling_factor', "1.0")
        st.session_state.fab_xtal = st.session_state.get('fab_xtal', "Xtal1")
        st.session_state.fab_sample_orientation = st.session_state.get('fab_sample_orientation', "Face up")
        st.session_state.fab_sample_mass_before_mg = st.session_state.get('fab_sample_mass_before_mg', "100.0")
        st.session_state.fab_sample_mass_after_mg = st.session_state.get('fab_sample_mass_after_mg', "101.0")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state.fab_process_number = st.text_input("Process Number", value=st.session_state.fab_process_number, key="pvdj_pn")
            st.session_state.fab_recipe_name = st.text_input("Recipe Name", value=st.session_state.fab_recipe_name, key="pvdj_rn")
            st.session_state.fab_box_type = st.text_input("Box Type", value=st.session_state.fab_box_type, key="pvdj_bt")
            st.session_state.fab_duration_minutes = st.text_input("Duration (minutes)", value=st.session_state.fab_duration_minutes, key="pvdj_dur")
        
        with col2:
            st.session_state.fab_substrate_temperature_celsius = st.text_input("Substrate Temperature (C)", value=st.session_state.fab_substrate_temperature_celsius, key="pvdj_st")
            st.session_state.fab_cooling_temperature_celsius = st.text_input("Cooling Temperature (C)", value=st.session_state.fab_cooling_temperature_celsius, key="pvdj_ct")
            st.session_state.fab_holding_time_seconds = st.text_input("Holding Time (seconds)", value=st.session_state.fab_holding_time_seconds, key="pvdj_ht")
            st.session_state.fab_rate_nmol_per_cm2_per_sec = st.text_input("Rate (nmol/cm2/sec)", value=st.session_state.fab_rate_nmol_per_cm2_per_sec, key="pvdj_rate")
        
        with col3:
            st.session_state.fab_power_W = st.text_input("Power (W)", value=st.session_state.fab_power_W, key="pvdj_pow")
            st.session_state.fab_tooling_factor = st.text_input("Tooling Factor", value=st.session_state.fab_tooling_factor, key="pvdj_tf")
            st.session_state.fab_xtal = st.text_input("Xtal", value=st.session_state.fab_xtal, key="pvdj_xtal")
            st.session_state.fab_sample_orientation = st.text_input("Sample Orientation", value=st.session_state.fab_sample_orientation, key="pvdj_so")
        
        col4, col5 = st.columns(2)
        with col4:
            st.session_state.fab_sample_mass_before_mg = st.text_input("Sample Mass Before (mg)", value=st.session_state.fab_sample_mass_before_mg, key="pvdj_mb")
        with col5:
            st.session_state.fab_sample_mass_after_mg = st.text_input("Sample Mass After (mg)", value=st.session_state.fab_sample_mass_after_mg, key="pvdj_ma")
    
    elif fab_method == "Sputtering":
        st.subheader("Sputtering Parameters")
        
        st.session_state.fab_program = st.session_state.get('fab_program', "Program1")
        st.session_state.fab_duration_minutes = st.session_state.get('fab_duration_minutes', "20")
        st.session_state.fab_power_W = st.session_state.get('fab_power_W', "200")
        st.session_state.fab_current_A = st.session_state.get('fab_current_A', "0.5")
        st.session_state.fab_voltage_V = st.session_state.get('fab_voltage_V', "400")
        st.session_state.fab_gas_mix = st.session_state.get('fab_gas_mix', "Ar 95% O2 5%")
        st.session_state.fab_process_pressure_mbar = st.session_state.get('fab_process_pressure_mbar', "0.005")
        st.session_state.fab_pre_fab_pressure = st.session_state.get('fab_pre_fab_pressure', "1e-6")
        st.session_state.fab_note = st.session_state.get('fab_note', "Notes here")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state.fab_program = st.text_input("Program", value=st.session_state.fab_program, key="spt_prog")
            st.session_state.fab_duration_minutes = st.text_input("Duration (minutes)", value=st.session_state.fab_duration_minutes, key="spt_dur")
            st.session_state.fab_power_W = st.text_input("Power (W)", value=st.session_state.fab_power_W, key="spt_pow")
        
        with col2:
            st.session_state.fab_current_A = st.text_input("Current (A)", value=st.session_state.fab_current_A, key="spt_cur")
            st.session_state.fab_voltage_V = st.text_input("Voltage (V)", value=st.session_state.fab_voltage_V, key="spt_volt")
            st.session_state.fab_gas_mix = st.text_input("Gas Mix", value=st.session_state.fab_gas_mix, key="spt_gas")
        
        with col3:
            st.session_state.fab_process_pressure_mbar = st.text_input("Process Pressure (mbar)", value=st.session_state.fab_process_pressure_mbar, key="spt_pp")
            st.session_state.fab_pre_fab_pressure = st.text_input("Pre-Fab Pressure", value=st.session_state.fab_pre_fab_pressure, key="spt_pfp")
            st.session_state.fab_note = st.text_area("Note", value=st.session_state.fab_note, key="spt_note")
    
    elif fab_method == "Tube Furnace":
        st.subheader("Tube Furnace Parameters")
        
        st.session_state.fab_temperature_celsius = st.session_state.get('fab_temperature_celsius', "550")
        st.session_state.fab_rample_celsius_per_min = st.session_state.get('fab_rample_celsius_per_min', "10")
        st.session_state.fab_amount_selenium_g = st.session_state.get('fab_amount_selenium_g', "0.5")
        st.session_state.fab_amount_sulfur_g = st.session_state.get('fab_amount_sulfur_g', "0.2")
        st.session_state.fab_pressure_mbar = st.session_state.get('fab_pressure_mbar', "1013")
        st.session_state.fab_humidity_percent = st.session_state.get('fab_humidity_percent', "50")
        st.session_state.fab_duration_minutes = st.session_state.get('fab_duration_minutes', "60")
        st.session_state.fab_cooling_time_minutes = st.session_state.get('fab_cooling_time_minutes', "30")
        st.session_state.fab_storage_days = st.session_state.get('fab_storage_days', "1")
        st.session_state.fab_sample_orientation_in_box = st.session_state.get('fab_sample_orientation_in_box', "Face up")
        st.session_state.fab_position_in_oven = st.session_state.get('fab_position_in_oven', "Center")
        st.session_state.fab_sample_weight_before_mg = st.session_state.get('fab_sample_weight_before_mg', "100.0")
        st.session_state.fab_sample_weight_after_mg = st.session_state.get('fab_sample_weight_after_mg', "101.0")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state.fab_temperature_celsius = st.text_input("Temperature (C)", value=st.session_state.fab_temperature_celsius, key="tf_temp")
            st.session_state.fab_rample_celsius_per_min = st.text_input("Ramp Rate (C/min)", value=st.session_state.fab_rample_celsius_per_min, key="tf_ramp")
            st.session_state.fab_amount_selenium_g = st.text_input("Amount Selenium (g)", value=st.session_state.fab_amount_selenium_g, key="tf_se")
            st.session_state.fab_amount_sulfur_g = st.text_input("Amount Sulfur (g)", value=st.session_state.fab_amount_sulfur_g, key="tf_s")
        
        with col2:
            st.session_state.fab_pressure_mbar = st.text_input("Pressure (mbar)", value=st.session_state.fab_pressure_mbar, key="tf_press")
            st.session_state.fab_humidity_percent = st.text_input("Humidity (%)", value=st.session_state.fab_humidity_percent, key="tf_hum")
            st.session_state.fab_duration_minutes = st.text_input("Duration (minutes)", value=st.session_state.fab_duration_minutes, key="tf_dur")
            st.session_state.fab_cooling_time_minutes = st.text_input("Cooling Time (minutes)", value=st.session_state.fab_cooling_time_minutes, key="tf_cool")
        
        with col3:
            st.session_state.fab_storage_days = st.text_input("Storage Days", value=st.session_state.fab_storage_days, key="tf_stor")
            st.session_state.fab_sample_orientation_in_box = st.text_input("Sample Orientation in Box", value=st.session_state.fab_sample_orientation_in_box, key="tf_orient")
            st.session_state.fab_position_in_oven = st.text_input("Position in Oven", value=st.session_state.fab_position_in_oven, key="tf_pos")
        
        col4, col5 = st.columns(2)
        with col4:
            st.session_state.fab_sample_weight_before_mg = st.text_input("Sample Weight Before (mg)", value=st.session_state.fab_sample_weight_before_mg, key="tf_wb")
        with col5:
            st.session_state.fab_sample_weight_after_mg = st.text_input("Sample Weight After (mg)", value=st.session_state.fab_sample_weight_after_mg, key="tf_wa")
    
    elif fab_method == "RTP":
        st.subheader("RTP Parameters")
        
        st.session_state.fab_pressure_mbar = st.session_state.get('fab_pressure_mbar', "1013")
        st.session_state.fab_box_type = st.session_state.get('fab_box_type', "Standard Box")
        st.session_state.fab_amount_selenium_g = st.session_state.get('fab_amount_selenium_g', "0.5")
        st.session_state.fab_amount_sulfur_g = st.session_state.get('fab_amount_sulfur_g', "0.2")
        st.session_state.fab_steps = st.session_state.get('fab_steps', "3")
        st.session_state.fab_recipe = st.session_state.get('fab_recipe', "Recipe1")
        st.session_state.fab_rampe_K_per_second = st.session_state.get('fab_rampe_K_per_second', "5")
        st.session_state.fab_holding_time_minutes = st.session_state.get('fab_holding_time_minutes', "10")
        st.session_state.fab_sample_weight_before_mg = st.session_state.get('fab_sample_weight_before_mg', "100.0")
        st.session_state.fab_sample_weight_after_mg = st.session_state.get('fab_sample_weight_after_mg', "101.0")
        st.session_state.fab_orientation = st.session_state.get('fab_orientation', "Face up")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state.fab_pressure_mbar = st.text_input("Pressure (mbar)", value=st.session_state.fab_pressure_mbar, key="rtp_press")
            st.session_state.fab_box_type = st.text_input("Box Type", value=st.session_state.fab_box_type, key="rtp_box")
            st.session_state.fab_amount_selenium_g = st.text_input("Amount Selenium (g)", value=st.session_state.fab_amount_selenium_g, key="rtp_se")
            st.session_state.fab_amount_sulfur_g = st.text_input("Amount Sulfur (g)", value=st.session_state.fab_amount_sulfur_g, key="rtp_s")
        
        with col2:
            st.session_state.fab_steps = st.text_input("Steps", value=st.session_state.fab_steps, key="rtp_steps")
            st.session_state.fab_recipe = st.text_input("Recipe", value=st.session_state.fab_recipe, key="rtp_recipe")
            st.session_state.fab_rampe_K_per_second = st.text_input("Ramp Rate (K/s)", value=st.session_state.fab_rampe_K_per_second, key="rtp_ramp")
            st.session_state.fab_holding_time_minutes = st.text_input("Holding Time (minutes)", value=st.session_state.fab_holding_time_minutes, key="rtp_hold")
        
        with col3:
            st.session_state.fab_orientation = st.text_input("Orientation", value=st.session_state.fab_orientation, key="rtp_orient")
        
        col4, col5 = st.columns(2)
        with col4:
            st.session_state.fab_sample_weight_before_mg = st.text_input("Sample Weight Before (mg)", value=st.session_state.fab_sample_weight_before_mg, key="rtp_wb")
        with col5:
            st.session_state.fab_sample_weight_after_mg = st.text_input("Sample Weight After (mg)", value=st.session_state.fab_sample_weight_after_mg, key="rtp_wa")
    
    elif fab_method == "PLD":
        st.subheader("PLD Parameters")
        
        st.markdown("**Pre-Ablation Parameters**")
        st.session_state.pre_shots = st.session_state.get('pre_shots', "100")
        st.session_state.pre_laser_frequency_hz = st.session_state.get('pre_laser_frequency_hz', "10")
        st.session_state.pre_laser_fluence = st.session_state.get('pre_laser_fluence', "2.0")
        st.session_state.pre_gas_pressure_mbar = st.session_state.get('pre_gas_pressure_mbar', "0.1")
        st.session_state.pre_gas_type = st.session_state.get('pre_gas_type', "Oxygen")
        st.session_state.pre_duration_minutes = st.session_state.get('pre_duration_minutes', "5")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state.pre_shots = st.text_input("Pre-Ablation Shots", value=st.session_state.pre_shots, key="pld_pre_shots")
            st.session_state.pre_laser_frequency_hz = st.text_input("Pre-Ablation Laser Frequency (Hz)", value=st.session_state.pre_laser_frequency_hz, key="pld_pre_freq")
        
        with col2:
            st.session_state.pre_laser_fluence = st.text_input("Pre-Ablation Laser Fluence", value=st.session_state.pre_laser_fluence, key="pld_pre_flu")
            st.session_state.pre_gas_pressure_mbar = st.text_input("Pre-Ablation Gas Pressure (mbar)", value=st.session_state.pre_gas_pressure_mbar, key="pld_pre_press")
        
        with col3:
            st.session_state.pre_gas_type = st.text_input("Pre-Ablation Gas Type", value=st.session_state.pre_gas_type, key="pld_pre_gas")
            st.session_state.pre_duration_minutes = st.text_input("Pre-Ablation Duration (minutes)", value=st.session_state.pre_duration_minutes, key="pld_pre_dur")
        
        st.markdown("**Deposition Parameters**")
        st.session_state.fab_temperature_C = st.session_state.get('fab_temperature_C', "500")
        st.session_state.fab_shots = st.session_state.get('fab_shots', "1000")
        st.session_state.fab_laser_frequency_hz = st.session_state.get('fab_laser_frequency_hz', "10")
        st.session_state.fab_laser_fluence = st.session_state.get('fab_laser_fluence', "2.0")
        st.session_state.fab_gas_pressure_mbar = st.session_state.get('fab_gas_pressure_mbar', "0.2")
        st.session_state.fab_gas_type = st.session_state.get('fab_gas_type', "Oxygen")
        st.session_state.fab_duration_minutes = st.session_state.get('fab_duration_minutes', "30")
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.session_state.fab_temperature_C = st.text_input("Temperature (C)", value=st.session_state.fab_temperature_C, key="pld_temp")
            st.session_state.fab_shots = st.text_input("Shots", value=st.session_state.fab_shots, key="pld_shots")
        
        with col5:
            st.session_state.fab_laser_frequency_hz = st.text_input("Laser Frequency (Hz)", value=st.session_state.fab_laser_frequency_hz, key="pld_freq")
            st.session_state.fab_laser_fluence = st.text_input("Laser Fluence", value=st.session_state.fab_laser_fluence, key="pld_flu")
        
        with col6:
            st.session_state.fab_gas_pressure_mbar = st.text_input("Gas Pressure (mbar)", value=st.session_state.fab_gas_pressure_mbar, key="pld_press")
            st.session_state.fab_gas_type = st.text_input("Gas Type", value=st.session_state.fab_gas_type, key="pld_gas")
            st.session_state.fab_duration_minutes = st.text_input("Duration (minutes)", value=st.session_state.fab_duration_minutes, key="pld_dur")
    
    if fab_method != "PVD-P":
        st.divider()
        
        if st.button("Generate Fabrication CSV File", type="primary"):
            if not st.session_state.fab_substrate_number or not st.session_state.fab_institution or not st.session_state.fab_operator or not st.session_state.fab_sequence:
                st.error("Please fill in all required fields: Substrate Number, Institution, Operator, and Sequence")
            elif not validate_operator_name(st.session_state.fab_operator):
                st.error("Operator must include both First Name and Last Name")
            else:
                date_formatted = format_date(st.session_state.fab_date)
                time_formatted = convert_time_to_12hour(st.session_state.fab_time)
                
                if not time_formatted:
                    st.error("Invalid time format. Please use HH:MM:SS format")
                else:
                    common_data = {
                        'substrate_number': st.session_state.fab_substrate_number,
                        'institution': st.session_state.fab_institution,
                        'operator': st.session_state.fab_operator,
                        'method': fab_method,
                        'sequence': st.session_state.fab_sequence,
                        'date': date_formatted,
                        'time': time_formatted
                    }
                    
                    csv_content = ""
                    
                    if fab_method == "PVD-J":
                        specific_data = {
                            'process_number': st.session_state.fab_process_number,
                            'recipe_name': st.session_state.fab_recipe_name,
                            'box_type': st.session_state.fab_box_type,
                            'duration_minutes': st.session_state.fab_duration_minutes,
                            'substrate_temperature_celsius': st.session_state.fab_substrate_temperature_celsius,
                            'cooling_temperature_celsius': st.session_state.fab_cooling_temperature_celsius,
                            'holding_time_seconds': st.session_state.fab_holding_time_seconds,
                            'rate_nmol_per_cm2_per_sec': st.session_state.fab_rate_nmol_per_cm2_per_sec,
                            'power_W': st.session_state.fab_power_W,
                            'tooling_factor': st.session_state.fab_tooling_factor,
                            'xtal': st.session_state.fab_xtal,
                            'sample_orientation': st.session_state.fab_sample_orientation,
                            'sample_mass_before_mg': st.session_state.fab_sample_mass_before_mg,
                            'sample_mass_after_mg': st.session_state.fab_sample_mass_after_mg
                        }
                        csv_content = generate_pvdj_csv_content(common_data, specific_data)
                    
                    elif fab_method == "Sputtering":
                        specific_data = {
                            'program': st.session_state.fab_program,
                            'duration_minutes': st.session_state.fab_duration_minutes,
                            'power_W': st.session_state.fab_power_W,
                            'current_A': st.session_state.fab_current_A,
                            'voltage_V': st.session_state.fab_voltage_V,
                            'gas_mix': st.session_state.fab_gas_mix,
                            'process_pressure_mbar': st.session_state.fab_process_pressure_mbar,
                            'pre_fab_pressure': st.session_state.fab_pre_fab_pressure,
                            'note': st.session_state.fab_note
                        }
                        csv_content = generate_sputtering_csv_content(common_data, specific_data)
                    
                    elif fab_method == "Tube Furnace":
                        specific_data = {
                            'temperature_celsius': st.session_state.fab_temperature_celsius,
                            'rample_celsius_per_min': st.session_state.fab_rample_celsius_per_min,
                            'amount_selenium_g': st.session_state.fab_amount_selenium_g,
                            'amount_sulfur_g': st.session_state.fab_amount_sulfur_g,
                            'pressure_mbar': st.session_state.fab_pressure_mbar,
                            'humidity_percent': st.session_state.fab_humidity_percent,
                            'duration_minutes': st.session_state.fab_duration_minutes,
                            'cooling_time_minutes': st.session_state.fab_cooling_time_minutes,
                            'storage_days': st.session_state.fab_storage_days,
                            'sample_orientation_in_box': st.session_state.fab_sample_orientation_in_box,
                            'position_in_oven': st.session_state.fab_position_in_oven,
                            'sample_weight_before_mg': st.session_state.fab_sample_weight_before_mg,
                            'sample_weight_after_mg': st.session_state.fab_sample_weight_after_mg
                        }
                        csv_content = generate_tubefurnace_csv_content(common_data, specific_data)
                    
                    elif fab_method == "RTP":
                        specific_data = {
                            'pressure_mbar': st.session_state.fab_pressure_mbar,
                            'box_type': st.session_state.fab_box_type,
                            'amount_selenium_g': st.session_state.fab_amount_selenium_g,
                            'amount_sulfur_g': st.session_state.fab_amount_sulfur_g,
                            'steps': st.session_state.fab_steps,
                            'recipe': st.session_state.fab_recipe,
                            'rampe_K_per_second': st.session_state.fab_rampe_K_per_second,
                            'holding_time_minutes': st.session_state.fab_holding_time_minutes,
                            'sample_weight_before_mg': st.session_state.fab_sample_weight_before_mg,
                            'sample_weight_after_mg': st.session_state.fab_sample_weight_after_mg,
                            'orientation': st.session_state.fab_orientation
                        }
                        csv_content = generate_rtp_csv_content(common_data, specific_data)
                    
                    elif fab_method == "PLD":
                        specific_data = {
                            'pre_shots': st.session_state.pre_shots,
                            'pre_laser_frequency_hz': st.session_state.pre_laser_frequency_hz,
                            'pre_laser_fluence': st.session_state.pre_laser_fluence,
                            'pre_gas_pressure_mbar': st.session_state.pre_gas_pressure_mbar,
                            'pre_gas_type': st.session_state.pre_gas_type,
                            'pre_duration_minutes': st.session_state.pre_duration_minutes,
                            'temperature_C': st.session_state.fab_temperature_C,
                            'shots': st.session_state.fab_shots,
                            'laser_frequency_hz': st.session_state.fab_laser_frequency_hz,
                            'laser_fluence': st.session_state.fab_laser_fluence,
                            'gas_pressure_mbar': st.session_state.fab_gas_pressure_mbar,
                            'gas_type': st.session_state.fab_gas_type,
                            'duration_minutes': st.session_state.fab_duration_minutes
                        }
                        csv_content = generate_pld_csv_content(common_data, specific_data)
                    
                    filename = generate_fabrication_filename(
                        st.session_state.fab_substrate_number,
                        st.session_state.fab_institution,
                        st.session_state.fab_operator,
                        st.session_state.fab_sequence,
                        fab_method
                    )
                    
                    st.success("CSV file generated successfully")
                    
                    st.download_button(
                        label="Download CSV File",
                        data=csv_content,
                        file_name=filename,
                        mime="text/csv"
                    )
                    
                    with st.expander("Preview CSV Content"):
                        st.text(csv_content)


with tab3:
    st.header("Treatment CSV File Generator")
    
    treat_method = st.selectbox(
        "Select Treatment Method",
        ["Annealing", "As-deposited", "Storing-in-Glovebox", "Storing-out-Glovebox"]
    )
    
    # Set common default values
    st.session_state.treat_substrate_number = st.session_state.get('treat_substrate_number', "3716-15")
    st.session_state.treat_institution = st.session_state.get('treat_institution', "HZB")
    st.session_state.treat_operator = st.session_state.get('treat_operator', "Steinkopf Lars")
    
    # Set sequence based on method
    if treat_method == "As-deposited":
        st.session_state.treat_sequence = "0"
    else:
        st.session_state.treat_sequence = st.session_state.get('treat_sequence', "1")
    
    st.session_state.treat_date = st.session_state.get('treat_date', datetime.now().date())
    st.session_state.treat_time = st.session_state.get('treat_time', "14:01:15")
    
    st.subheader("Common Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.session_state.treat_substrate_number = st.text_input("Substrate Number", value=st.session_state.treat_substrate_number, key="treat_sn")
        st.session_state.treat_institution = st.text_input("Institution", value=st.session_state.treat_institution, key="treat_inst")
    
    with col2:
        st.session_state.treat_operator = st.text_input("Operator (First and Last Name)", value=st.session_state.treat_operator, key="treat_op")
        
        # Sequence input - always 0 for As-deposited
        if treat_method == "As-deposited":
            st.text_input(
                "Treatment Sequence", 
                value="0", 
                key="treat_seq",
                disabled=True,
                help="As-deposited always uses sequence 0"
            )
            st.session_state.treat_sequence = "0"
        else:
            st.session_state.treat_sequence = st.text_input(
                "Treatment Sequence", 
                value=st.session_state.get('treat_sequence', "1"), 
                key="treat_seq"
            )
    
    with col3:
        st.session_state.treat_date = st.date_input("Treatment Date", value=st.session_state.treat_date, key="treat_date_input")
        st.session_state.treat_time = st.text_input("Treatment Time", value=st.session_state.treat_time, help="Format: HH:MM:SS (24-hour)", key="treat_time_input")
    
    st.divider()
    
    # Treatment Parameters
    st.subheader(f"{treat_method} Parameters")
    
    # Set method-specific defaults
    if treat_method == "As-deposited":
        default_place = "Lab Room 101"
        default_temp = "Room temperature"
        default_duration = "0"
        default_humidity = "0"
        default_oxygen = "0"
        default_gas = "N2"
        default_pressure = "1013"
    elif treat_method in ["Storing-in-Glovebox", "Storing-out-Glovebox"]:
        default_place = "Lab Room 101"
        default_temp = "Room temperature"
        default_duration = "Overnight"
        default_humidity = "100"
        default_oxygen = "50"
        default_gas = "N2"
        default_pressure = "1013"
    else:  # Annealing
        default_place = "Lab Room 101"
        default_temp = "150"
        default_duration = "3600"
        default_humidity = "100"
        default_oxygen = "50"
        default_gas = "N2"
        default_pressure = "1013"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        treat_place = st.text_input("Treatment Place", value=default_place, key=f"treat_place_input_{treat_method}")
        treat_temperature = st.text_input("Temperature (°C)", value=default_temp, key=f"treat_temp_input_{treat_method}")
        treat_duration = st.text_input("Duration (seconds)", value=default_duration, key=f"treat_dur_input_{treat_method}")
    
    with col2:
        treat_humidity = st.text_input("Humidity (ppm)", value=default_humidity, key=f"treat_hum_input_{treat_method}")
        treat_oxygen = st.text_input("O2 Concentration (ppm)", value=default_oxygen, key=f"treat_o2_input_{treat_method}")
    
    with col3:
        treat_gas = st.text_input("Gas", value=default_gas, key=f"treat_gas_input_{treat_method}")
        treat_pressure = st.text_input("Pressure (mbar)", value=default_pressure, key=f"treat_press_input_{treat_method}")
    
    if treat_method == "As-deposited":
        st.info("As-deposited represents samples without post-deposition treatment (sequence is always 0). Environmental parameters can be left at default/ambient values.")
    elif treat_method in ["Storing-in-Glovebox", "Storing-out-Glovebox"]:
        st.info(f"{treat_method}: Samples stored at room temperature. Adjust duration, humidity, and oxygen levels as needed.")
    
    st.divider()
    
    if st.button("Generate Treatment CSV File", type="primary"):
        if not st.session_state.treat_substrate_number or not st.session_state.treat_institution or not st.session_state.treat_operator or not st.session_state.treat_sequence:
            st.error("Please fill in all required fields: Substrate Number, Institution, Operator, and Sequence")
        elif not validate_operator_name(st.session_state.treat_operator):
            st.error("Operator must include both First Name and Last Name")
        else:
            date_formatted = format_date(st.session_state.treat_date)
            time_formatted = convert_time_to_12hour(st.session_state.treat_time)
            
            if not time_formatted:
                st.error("Invalid time format. Please use HH:MM:SS format")
            else:
                common_data = {
                    'substrate_number': st.session_state.treat_substrate_number,
                    'institution': st.session_state.treat_institution,
                    'operator': st.session_state.treat_operator,
                    'method': treat_method,
                    'sequence': st.session_state.treat_sequence,
                    'date': date_formatted,
                    'time': time_formatted
                }
                
                specific_data = {
                    'place': treat_place,
                    'temperature_celsius': treat_temperature,
                    'duration_second': treat_duration,
                    'humidity_ppm': treat_humidity,
                    'oxygen_concentration_ppm': treat_oxygen,
                    'gas': treat_gas,
                    'pressure_mbar': treat_pressure
                }
                
                csv_content = generate_treatment_csv_content(common_data, specific_data)
                
                filename = generate_treatment_filename(
                    st.session_state.treat_substrate_number,
                    st.session_state.treat_institution,
                    st.session_state.treat_operator,
                    st.session_state.treat_sequence,
                    treat_method
                )
                
                st.success("CSV file generated successfully")
                
                st.download_button(
                    label="Download CSV File",
                    data=csv_content,
                    file_name=filename,
                    mime="text/csv"
                )
                
                with st.expander("Preview CSV Content"):
                    st.text(csv_content)


with tab4:
    st.header("SPX & XRF CSV File Generation")
    
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    metadata_dict = render_metadata_section()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### XRF Data (XLS)")
        xrf_xls_file = st.file_uploader(
            "Upload XRF XLS File", type=["xls", "xlsx"],
            help="Upload the XRF analysis results XLS/XLSX file"
        )
    
    with col2:
        st.markdown("##### SPX Files (ZIP)")
        spx_zip_file = st.file_uploader(
            "Upload ZIP file containing SPX files", type=['zip'],
            help="Upload a ZIP file with all SPX spectrum files"
        )
    
    if st.button("Combine XRF and SPX", type="primary"):
        if not metadata_dict['operator_valid']:
            st.error("Please fix the operator name before processing")
        elif xrf_xls_file is None or spx_zip_file is None:
            st.error("Please upload both XRF XLS file and SPX ZIP file")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Reading XRF XLS file...")
                xrf_df = read_xrf_xls(xrf_xls_file)
                xrf_data_list = parse_xrf_xls_to_dict(xrf_df)
                progress_bar.progress(0.2)
                
                status_text.text("Extracting SPX files from ZIP...")
                temp_dir, spx_files = extract_spx_files_from_zip(spx_zip_file)
                
                if not spx_files:
                    st.error("No SPX files found in ZIP")
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                else:
                    st.info(f"Found {len(spx_files)} SPX files")
                    progress_bar.progress(0.3)
                    
                    status_text.text("Processing SPX files...")
                    spx_data_list = []
                    for i, spx_file in enumerate(spx_files):
                        spx_data = parse_spx_file(spx_file)
                        spx_data_list.append(spx_data)
                        progress_bar.progress(0.3 + (0.5 * (i + 1) / len(spx_files)))
                    
                    status_text.text("Matching SPX with XRF data...")
                    combined_data = match_spx_with_xrf_csv(spx_data_list, xrf_data_list)
                    progress_bar.progress(0.9)
                    
                    metadata = {
                        'substrate_number': metadata_dict['substrate_number'],
                        'substrate': metadata_dict['substrate'],
                        'sample_description': metadata_dict['sample_description'],
                        'substrate_size': metadata_dict['substrate_size'],
                        'fabrication_method': metadata_dict['fabrication_method'],
                        'treatment_method': metadata_dict['treatment_method'],
                        'treatment_sequence': metadata_dict['treatment_sequence'],
                        'air_exposure_duration': metadata_dict['air_exposure_duration'],
                        'operator': metadata_dict['operator'],
                        'institution': metadata_dict['institution'],
                        'measurement_type': metadata_dict['measurement_type'],
                        'spectrometer': 'Bruker M4 Tornado',
                        'xrf_fitting_method': metadata_dict['xrf_fitting_method'],
                    }
                    
                    csv_content = create_combined_csv_horizontal_layers(combined_data, metadata, None)
                    
                    created_date = datetime.now().strftime("%Y%m%d")
                    created_time = datetime.now().strftime("%H%M%S")
                    operator_formatted = metadata_dict['operator']
                    csv_filename = f"{metadata_dict['substrate_number']}_{metadata_dict['institution']}_{operator_formatted}_{metadata_dict['treatment_method']}_{metadata_dict['treatment_sequence']}_mapping_xrf_{metadata_dict['xrf_fitting_method']}_original_{created_date}_{created_time}.csv"
                    
                    st.session_state.processed_data = {
                        'combined_data': combined_data,
                        'csv_content': csv_content,
                        'csv_filename': csv_filename,
                        'metadata': metadata,
                        'xrf_df': xrf_df
                    }
                    
                    st.session_state.current_metadata = metadata
                    
                    status_text.empty()
                    progress_bar.progress(1.0)
                    progress_bar.empty()
                    
                    unique_spx = len(set(d['spx_name'] for d in combined_data))
                    matched_count = len(set(d['spx_name'] for d in combined_data if d['matched']))
                    st.success(f"Successfully processed {unique_spx} SPX files! ({matched_count} matched with XRF)")
                    
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    if st.session_state.processed_data is not None:
        processed = st.session_state.processed_data
        combined_data = processed['combined_data']
        
        with st.expander("Combined Data Table & Spectrum Viewer", expanded=False):
            st.markdown("#### Combined Data Table")
            
            df_display = pd.DataFrame([{
                'SPX Name': d['spx_name'],
                'XRF Spectrum': d['xrf_spectrum_name'] if d['xrf_spectrum_name'] else 'N/A',
                'Layer': d['layer_name'],
                'X (mm)': f"{d['x_position_mm']:.3f}" if d['x_position_mm'] else 'N/A',
                'Y (mm)': f"{d['y_position_mm']:.3f}" if d['y_position_mm'] else 'N/A',
                'Z (mm)': f"{d['z_position_mm']:.3f}" if d['z_position_mm'] else 'N/A',
                'Thickness (nm)': f"{d['thickness_nm']:.2f}" if d['thickness_nm'] is not None else 'N/A',
                'Date': d['date'],
                'Time': d['time'],
                'Matched': 'Yes' if d['matched'] else 'No',
            } for d in combined_data])
            
            st.dataframe(df_display, use_container_width=True, height=400, hide_index=True)
                       
            st.markdown("#### Spectrum Viewer")

            if combined_data:
                unique_spx_files = sorted(list(set(d['spx_name'] for d in combined_data)))
                selected_file = st.selectbox("Select file to view spectrum", unique_spx_files, index=0)
                
                # Scale selection
                scale_option = st.radio("Y-axis scale", ["Linear", "Logarithmic"], index=0, horizontal=True)
                yscale = "log" if scale_option == "Logarithmic" else "linear"

                selected_data = next(d for d in combined_data if d['spx_name'] == selected_file)
                selected_layers = [d for d in combined_data if d['spx_name'] == selected_file]

                cols = st.columns(min(len(selected_layers) + 2, 6))
                with cols[0]:
                    st.info(f"**File:** {selected_file}")
                with cols[1]:
                    st.info(f"**Position:** ({selected_data['x_position_mm']:.3f}, {selected_data['y_position_mm']:.3f}) mm")

                for i, layer_data in enumerate(selected_layers):
                    if i + 2 < len(cols):
                        with cols[i+2]:
                            thickness_str = f"{layer_data['thickness_nm']:.2f} nm" if layer_data['thickness_nm'] else "N/A"
                            st.info(f"**{layer_data['layer_name']}:** {thickness_str}")

                fig = plot_spectrum(selected_data, f"Spectrum: {selected_file}", yscale=yscale)
                st.plotly_chart(fig, use_container_width=True)

        
        st.markdown("---")
        st.markdown("### Download CSV (Original XRF Coordinates)")
        st.download_button(
            label="Download CSV File",
            data=processed['csv_content'],
            file_name=processed['csv_filename'],
            mime="text/csv"
        )
        
        st.markdown("---")
        st.subheader("Converting Coordinates: XRF Measurement to Optical Measurement")
        
        xrf_bounds = extract_xrf_bounds(combined_data)
        
        if xrf_bounds is None:
            st.error("Could not extract XRF coordinate bounds from the data")
        else:
            st.success("XRF coordinate bounds extracted from files")
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("##### XRF Measurement Coordinates")
                st.info(f"""
                **First point (x1, y1):** ({xrf_bounds['x_1']:.3f}, {xrf_bounds['y_1']:.3f}) mm  
                **Last point (x2, y2):** ({xrf_bounds['x_2']:.3f}, {xrf_bounds['y_2']:.3f}) mm
                """)
            
            with col_right:
                st.markdown("##### Optical Measurement Coordinates")
                col5, col6 = st.columns(2)
                x_1_opt = col5.number_input("x1 (mm)", value=5.0, format="%.3f")
                y_1_opt = col6.number_input("y1 (mm)", value=5.0, format="%.3f")
                col7, col8 = st.columns(2)
                x_2_opt = col7.number_input("x2 (mm)", value=45.0, format="%.3f")
                y_2_opt = col8.number_input("y2 (mm)", value=45.0, format="%.3f")
            
            optical_bounds = {
                'x_1': x_1_opt, 'y_1': y_1_opt,
                'x_2': x_2_opt, 'y_2': y_2_opt
            }
            
            if st.button("Convert Coordinates", type="primary"):
                converted_data_result = convert_xrf_to_optical(combined_data, xrf_bounds, optical_bounds)
                st.session_state['converted_data'] = converted_data_result
                
                unique_converted = len(set(d['spx_name'] for d in converted_data_result))
                st.success(f"Converted {unique_converted} unique coordinates successfully!")
            
            if 'converted_data' in st.session_state:
                converted_data_result = st.session_state['converted_data']
                
                st.markdown("---")
                fig = plot_coordinate_comparison(xrf_bounds, optical_bounds, converted_data_result)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.subheader("Download CSV with Converted Coordinates")
                
                metadata = st.session_state.current_metadata
                csv_content_converted = create_combined_csv_horizontal_layers(
                    combined_data, metadata, converted_data_result
                )
                
                created_date = datetime.now().strftime("%Y%m%d")
                created_time = datetime.now().strftime("%H%M%S")
                operator_formatted = metadata['operator']
                csv_filename_converted = f"{metadata['substrate_number']}_{metadata['institution']}_{operator_formatted}_{metadata['treatment_method']}_{metadata['treatment_sequence']}_mapping_xrf_{metadata['xrf_fitting_method']}_{created_date}_{created_time}.csv"
                
                st.download_button(
                    label="Download CSV File with Optical Coordinates",
                    data=csv_content_converted,
                    file_name=csv_filename_converted,
                    mime="text/csv",
                    type="primary"

                )






