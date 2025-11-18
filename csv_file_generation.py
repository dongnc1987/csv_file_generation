import streamlit as st
from datetime import datetime

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


def generate_substrate_filename(sample_number, institution, operator, substrate_type):
    """Generate substrate CSV filename"""
    operator_formatted = operator.replace(' ', '_')
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{sample_number}_{institution}_{operator_formatted}_substrate_{substrate_type}_{current_datetime}.csv"


def generate_fabrication_filename(sample_number, institution, operator, sequence, method):
    """Generate fabrication CSV filename"""
    operator_formatted = operator.replace(' ', '_')
    method_map = {
        'PVD-J': 'PVDJ',
        'Sputtering': 'Sputtering',
        'Tube Furnace': 'TubeFurnace',
        'RTP': 'RTP',
        'PLD': 'PLD'
    }
    method_formatted = method_map.get(method, method.upper())
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{sample_number}_{institution}_{operator_formatted}_fab{sequence}_{method_formatted}_{current_datetime}.csv"


def generate_substrate_csv_content(data):
    """Generate substrate CSV content"""
    return f"""sample_number,{data['sample_number']}
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
    csv_content = f"""sample_number,{common_data['sample_number']}
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
    csv_content = f"""sample_number,{common_data['sample_number']}
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
    csv_content = f"""sample_number,{common_data['sample_number']}
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
    csv_content = f"""sample_number,{common_data['sample_number']}
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
    csv_content = f"""sample_number,{common_data['sample_number']}
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

def generate_treatment_filename(sample_number, institution, operator, sequence, method):
    """Generate treatment CSV filename"""
    operator_formatted = operator.replace(' ', '_')
    method_map = {
        'Annealing': 'Annealing',
        'As-deposited': 'As-deposited'
    }
    method_formatted = method_map.get(method, method)
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{sample_number}_{institution}_{operator_formatted}_treat{sequence}_{method_formatted}_{current_datetime}.csv"


# Replace the generate functions with this single unified function:

def generate_treatment_csv_content(common_data, specific_data):
    """Generate Treatment CSV content (works for both Annealing and As-deposited)"""
    csv_content = f"""sample_number,{common_data['sample_number']}
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

tab1, tab2, tab3 = st.tabs(["Substrate Generation", "Fabrication Generation", "Treatment Generation"])


with tab1:
    st.header("Substrate CSV File Generator")
    
    # Set default values
    st.session_state.sub_sample_number = st.session_state.get('sub_sample_number', "3716-15")
    st.session_state.sub_institution = st.session_state.get('sub_institution', "HZB")
    st.session_state.sub_operator = st.session_state.get('sub_operator', "Dong Nguyen")
    st.session_state.sub_substrate_type = st.session_state.get('sub_substrate_type', "quartz")
    st.session_state.sub_thickness = st.session_state.get('sub_thickness', "1.1")
    st.session_state.sub_size = st.session_state.get('sub_size', "50.8x50.8")
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
        st.session_state.sub_sample_number = st.text_input("Sample Number", value=st.session_state.sub_sample_number, key="sub_sn")
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
        if not st.session_state.sub_sample_number or not st.session_state.sub_institution or not st.session_state.sub_operator or not st.session_state.sub_substrate_type:
            st.error("Please fill in all required fields: Sample Number, Institution, Operator, and Substrate Type")
        elif not validate_operator_name(st.session_state.sub_operator):
            st.error("Operator must include both First Name and Last Name (e.g., Dong Nguyen)")
        else:
            date_formatted = format_date(st.session_state.sub_clean_date)
            time_formatted = convert_time_to_12hour(st.session_state.sub_clean_time)
            
            if not time_formatted:
                st.error("Invalid time format. Please use HH:MM:SS format")
            else:
                substrate_data = {
                    'sample_number': st.session_state.sub_sample_number,
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
                    st.session_state.sub_sample_number,
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
        ["PVD-J", "Sputtering", "Tube Furnace", "RTP", "PLD"]
    )
    
    # Set common default values
    st.session_state.fab_sample_number = st.session_state.get('fab_sample_number', "3716-15")
    st.session_state.fab_institution = st.session_state.get('fab_institution', "HZB")
    st.session_state.fab_operator = st.session_state.get('fab_operator', "Lars Drescher")
    st.session_state.fab_sequence = st.session_state.get('fab_sequence', "1")
    st.session_state.fab_date = st.session_state.get('fab_date', datetime.now().date())
    st.session_state.fab_time = st.session_state.get('fab_time', "14:01:15")
    
    st.subheader("Common Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.session_state.fab_sample_number = st.text_input("Sample Number", value=st.session_state.fab_sample_number, key="fab_sn")
        st.session_state.fab_institution = st.text_input("Institution", value=st.session_state.fab_institution, key="fab_inst")
    
    with col2:
        st.session_state.fab_operator = st.text_input("Operator (First and Last Name)", value=st.session_state.fab_operator, key="fab_op")
        st.session_state.fab_sequence = st.text_input("Fabrication Sequence", value=st.session_state.fab_sequence, key="fab_seq")
    
    with col3:
        st.session_state.fab_date = st.date_input("Fabrication Date", value=st.session_state.fab_date, key="fab_date_input")
        st.session_state.fab_time = st.text_input("Fabrication Time", value=st.session_state.fab_time, help="Format: HH:MM:SS (24-hour)", key="fab_time_input")
    
    st.divider()
    
    # Method-specific parameters with default values
    if fab_method == "PVD-J":
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
    
    st.divider()
    
    if st.button("Generate Fabrication CSV File", type="primary"):
        if not st.session_state.fab_sample_number or not st.session_state.fab_institution or not st.session_state.fab_operator or not st.session_state.fab_sequence:
            st.error("Please fill in all required fields: Sample Number, Institution, Operator, and Sequence")
        elif not validate_operator_name(st.session_state.fab_operator):
            st.error("Operator must include both First Name and Last Name")
        else:
            date_formatted = format_date(st.session_state.fab_date)
            time_formatted = convert_time_to_12hour(st.session_state.fab_time)
            
            if not time_formatted:
                st.error("Invalid time format. Please use HH:MM:SS format")
            else:
                common_data = {
                    'sample_number': st.session_state.fab_sample_number,
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
                    st.session_state.fab_sample_number,
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
        ["Annealing", "As-deposited"]
    )
    
    # Set common default values
    st.session_state.treat_sample_number = st.session_state.get('treat_sample_number', "3716-15")
    st.session_state.treat_institution = st.session_state.get('treat_institution', "HZB")
    st.session_state.treat_operator = st.session_state.get('treat_operator', "Dong Nguyen")
    
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
        st.session_state.treat_sample_number = st.text_input("Sample Number", value=st.session_state.treat_sample_number, key="treat_sn")
        st.session_state.treat_institution = st.text_input("Institution", value=st.session_state.treat_institution, key="treat_inst")
    
    with col2:
        st.session_state.treat_operator = st.text_input("Operator (First and Last Name)", value=st.session_state.treat_operator, key="treat_op")
        
        # Sequence input - disabled for As-deposited
        if treat_method == "As-deposited":
            st.session_state.treat_sequence = st.text_input(
                "Treatment Sequence", 
                value="0", 
                key="treat_seq",
                disabled=True,
                help="As-deposited always uses sequence 0"
            )
        else:
            st.session_state.treat_sequence = st.text_input(
                "Treatment Sequence", 
                value=st.session_state.treat_sequence, 
                key="treat_seq"
            )
    
    with col3:
        st.session_state.treat_date = st.date_input("Treatment Date", value=st.session_state.treat_date, key="treat_date_input")
        st.session_state.treat_time = st.text_input("Treatment Time", value=st.session_state.treat_time, help="Format: HH:MM:SS (24-hour)", key="treat_time_input")
    
    st.divider()
    
    # Treatment Parameters (same for both methods)
    st.subheader(f"{treat_method} Parameters")
    
    # Set default values based on method
    if treat_method == "As-deposited":
        default_temp = "25"
        default_duration = "0"
        default_humidity = "0"
        default_oxygen = "0"
        default_gas = "Air"
        default_pressure = "1013"
    else:  # Annealing
        default_temp = "150"
        default_duration = "3600"
        default_humidity = "100"
        default_oxygen = "50"
        default_gas = "N2"
        default_pressure = "1013"
    
    # Initialize default values (use different keys to avoid conflicts)
    default_place = st.session_state.get('treat_place_val', "Lab Room 101")
    default_temp = st.session_state.get('treat_temp_val', default_temp)
    default_duration = st.session_state.get('treat_dur_val', default_duration)
    default_humidity = st.session_state.get('treat_hum_val', default_humidity)
    default_oxygen = st.session_state.get('treat_o2_val', default_oxygen)
    default_gas = st.session_state.get('treat_gas_val', default_gas)
    default_pressure = st.session_state.get('treat_press_val', default_pressure)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        treat_place = st.text_input("Treatment Place", value=default_place, key="treat_place_input")
        treat_temperature = st.text_input("Temperature (°C)", value=default_temp, key="treat_temp_input")
        treat_duration = st.text_input("Duration (seconds)", value=default_duration, key="treat_dur_input")
    
    with col2:
        treat_humidity = st.text_input("Humidity (ppm)", value=default_humidity, key="treat_hum_input")
        treat_oxygen = st.text_input("O2 Concentration (ppm)", value=default_oxygen, key="treat_o2_input")
    
    with col3:
        treat_gas = st.text_input("Gas", value=default_gas, key="treat_gas_input")
        treat_pressure = st.text_input("Pressure (mbar)", value=default_pressure, key="treat_press_input")
    
    if treat_method == "As-deposited":
        st.info(" As-deposited represents samples without post-deposition treatment (sequence is always 0). Environmental parameters can be left at default/ambient values.")
    
    st.divider()
    
    if st.button("Generate Treatment CSV File", type="primary"):
        if not st.session_state.treat_sample_number or not st.session_state.treat_institution or not st.session_state.treat_operator or not st.session_state.treat_sequence:
            st.error("Please fill in all required fields: Sample Number, Institution, Operator, and Sequence")
        elif not validate_operator_name(st.session_state.treat_operator):
            st.error("Operator must include both First Name and Last Name")
        else:
            date_formatted = format_date(st.session_state.treat_date)
            time_formatted = convert_time_to_12hour(st.session_state.treat_time)
            
            if not time_formatted:
                st.error("Invalid time format. Please use HH:MM:SS format")
            else:
                common_data = {
                    'sample_number': st.session_state.treat_sample_number,
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
                
                # Save current values to session state for next time
                st.session_state.treat_place_val = treat_place
                st.session_state.treat_temp_val = treat_temperature
                st.session_state.treat_dur_val = treat_duration
                st.session_state.treat_hum_val = treat_humidity
                st.session_state.treat_o2_val = treat_oxygen
                st.session_state.treat_gas_val = treat_gas
                st.session_state.treat_press_val = treat_pressure
                
                csv_content = generate_treatment_csv_content(common_data, specific_data)
                
                filename = generate_treatment_filename(
                    st.session_state.treat_sample_number,
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