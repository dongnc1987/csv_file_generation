import streamlit as st
from datetime import datetime

from substrate_func import validate_operator_name, convert_time_to_12hour, format_date


# ==================== FABRICATION FUNCTIONS ====================

def generate_fabrication_filename(substrate_number, institution, operator, sequence, method):
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
    return f"{substrate_number}_{institution}_{operator}_fab{sequence}_{method_formatted}_{current_datetime}.csv"


def parse_pvdp_csv(uploaded_file):
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
            metadata['date'] = line.replace('# Date:', '').strip()
        elif line.startswith('# Time:'):
            metadata['time'] = line.replace('# Time:', '').strip()
        elif line.startswith('# Substrate Number:'):
            metadata['substrate_number'] = line.replace('# Substrate Number:', '').strip()
        elif line.startswith('# process ID:'):
            metadata['process_id'] = line.replace('# process ID:', '').strip()
        elif line.startswith('# operator:'):
            metadata['operator_code'] = line.replace('# operator:', '').strip()
    return metadata, content


def replace_operator_in_csv(csv_content, old_operator, new_operator):
    return csv_content.replace(f"# operator: {old_operator}", f"# operator: {new_operator}")


def generate_pvdp_filename_from_metadata(substrate_number, institution, operator, sequence, date_str, time_str):
    date_obj = datetime.strptime(date_str, "%Y/%m/%d")
    date_formatted = date_obj.strftime("%Y%m%d")
    time_formatted = time_str.replace(':', '')
    return f"{substrate_number}_{institution}_{operator}_fab{sequence}_PVDP_{date_formatted}_{time_formatted}.csv"


def generate_pvdj_csv_content(common_data, specific_data):
    return f"""substrate_number,{common_data['substrate_number']}
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


def generate_sputtering_csv_content(common_data, specific_data):
    return f"""substrate_number,{common_data['substrate_number']}
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


def generate_tubefurnace_csv_content(common_data, specific_data):
    return f"""substrate_number,{common_data['substrate_number']}
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


def generate_rtp_csv_content(common_data, specific_data):
    return f"""substrate_number,{common_data['substrate_number']}
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


_PLD_RECIPE_DEFAULTS = {
    'name': '',
    'type': 'Deposition',
    'index': '0',
    'duration_min': '300',
    'wedge': 'n',
    'wedge_corrected': 'n',
    'volume_per_shot_mm3': '-1',
    'desired_thickness_mm': 'NA',
    'xsub_left_lim_mm': '20',
    'xsub_right_lim_mm': '20',
    'scanner_amp_mm': '35',
    'laser_frequency_hz': '30',
    'nos': '40000',
    'xsub_velocity_mm_per_s': '2',
    'scanner_velocity_mm_per_s': '60.00',
    'layers': 'NA',
    'sub_temp_heater1_C': '20',
    'heat_rate_heater1_C_per_min': '20',
    'sub_temp_heater2_C': '20',
    'heat_rate_heater2_C_per_min': '20',
    'sub_temp_heater3_C': '20',
    'heat_rate_heater3_C_per_min': '20',
    'target_material': 'Cu',
    'target_diameter_mm': '',
    'target_cycle': '',
    'target_premix_fill_date': '',
    'target_rotation_rpm': '20',
    'fluence_energy_mJ': '3',
    'pressure_mbar': '1E-4',
    'gas_O2_sccm': '0',
    'gas_N2_sccm': '1',
    'gas_Ar_sccm': '0',
    'substrate_rot_deg': '0',
    'zsub_pos_cm': '75',
    'atten_angle_deg': '0',
    'correction_1x': 'NA',
    'correction_2x': 'NA',
    'correction_3x': 'NA',
    'correction_4x': 'NA',
    'correction_5x': 'NA',
    'dx_corrected_1_mm': 'NA',
    'dx_corrected_2_mm': 'NA',
    'dx_corrected_3_mm': 'NA',
}


def _make_pld_recipe():
    if 'pld_recipe_id_counter' not in st.session_state:
        st.session_state.pld_recipe_id_counter = 0
    st.session_state.pld_recipe_id_counter += 1
    recipe = dict(_PLD_RECIPE_DEFAULTS)
    recipe['_id'] = st.session_state.pld_recipe_id_counter
    return recipe


def generate_pld_csv_content(common_data, pre_ablation, process_data, recipes):
    lines = [
        f"substrate_number,{common_data['substrate_number']}",
        f"fab_method,{common_data['method']}",
        f"fab_sequence,{common_data['sequence']}",
        f"fab_operator,{common_data['operator']}",
        f"fab_institution,{common_data['institution']}",
        f'fab_date,"{common_data["date"]}"',
        f'fab_time,"{common_data["time"]}"',
        "pre_ablation,Pre-Ablation Parameters",
        f"pre_ablation_target,{pre_ablation['target']}",
        f"pre_shots,{pre_ablation['shots']}",
        f"pre_laser_frequency_hz,{pre_ablation['laser_frequency_hz']}",
        f"pre_laser_fluence,{pre_ablation['laser_fluence']}",
        f"pre_gas_pressure_mbar,{pre_ablation['gas_pressure_mbar']}",
        f"pre_gas_type,{pre_ablation['gas_type']}",
        f"pre_duration_minutes,{pre_ablation['duration_minutes']}",
        "fab_process,Process Parameters",
        f"fab_process_name,{process_data['name']}",
        f"fab_substrate_size,{process_data['substrate_size']}",
        f"fab_sample_count,{process_data['sample_count']}",
        f"fab_mask_aperture,{process_data['mask_aperture']}",
        f"fab_plasma,{process_data['plasma']}",
        f"fab_sample_holder,{process_data['sample_holder']}",
        f"fab_recipe_count,{len(recipes)}",
    ]

    for i, recipe in enumerate(recipes, start=1):
        p = f"recipe_{i}"
        lines.append(f"{p},")
        lines.append(f"{p}_name,{recipe['name']}")
        lines.append(f"{p}_type,{recipe['type']}")
        lines.append(f"{p}_index,{recipe['index']}")
        lines.append(f"{p}_duration_min,{recipe['duration_min']}")
        lines.append(f"{p}_wedge,{recipe['wedge']}")
        lines.append(f"{p}_wedge_corrected,{recipe['wedge_corrected']}")
        lines.append(f"{p}_volume_per_shot_mm3,{recipe['volume_per_shot_mm3']}")
        lines.append(f"{p}_desired_thickness_mm,{recipe['desired_thickness_mm']}")
        lines.append(f"{p}_xsub_left_lim_mm,{recipe['xsub_left_lim_mm']}")
        lines.append(f"{p}_xsub_right_lim_mm,{recipe['xsub_right_lim_mm']}")
        lines.append(f"{p}_scanner_amp_mm,{recipe['scanner_amp_mm']}")
        lines.append(f"{p}_laser_frequency_hz,{recipe['laser_frequency_hz']}")
        lines.append(f"{p}_nos,{recipe['nos']}")
        lines.append(f"{p}_xsub_velocity_mm_per_s,{recipe['xsub_velocity_mm_per_s']}")
        lines.append(f"{p}_scanner_velocity_mm_per_s,{recipe['scanner_velocity_mm_per_s']}")
        lines.append(f"{p}_layers,{recipe['layers']}")
        lines.append(f"{p}_sub_temp_heater1_C,{recipe['sub_temp_heater1_C']}")
        lines.append(f"{p}_heat_rate_heater1_C_per_min,{recipe['heat_rate_heater1_C_per_min']}")
        lines.append(f"{p}_sub_temp_heater2_C,{recipe['sub_temp_heater2_C']}")
        lines.append(f"{p}_heat_rate_heater2_C_per_min,{recipe['heat_rate_heater2_C_per_min']}")
        lines.append(f"{p}_sub_temp_heater3_C,{recipe['sub_temp_heater3_C']}")
        lines.append(f"{p}_heat_rate_heater3_C_per_min,{recipe['heat_rate_heater3_C_per_min']}")
        lines.append(f"{p}_target_material,{recipe['target_material']}")
        lines.append(f"{p}_target_diameter_mm,{recipe['target_diameter_mm']}")
        lines.append(f"{p}_target_cycle,{recipe['target_cycle']}")
        lines.append(f"{p}_target_premix_fill_date,{recipe['target_premix_fill_date']}")
        lines.append(f"{p}_target_rotation_rpm,{recipe['target_rotation_rpm']}")
        lines.append(f"{p}_fluence_energy_mJ,{recipe['fluence_energy_mJ']}")
        lines.append(f"{p}_pressure_mbar,{recipe['pressure_mbar']}")
        lines.append(f"{p}_gas_O2_sccm,{recipe['gas_O2_sccm']}")
        lines.append(f"{p}_gas_N2_sccm,{recipe['gas_N2_sccm']}")
        lines.append(f"{p}_gas_Ar_sccm,{recipe['gas_Ar_sccm']}")
        lines.append(f"{p}_substrate_rot_deg,{recipe['substrate_rot_deg']}")
        lines.append(f"{p}_zsub_pos_cm,{recipe['zsub_pos_cm']}")
        lines.append(f"{p}_atten_angle_deg,{recipe['atten_angle_deg']}")
        lines.append(f"{p}_correction_1x,{recipe['correction_1x']}")
        lines.append(f"{p}_correction_2x,{recipe['correction_2x']}")
        lines.append(f"{p}_correction_3x,{recipe['correction_3x']}")
        lines.append(f"{p}_correction_4x,{recipe['correction_4x']}")
        lines.append(f"{p}_correction_5x,{recipe['correction_5x']}")
        lines.append(f"{p}_dx_corrected_1_mm,{recipe['dx_corrected_1_mm']}")
        lines.append(f"{p}_dx_corrected_2_mm,{recipe['dx_corrected_2_mm']}")
        lines.append(f"{p}_dx_corrected_3_mm,{recipe['dx_corrected_3_mm']}")

    return "\n".join(lines) + "\n"


# ==================== FABRICATION TAB UI ====================

def render_fabrication_tab():
    st.header("Fabrication CSV File Generator")

    fab_method = st.selectbox(
        "Select Fabrication Method",
        ["PVD-J", "Sputtering", "Tube Furnace", "RTP", "PLD", "PVD-P"]
    )

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
        st.session_state.pre_ablation_target = st.session_state.get('pre_ablation_target', "On Sample")
        st.session_state.pre_ablation_target = st.radio(
            "Pre-Ablation Target",
            ["On Sample", "On Dummy"],
            index=["On Sample", "On Dummy"].index(st.session_state.pre_ablation_target),
            horizontal=True,
            key="pld_pre_target"
        )
        st.session_state.pre_shots = st.session_state.get('pre_shots', "100")
        st.session_state.pre_laser_frequency_hz = st.session_state.get('pre_laser_frequency_hz', "10")
        st.session_state.pre_laser_fluence = st.session_state.get('pre_laser_fluence', "2.0")
        st.session_state.pre_gas_pressure_mbar = st.session_state.get('pre_gas_pressure_mbar', "0.1")
        st.session_state.pre_gas_type = st.session_state.get('pre_gas_type', "Oxygen")
        st.session_state.pre_duration_minutes = st.session_state.get('pre_duration_minutes', "5")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.pre_shots = st.text_input("Shots", value=st.session_state.pre_shots, key="pld_pre_shots")
            st.session_state.pre_laser_frequency_hz = st.text_input("Laser Frequency (Hz)", value=st.session_state.pre_laser_frequency_hz, key="pld_pre_freq")
        with col2:
            st.session_state.pre_laser_fluence = st.text_input("Laser Fluence", value=st.session_state.pre_laser_fluence, key="pld_pre_flu")
            st.session_state.pre_gas_pressure_mbar = st.text_input("Gas Pressure (mbar)", value=st.session_state.pre_gas_pressure_mbar, key="pld_pre_press")
        with col3:
            st.session_state.pre_gas_type = st.text_input("Gas Type", value=st.session_state.pre_gas_type, key="pld_pre_gas")
            st.session_state.pre_duration_minutes = st.text_input("Duration (minutes)", value=st.session_state.pre_duration_minutes, key="pld_pre_dur")

        st.divider()
        st.markdown("**Process**")
        st.session_state.pld_process_name = st.session_state.get('pld_process_name', "")
        st.session_state.pld_substrate_size = st.session_state.get('pld_substrate_size', "")
        st.session_state.pld_sample_count = st.session_state.get('pld_sample_count', "")
        st.session_state.pld_mask_aperture = st.session_state.get('pld_mask_aperture', "")
        st.session_state.pld_plasma = st.session_state.get('pld_plasma', "")
        st.session_state.pld_sample_holder = st.session_state.get('pld_sample_holder', "")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.pld_process_name = st.text_input(
                "Process Name", value=st.session_state.pld_process_name,
                placeholder="e.g. Cu_20260506_run1", key="pld_process_name_input"
            )
        with col2:
            st.session_state.pld_substrate_size = st.text_input(
                "Substrate Size (mm)", value=st.session_state.pld_substrate_size,
                placeholder="e.g. 50x50, 12x50, 10x10", key="pld_substrate_size_input"
            )
        with col3:
            st.session_state.pld_sample_count = st.text_input(
                "Number of Samples", value=st.session_state.pld_sample_count,
                placeholder="e.g. 4", key="pld_sample_count_input"
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.session_state.pld_mask_aperture = st.text_input(
                "Mask Aperture", value=st.session_state.pld_mask_aperture,
                placeholder="e.g. 10x10 mm", key="pld_mask_aperture_input"
            )
        with col2:
            st.session_state.pld_plasma = st.text_input(
                "Plasma", value=st.session_state.pld_plasma,
                placeholder="e.g. off", key="pld_plasma_input"
            )
        with col3:
            st.session_state.pld_sample_holder = st.text_input(
                "Sample Holder", value=st.session_state.pld_sample_holder,
                placeholder="e.g. Standard", key="pld_sample_holder_input"
            )

        st.markdown("**Deposition Recipes**")

        if 'pld_recipes' not in st.session_state:
            st.session_state.pld_recipes = [_make_pld_recipe()]

        col_add, col_info = st.columns([1, 3])
        with col_add:
            if st.button("Add Recipe", key="pld_add_recipe"):
                st.session_state.pld_recipes.append(_make_pld_recipe())
                st.rerun()
        with col_info:
            st.info(f"Total recipes: {len(st.session_state.pld_recipes)}")

        for i, recipe in enumerate(st.session_state.pld_recipes):
            rid = recipe['_id']
            with st.expander(f"Recipe {i + 1}: {recipe['name']}", expanded=(i == 0)):
                if len(st.session_state.pld_recipes) > 1:
                    if st.button(f"Remove Recipe {i + 1}", key=f"pld_remove_{rid}"):
                        st.session_state.pld_recipes.pop(i)
                        st.rerun()

                st.markdown("**Recipe Metadata**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    recipe['name'] = st.text_input("Recipe Name", value=recipe['name'], placeholder="e.g. K1_Cu_QT_P1E", key=f"pld_{rid}_name")
                with col2:
                    recipe['index'] = st.text_input("Index", value=recipe['index'], key=f"pld_{rid}_index")
                with col3:
                    recipe['duration_min'] = st.text_input("Duration (min)", value=recipe['duration_min'], key=f"pld_{rid}_duration")

                st.markdown("**Scan Parameters**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    recipe['wedge'] = st.text_input("Wedge (y/n)", value=recipe['wedge'], key=f"pld_{rid}_wedge")
                    recipe['wedge_corrected'] = st.text_input("Wedge Corrected (y/n)", value=recipe['wedge_corrected'], key=f"pld_{rid}_wedge_corr")
                    recipe['volume_per_shot_mm3'] = st.text_input("Volume/Shot (mm3)", value=recipe['volume_per_shot_mm3'], key=f"pld_{rid}_vol_shot")
                    recipe['desired_thickness_mm'] = st.text_input("Desired Thickness (mm)", value=recipe['desired_thickness_mm'], key=f"pld_{rid}_des_thick")
                with col2:
                    recipe['xsub_left_lim_mm'] = st.text_input("xSub Left Lim (mm)", value=recipe['xsub_left_lim_mm'], key=f"pld_{rid}_xleft")
                    recipe['xsub_right_lim_mm'] = st.text_input("xSub Right Lim (mm)", value=recipe['xsub_right_lim_mm'], key=f"pld_{rid}_xright")
                    recipe['scanner_amp_mm'] = st.text_input("Scanner Amp (mm)", value=recipe['scanner_amp_mm'], key=f"pld_{rid}_scan_amp")
                    recipe['laser_frequency_hz'] = st.text_input("Laser Frequency (Hz)", value=recipe['laser_frequency_hz'], key=f"pld_{rid}_laser_freq")
                with col3:
                    recipe['nos'] = st.text_input("nos (#)", value=recipe['nos'], key=f"pld_{rid}_nos")
                    recipe['xsub_velocity_mm_per_s'] = st.text_input("xSub Velocity (mm/s)", value=recipe['xsub_velocity_mm_per_s'], key=f"pld_{rid}_xsub_vel")
                    recipe['scanner_velocity_mm_per_s'] = st.text_input("Scanner Velocity (mm/s)", value=recipe['scanner_velocity_mm_per_s'], key=f"pld_{rid}_scan_vel")
                    recipe['layers'] = st.text_input("Layers (#)", value=recipe['layers'], key=f"pld_{rid}_layers")

                st.markdown("**Heater Parameters**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    recipe['sub_temp_heater1_C'] = st.text_input("Sub. Temp. Heater 1 (C)", value=recipe['sub_temp_heater1_C'], key=f"pld_{rid}_h1_temp")
                    recipe['heat_rate_heater1_C_per_min'] = st.text_input("Heat Rate Heater 1 (C/min)", value=recipe['heat_rate_heater1_C_per_min'], key=f"pld_{rid}_h1_rate")
                with col2:
                    recipe['sub_temp_heater2_C'] = st.text_input("Sub. Temp. Heater 2 (C)", value=recipe['sub_temp_heater2_C'], key=f"pld_{rid}_h2_temp")
                    recipe['heat_rate_heater2_C_per_min'] = st.text_input("Heat Rate Heater 2 (C/min)", value=recipe['heat_rate_heater2_C_per_min'], key=f"pld_{rid}_h2_rate")
                with col3:
                    recipe['sub_temp_heater3_C'] = st.text_input("Sub. Temp. Heater 3 (C)", value=recipe['sub_temp_heater3_C'], key=f"pld_{rid}_h3_temp")
                    recipe['heat_rate_heater3_C_per_min'] = st.text_input("Heat Rate Heater 3 (C/min)", value=recipe['heat_rate_heater3_C_per_min'], key=f"pld_{rid}_h3_rate")

                st.markdown("**Target and Process Parameters**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    recipe['target_material'] = st.text_input("Target Material", value=recipe['target_material'], key=f"pld_{rid}_target_mat")
                    recipe['target_diameter_mm'] = st.text_input("Target Diameter (mm)", value=recipe['target_diameter_mm'], placeholder="e.g. 50", key=f"pld_{rid}_target_dia")
                    recipe['target_cycle'] = st.text_input("Target Cycle", value=recipe['target_cycle'], placeholder="e.g. 1", key=f"pld_{rid}_target_cycle")
                    recipe['target_premix_fill_date'] = st.text_input("Pre-mix Fill Date", value=recipe['target_premix_fill_date'], placeholder="e.g. 2026-05-01", key=f"pld_{rid}_premix_date")
                    recipe['target_rotation_rpm'] = st.text_input("Target Rotation (rpm)", value=recipe['target_rotation_rpm'], key=f"pld_{rid}_target_rot")
                    recipe['fluence_energy_mJ'] = st.text_input("Fluence/Energy (mJ)", value=recipe['fluence_energy_mJ'], key=f"pld_{rid}_fluence")
                    recipe['pressure_mbar'] = st.text_input("Pressure (mbar)", value=recipe['pressure_mbar'], key=f"pld_{rid}_pressure")
                with col2:
                    recipe['gas_O2_sccm'] = st.text_input("Gas O2 (sccm)", value=recipe['gas_O2_sccm'], key=f"pld_{rid}_gas_o2")
                    recipe['gas_N2_sccm'] = st.text_input("Gas N2 (sccm)", value=recipe['gas_N2_sccm'], key=f"pld_{rid}_gas_n2")
                    recipe['gas_Ar_sccm'] = st.text_input("Gas Ar (sccm)", value=recipe['gas_Ar_sccm'], key=f"pld_{rid}_gas_ar")
                with col3:
                    recipe['substrate_rot_deg'] = st.text_input("Substrate Rotation (deg)", value=recipe['substrate_rot_deg'], key=f"pld_{rid}_sub_rot")
                    recipe['zsub_pos_cm'] = st.text_input("zSub Position (cm)", value=recipe['zsub_pos_cm'], key=f"pld_{rid}_zsub")
                    recipe['atten_angle_deg'] = st.text_input("Attenuation Angle (deg)", value=recipe['atten_angle_deg'], key=f"pld_{rid}_atten")

                st.markdown("**Correction Parameters**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    recipe['correction_1x'] = st.text_input("Correction 1x", value=recipe['correction_1x'], key=f"pld_{rid}_corr1")
                    recipe['correction_2x'] = st.text_input("Correction 2x", value=recipe['correction_2x'], key=f"pld_{rid}_corr2")
                    recipe['dx_corrected_1_mm'] = st.text_input("dX-Corrected 1 (mm)", value=recipe['dx_corrected_1_mm'], key=f"pld_{rid}_dx1")
                with col2:
                    recipe['correction_3x'] = st.text_input("Correction 3x", value=recipe['correction_3x'], key=f"pld_{rid}_corr3")
                    recipe['correction_4x'] = st.text_input("Correction 4x", value=recipe['correction_4x'], key=f"pld_{rid}_corr4")
                    recipe['dx_corrected_2_mm'] = st.text_input("dX-Corrected 2 (mm)", value=recipe['dx_corrected_2_mm'], key=f"pld_{rid}_dx2")
                with col3:
                    recipe['correction_5x'] = st.text_input("Correction 5x", value=recipe['correction_5x'], key=f"pld_{rid}_corr5")
                    recipe['dx_corrected_3_mm'] = st.text_input("dX-Corrected 3 (mm)", value=recipe['dx_corrected_3_mm'], key=f"pld_{rid}_dx3")

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
                        pre_ablation = {
                            'target': st.session_state.pre_ablation_target,
                            'shots': st.session_state.pre_shots,
                            'laser_frequency_hz': st.session_state.pre_laser_frequency_hz,
                            'laser_fluence': st.session_state.pre_laser_fluence,
                            'gas_pressure_mbar': st.session_state.pre_gas_pressure_mbar,
                            'gas_type': st.session_state.pre_gas_type,
                            'duration_minutes': st.session_state.pre_duration_minutes,
                        }
                        process_data = {
                            'name': st.session_state.pld_process_name,
                            'substrate_size': st.session_state.pld_substrate_size,
                            'sample_count': st.session_state.pld_sample_count,
                            'mask_aperture': st.session_state.pld_mask_aperture,
                            'plasma': st.session_state.pld_plasma,
                            'sample_holder': st.session_state.pld_sample_holder,
                        }
                        csv_content = generate_pld_csv_content(
                            common_data, pre_ablation, process_data,
                            st.session_state.pld_recipes
                        )

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
