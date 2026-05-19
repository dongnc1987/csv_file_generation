import streamlit as st
from datetime import datetime
import re
import io
import zipfile


# ==================== SHARED UTILITIES ====================

def validate_operator_name(operator):
    operator_parts = operator.strip().split()
    return len(operator_parts) >= 2


def convert_time_to_12hour(time_str):
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
    except Exception:
        return None


def format_date(date_obj):
    return date_obj.strftime("%A, %B %d, %Y")


def parse_substrate_range(input_str):
    """Parse single substrate number or range syntax (e.g. '3716-01 to 30')."""
    input_str = input_str.strip()
    range_match = re.match(r'^(.+?)-(\d+)\s+to\s+(\d+)$', input_str, re.IGNORECASE)
    if range_match:
        prefix = range_match.group(1)
        start = int(range_match.group(2))
        end = int(range_match.group(3))
        if start > end:
            raise ValueError(f"Range start ({start}) must not exceed range end ({end})")
        pad_width = len(str(end))
        return [f"{prefix}-{str(i).zfill(pad_width)}" for i in range(start, end + 1)]
    return [input_str]


# ==================== SUBSTRATE FUNCTIONS ====================

def generate_substrate_filename(substrate_number, institution, operator, substrate_type):
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{substrate_number}_{institution}_{operator}_substrate_{substrate_type}_{current_datetime}.csv"


def generate_substrate_csv_content(data):
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


# ==================== SUBSTRATE TAB UI ====================

def render_substrate_tab():
    st.header("Substrate CSV File Generator")

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
        st.session_state.sub_substrate_number = st.text_input(
            "Substrate Number",
            value=st.session_state.sub_substrate_number,
            help="Single substrate (e.g. 3716-15) or range (e.g. 3716-1 to 30)",
            key="sub_sn"
        )
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
                try:
                    substrate_numbers = parse_substrate_range(st.session_state.sub_substrate_number)
                except ValueError as e:
                    st.error(str(e))
                    substrate_numbers = []

                if substrate_numbers:
                    common_data = {
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

                    generated_files = []
                    for sub_num in substrate_numbers:
                        substrate_data = {'substrate_number': sub_num, **common_data}
                        csv_content = generate_substrate_csv_content(substrate_data)
                        filename = generate_substrate_filename(
                            sub_num,
                            st.session_state.sub_institution,
                            st.session_state.sub_operator,
                            st.session_state.sub_substrate_type
                        )
                        generated_files.append((filename, csv_content))

                    if len(generated_files) == 1:
                        filename, csv_content = generated_files[0]
                        st.success("CSV file generated successfully")
                        st.download_button(
                            label="Download CSV File",
                            data=csv_content,
                            file_name=filename,
                            mime="text/csv"
                        )
                        with st.expander("Preview CSV Content"):
                            st.text(csv_content)
                    else:
                        zip_buffer = io.BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                            for filename, csv_content in generated_files:
                                zf.writestr(filename, csv_content)
                        zip_buffer.seek(0)
                        st.success(f"{len(generated_files)} CSV files generated successfully")
                        st.download_button(
                            label=f"Download All {len(generated_files)} CSV Files (ZIP)",
                            data=zip_buffer,
                            file_name="substrates.zip",
                            mime="application/zip"
                        )
                        with st.expander("Preview generated substrate numbers"):
                            for sub_num in substrate_numbers:
                                st.text(sub_num)
