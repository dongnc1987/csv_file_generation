import streamlit as st
from datetime import datetime

from substrate_func import validate_operator_name, convert_time_to_12hour, format_date


# ==================== TREATMENT FUNCTIONS ====================

def generate_treatment_filename(substrate_number, institution, operator, sequence, method):
    method_map = {
        'Annealing': 'Annealing',
        'As-deposited': 'As-deposited',
        'Storing-in-Glovebox': 'Storing-in-Glovebox',
        'Storing-out-Glovebox': 'Storing-out-Glovebox'
    }
    method_formatted = method_map.get(method, method)
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{substrate_number}_{institution}_{operator}_treat{sequence}_{method_formatted}_{current_datetime}.csv"


def generate_treatment_csv_content(common_data, specific_data):
    return f"""substrate_number,{common_data['substrate_number']}
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


# ==================== TREATMENT TAB UI ====================

def render_treatment_tab():
    st.header("Treatment CSV File Generator")

    treat_method = st.selectbox(
        "Select Treatment Method",
        ["As-deposited", "Annealing", "Storing-in-Glovebox", "Storing-out-Glovebox"]
    )

    st.session_state.treat_substrate_number = st.session_state.get('treat_substrate_number', "3716-15")
    st.session_state.treat_institution = st.session_state.get('treat_institution', "HZB")
    st.session_state.treat_operator = st.session_state.get('treat_operator', "Steinkopf Lars")

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

    st.subheader(f"{treat_method} Parameters")

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
    else:
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
        treat_temperature = st.text_input("Temperature (C)", value=default_temp, key=f"treat_temp_input_{treat_method}")
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
