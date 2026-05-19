import streamlit as st
from datetime import datetime
from pathlib import Path
import zipfile
import tempfile
import shutil

from spx_processing_func import *
from substrate_func import render_substrate_tab
from fabrication_func import render_fabrication_tab
from treatment_function import render_treatment_tab
from material_func import render_material_tab


def extract_spx_files_from_zip(zip_file):
    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        temp_path = Path(temp_dir)
        spx_files = list(temp_path.rglob('*.spx'))
        return temp_dir, sorted(spx_files)
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e


st.set_page_config(layout="wide")
st.title("CSV File Generator for Sample Database")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Substrate Generation",
    "Fabrication Generation",
    "Treatment Generation",
    "XRF&SPX Generation",
    "Material Registry"
])

with tab1:
    render_substrate_tab()

with tab2:
    render_fabrication_tab()

with tab3:
    render_treatment_tab()

with tab5:
    render_material_tab()

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
            help="Upload the XRF analysis results XLS/XLSX file",
            key="xrf_uploader"
        )

    with col2:
        st.markdown("##### SPX Files (ZIP)")
        spx_zip_file = st.file_uploader(
            "Upload ZIP file containing SPX files", type=['zip'],
            help="Upload a ZIP file with all SPX spectrum files",
            key="spx_uploader"
        )

    if st.button("Combine XRF and SPX", type="primary", key="combine_button"):
        if not metadata_dict['operator_valid']:
            st.error("Please fix the operator name before processing")
        elif xrf_xls_file is None or spx_zip_file is None:
            st.error("Please upload both XRF XLS file and SPX ZIP file")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                progress_bar.progress(0.1)

                status_text.text("Reading XRF XLS file...")
                xrf_df = read_xrf_xls(xrf_xls_file)
                xrf_data_list = parse_xrf_xls_to_dict(xrf_df)
                progress_bar.progress(0.2)

                status_text.text("Extracting SPX files from ZIP...")
                temp_dir, spx_files = extract_spx_files_from_zip(spx_zip_file)

                if not spx_files:
                    st.error("No SPX files found in ZIP")
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
                    progress_bar.progress(0.85)

                    status_text.text("Generating CSV with metadata...")

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
                        'x_method_name': metadata_dict['x_method_name'],
                        'x_method_description': metadata_dict['x_method_description']
                    }

                    csv_content = create_combined_csv_horizontal_layers(combined_data, metadata, None)

                    created_date = datetime.now().strftime("%Y%m%d")
                    created_time = datetime.now().strftime("%H%M%S")
                    operator_formatted = metadata_dict['operator']
                    x_method_base = Path(metadata['x_method_name']).stem

                    csv_filename = (
                        f"{metadata_dict['substrate_number']}_{metadata_dict['institution']}_"
                        f"{operator_formatted}_{metadata_dict['treatment_method']}_"
                        f"{metadata_dict['treatment_sequence']}_mapping_xrf_"
                        f"{x_method_base}_original_{created_date}_{created_time}.csv"
                    )

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
                selected_file = st.selectbox("Select file to view spectrum", unique_spx_files, index=0, key="spectrum_selector")

                scale_option = st.radio("Y-axis scale", ["Linear", "Logarithmic"], index=0, horizontal=True, key="yscale_radio")
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
                        with cols[i + 2]:
                            thickness_str = f"{layer_data['thickness_nm']:.2f} nm" if layer_data['thickness_nm'] else "N/A"
                            st.info(f"**{layer_data['layer_name']}:** {thickness_str}")

                fig = plot_spectrum(selected_data, f"Spectrum: {selected_file}", yscale=yscale)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### Download CSV (Original XRF Coordinates)")

        col_dl1, col_dl2 = st.columns([3, 1])
        with col_dl1:
            st.download_button(
                label="Download CSV File",
                data=processed['csv_content'],
                file_name=processed['csv_filename'],
                mime="text/csv",
                key="download_csv_original"
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
                x_1_opt = col5.number_input("x1 (mm)", value=5.0, format="%.3f", key="x1_opt")
                y_1_opt = col6.number_input("y1 (mm)", value=5.0, format="%.3f", key="y1_opt")
                col7, col8 = st.columns(2)
                x_2_opt = col7.number_input("x2 (mm)", value=45.0, format="%.3f", key="x2_opt")
                y_2_opt = col8.number_input("y2 (mm)", value=45.0, format="%.3f", key="y2_opt")

            optical_bounds = {
                'x_1': x_1_opt, 'y_1': y_1_opt,
                'x_2': x_2_opt, 'y_2': y_2_opt
            }

            if st.button("Convert Coordinates", type="primary", key="convert_coords_button"):
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
                x_method_base = Path(metadata['x_method_name']).stem
                csv_filename_converted = (
                    f"{metadata['substrate_number']}_{metadata['institution']}_"
                    f"{operator_formatted}_{metadata['treatment_method']}_"
                    f"{metadata['treatment_sequence']}_mapping_xrf_"
                    f"{x_method_base}_{created_date}_{created_time}.csv"
                )

                col_dl3, col_dl4 = st.columns([3, 1])
                with col_dl3:
                    st.download_button(
                        label="Download CSV File with Optical Coordinates",
                        data=csv_content_converted,
                        file_name=csv_filename_converted,
                        mime="text/csv",
                        type="primary",
                        key="download_csv_converted"
                    )
