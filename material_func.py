import streamlit as st
from datetime import datetime

from substrate_func import validate_operator_name, format_date


# ==================== MATERIAL FUNCTIONS ====================

# Material types used across deposition methods in the lab
MATERIAL_TYPES = [
    "Target",
    "Powder",
    "Gas",
    "Liquid",
    "Foil",
    "Wire",
    "Pellet",
    "Single Crystal",
]

# Common purity grades as encountered in thin-film research
PURITY_OPTIONS = [
    "2N  (99%)",
    "3N  (99.9%)",
    "4N  (99.99%)",
    "4N5 (99.995%)",
    "5N  (99.999%)",
    "6N  (99.9999%)",
    "Other",
]

# Physical forms relevant to PLD targets and other sources
PHYSICAL_FORMS = [
    "Disc",
    "Cylinder",
    "Rectangular",
    "Powder",
    "Granule",
    "Gas cylinder",
    "Liquid bottle",
    "Foil",
    "Wire",
    "Other",
]

# Hazard classification (simplified GHS)
HAZARD_CLASSES = [
    "None",
    "Flammable",
    "Oxidising",
    "Toxic",
    "Corrosive",
    "Irritant",
    "Environmental hazard",
    "Multiple",
]


def build_material_id(element, purity_grade, acquisition_date, identifier):
    """
    Construct a material ID following the lab convention:
    <element>_<date>_<purity>_<identifier>
    Example: Cu_20260101_4N_alpha
    """
    parts = [p for p in [element.strip(), acquisition_date, purity_grade.strip(), identifier.strip()] if p]
    return "_".join(parts)


def generate_material_filename(material_id, institution, operator):
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{material_id}_{institution}_{operator}_material_{current_datetime}.csv"


def generate_material_csv_content(data):
    return f"""material_id,{data['material_id']}
mat_name,{data['material_name']}
mat_formula,{data['formula']}
mat_type,{data['material_type']}
mat_physical_form,{data['physical_form']}
mat_purity,{data['purity']}
mat_lot_number,{data['lot_number']}
mat_cas_number,{data['cas_number']}
mat_date_of_acquisition,{data['date_of_acquisition']}
mat_origin,{data['origin']}
mat_initial_quantity,{data['initial_quantity']}
mat_quantity_unit,{data['quantity_unit']}
mat_dimensions,{data['dimensions']}
mat_storage_location,{data['storage_location']}
mat_expiry_date,{data['expiry_date']}
mat_hazard_class,{data['hazard_class']}
mat_notes,{data['notes']}
mat_operator,{data['operator']}
mat_institution,{data['institution']}
mat_registration_date,"{data['registration_date']}"
mat_registration_time,"{data['registration_time']}"
"""


# ==================== MATERIAL TAB UI ====================

def render_material_tab():
    st.header("Material Registry CSV Generator")
    st.info(
        "Register a new material (target, powder, gas, etc.) in the database. "
        "The Material ID is auto-generated from the element, acquisition date, purity grade, and optional identifier."
    )

    # ---- Material ID builder ----
    st.subheader("Material ID")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        mat_element = st.text_input(
            "Element / Compound abbreviation",
            value="",
            placeholder="e.g. Cu, Cu2BaSe2",
            key="mat_element"
        )
    with col2:
        mat_acq_date_raw = st.date_input(
            "Date of acquisition",
            value=datetime.now().date(),
            key="mat_acq_date"
        )
        mat_acq_date_str = mat_acq_date_raw.strftime("%Y%m%d")
    with col3:
        mat_purity_grade = st.selectbox(
            "Purity grade (for ID)",
            ["4N", "4N5", "5N", "6N", "3N", "2N", "Other"],
            key="mat_purity_grade"
        )
    with col4:
        mat_identifier = st.text_input(
            "Identifier (for duplicates)",
            value="",
            placeholder="e.g. alpha, beta, 1",
            key="mat_identifier"
        )

    material_id = build_material_id(mat_element, mat_purity_grade, mat_acq_date_str, mat_identifier)
    st.markdown(f"**Generated Material ID:** `{material_id}`")

    st.divider()

    # ---- Material properties ----
    st.subheader("Material Properties")
    col1, col2, col3 = st.columns(3)

    with col1:
        mat_name = st.text_input(
            "Material Name",
            value="",
            placeholder="e.g. Copper, Copper selenide",
            key="mat_name"
        )
        mat_formula = st.text_input(
            "Chemical Formula",
            value="",
            placeholder="e.g. Cu, Cu2Se",
            key="mat_formula"
        )
        mat_type = st.selectbox(
            "Material Type",
            MATERIAL_TYPES,
            key="mat_type"
        )
        mat_physical_form = st.selectbox(
            "Physical Form",
            PHYSICAL_FORMS,
            key="mat_physical_form"
        )

    with col2:
        mat_purity_full = st.selectbox(
            "Purity",
            PURITY_OPTIONS,
            index=2,
            key="mat_purity_full"
        )
        mat_cas = st.text_input(
            "CAS Number",
            value="",
            placeholder="e.g. 7440-50-8",
            key="mat_cas",
            help="Chemical Abstracts Service registry number"
        )
        mat_lot = st.text_input(
            "Lot / Batch Number",
            value="",
            placeholder="e.g. L2024110501",
            key="mat_lot"
        )
        mat_dimensions = st.text_input(
            "Dimensions",
            value="",
            placeholder="e.g. 50 mm dia x 6 mm thick",
            key="mat_dimensions",
            help="Relevant for targets: diameter x thickness; for foils: width x length x thickness"
        )

    with col3:
        mat_quantity = st.text_input(
            "Initial Quantity",
            value="",
            placeholder="e.g. 500",
            key="mat_quantity"
        )
        mat_unit = st.selectbox(
            "Quantity Unit",
            ["g", "kg", "mg", "pieces", "L", "mL", "m", "cm"],
            key="mat_unit"
        )
        mat_storage = st.text_input(
            "Storage Location",
            value="",
            placeholder="e.g. Cabinet A, Shelf 2",
            key="mat_storage"
        )
        mat_expiry = st.text_input(
            "Expiry Date",
            value="",
            placeholder="e.g. 2028-01-01 or N/A",
            key="mat_expiry"
        )

    st.divider()

    # ---- Supplier information ----
    st.subheader("Supplier Information")
    col1, col2 = st.columns(2)

    with col1:
        mat_origin = st.text_input(
            "Origin / Supplier",
            value="",
            placeholder="e.g. Testbourne Ltd, Sigma-Aldrich",
            key="mat_origin"
        )

    with col2:
        mat_hazard = st.selectbox(
            "Hazard Class",
            HAZARD_CLASSES,
            key="mat_hazard"
        )

    mat_notes = st.text_area(
        "Notes",
        value="",
        placeholder="Any additional information: visual inspection, pre-use conditioning, known issues, etc.",
        key="mat_notes"
    )

    st.divider()

    # ---- Registration metadata ----
    st.subheader("Registration")
    col1, col2, col3 = st.columns(3)

    with col1:
        mat_institution = st.text_input(
            "Institution",
            value=st.session_state.get('mat_institution', "HZB"),
            key="mat_institution"
        )
    with col2:
        mat_operator = st.text_input(
            "Operator (First and Last Name)",
            value=st.session_state.get('mat_operator', "Steinkopf Lars"),
            key="mat_operator"
        )
    with col3:
        mat_reg_date = st.date_input(
            "Registration Date",
            value=datetime.now().date(),
            key="mat_reg_date"
        )
        mat_reg_time = st.text_input(
            "Registration Time",
            value="14:01:15",
            help="Format: HH:MM:SS (24-hour)",
            key="mat_reg_time"
        )

    st.divider()

    if st.button("Generate Material CSV File", type="primary"):
        errors = []

        if not mat_element.strip():
            errors.append("Element / Compound abbreviation is required to build the Material ID.")
        if not mat_name.strip():
            errors.append("Material Name is required.")
        if not mat_institution.strip():
            errors.append("Institution is required.")
        if not mat_operator.strip():
            errors.append("Operator is required.")
        elif not validate_operator_name(mat_operator):
            errors.append("Operator must include both First Name and Last Name (e.g., Steinkopf Lars).")

        from substrate_func import convert_time_to_12hour
        time_formatted = convert_time_to_12hour(mat_reg_time)
        if not time_formatted:
            errors.append("Invalid registration time format. Please use HH:MM:SS.")

        if errors:
            for msg in errors:
                st.error(msg)
        else:
            date_formatted = format_date(mat_reg_date)

            data = {
                'material_id': material_id,
                'material_name': mat_name,
                'formula': mat_formula,
                'material_type': mat_type,
                'physical_form': mat_physical_form,
                'purity': mat_purity_full,
                'lot_number': mat_lot,
                'cas_number': mat_cas,
                'date_of_acquisition': mat_acq_date_raw.strftime("%Y-%m-%d"),
                'origin': mat_origin,
                'initial_quantity': mat_quantity,
                'quantity_unit': mat_unit,
                'dimensions': mat_dimensions,
                'storage_location': mat_storage,
                'expiry_date': mat_expiry if mat_expiry.strip() else "N/A",
                'hazard_class': mat_hazard,
                'notes': mat_notes.replace('\n', ' '),
                'operator': mat_operator,
                'institution': mat_institution,
                'registration_date': date_formatted,
                'registration_time': time_formatted,
            }

            csv_content = generate_material_csv_content(data)
            filename = generate_material_filename(material_id, mat_institution, mat_operator)

            st.success("Material CSV file generated successfully")
            st.download_button(
                label="Download CSV File",
                data=csv_content,
                file_name=filename,
                mime="text/csv"
            )
            with st.expander("Preview CSV Content"):
                st.text(csv_content)
