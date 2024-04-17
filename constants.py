from enum import Enum

# TO ADD A NEW TEMPLATE:
# 1) Define a name for the file type: add an entry to the FTYPE enum
# 2) Define the extension type: add an entry to the EXTENSION enum. This is
#    only used to have a list of extensions, and does is not used to map an
#    extension to a filetype.
# 3) Create a mapping of the data table headers to standardized headers: add a
#    dictionary entry to the LABELS_TEMPLATE enum
# 4) Create a unit conversion table to standardize units: add a disctionary
#    entry to the UNITS_TEMPLATE enum
# 5) Add a template with aggregated information: to the TEMPLATES dictionary,
#    add a dictionary entry containing data from LABELS_TEMPLATE and
#    UNITS_TEMPLATE defined in steps 3 and 4.


class MLABEL(Enum):
    """Metadata label names."""
    CELL_NAME = "cell_name"
    MOLAR_MASS = "molar_mass"
    IONS_PFU = "ions_pfu"
    CHAR_MASS = "char_mass"
    CHAR_VOL = "char_vol"
    ACTIVE_RATIO = "active_ratio"
    CARBON_RATIO = "carbon_ratio"
    BINDER_RATO = "binder_ratio"
    CATHODE_MAT = "cathode_mat"
    CATHODE_DIA = "cathode_dia"
    CATHODE_SA = "cathode_SA"
    CATHODE_GEOMETRY = "cathode_geometry"
    CATHODE_LOADING = "cathode_loading"
    ANODE_MAT = "anode_mat"
    ANODE_DIA = "anode_mat"
    ANODE_SA = "anode_SA"
    ANODE_GEOMETRY = "anode_geometry"
    BINDER = "binder"
    ION = "ion"
    ELY = "electrolyte"
    TEMP = "temperature"
    CELL_TYPE = "cell_type"
    EXPERIMENT = "experiment"
    THEORETICAL_CAP = "theoretical_capacity"


class LABEL(Enum):
    """Full data label names."""
    DATE_TIME = "date time"
    STEP_TIME = "step time"
    CONTROL_CURR = "control I (mA)"
    CURRENT = "current (A)"
    VOLTAGE = "voltage (V)"
    TEST_TIME = "test time (s)"
    CYCLE_INDEX = "cycle index"
    STEP_INDEX = "step index"
    STEP_TYPE = "step type"
    CAP = "capacity (mAh)"
    CAP_C = "charge_capacity (mAh)"
    CAP_D = "discharge capacity (mAh)"
    ENERGY_C = "charge_energy (Wh)"
    ENERGY_D = "discharge_energy (Wh)"
    GRAV_CAP_C = "gravimetric_charge_capacity (mAh/g)"
    GRAV_CAP_D = "gravimetric_discharge_capacity (mAh/g)"
    VOL_CAP_C = "volumetric_charge_capacity (mAh/L)"
    VOL_CAP_D = "volumentric_discharge capacity (mAh/L)"
    GRAV_ENERGY_C = "gravimetric_charge_energy (Wh/kg)"
    GRAV_ENERGY_D = "gravimetric_discharge_energy (Wh/kg)"
    VOL_ENERGY_C = "volumetric_charge_energy (Wh/L)"
    VOL_ENERGY_D = "volumentric_discharge energy (Wh/L)"
    DQ_DV = "dQ/dV (mAh/V)"
    GRAV_DQ_DV = "gravimetric_dQ/dV (mAh/V/mg)"
    VOL_DQ_DV = "volumetric_dQ/dV (mAh/V/L)"
    DQ = "dQ (mAh)"
    DV = "dV (V)"
    RATE = "rate (mA/g)"
    C_RATE = "C rate"


class CLABEL(Enum):
    """Cycle summary label names."""
    CYCLE_INDEX = "cycle index"
    CAP_C = 'charge capacity (mAh)'
    CAP_D = 'discharge capacity (mAh)'
    GRAV_CAP_C = 'charge capacity (mAh/g)'
    GRAV_CAP_D = 'discharge capacity (mAh/g)'
    AVG_V_C = 'avg. charge voltage (V)'
    AVG_V_D = 'avg. discharge voltage (V)'
    ENERGY_C = 'charge energy (Wh)'
    ENERGY_D = 'discharge energy (Wh)'
    GRAV_ENERGY_C = 'charge energy density (Wh/kg)'
    GRAV_ENERGY_D = 'discharge energy density (Wh/kg)'
    CE = 'coulombic effiency (%)'
    V_HYS = 'voltage hysteresis (V)'
    ENERGY_EFF = 'energy efficiency (%)'
    CHARGE_STATUS = 'charge status'
    DISCHARGE_STATUS = 'discharge status'
    CHARGE_RATE = "charge rate (mA/g)"
    DISCHARGE_RATE = "discharge rate (mA/g)"
    CHARGE_C_RATE = "charge C-rate"
    DISCHARGE_C_RATE = "discharge C-rate"


class FTYPE(Enum):
    BIOLOGIC = "biologic"
    ARBIN = "arbin"
    UNIV_CSV = "universal_csv"
    NEWARE = "neware"


class EXTENSION(Enum):
    BIOLOGIC = ".mpt"
    ARBIN = ".res"
    CSV = ".csv"
    NEWARE = ".ndax"


class LABELS_TEMPLATE(Enum):
    UNIV_CSV = {
    }
    BIOLOGIC = {
        'time/s': LABEL.TEST_TIME.value,
        'Ewe/V': LABEL.VOLTAGE.value,
        'I/mA': LABEL.CURRENT.value,
        'dq/mA.h': LABEL.DQ.value,
        'Energy charge/W.h': LABEL.ENERGY_C.value,
        'Energy discharge/W.h': LABEL.ENERGY_D.value,
        'Q charge/discharge/mA.h': LABEL.CAP.value,
        'Q charge/mA.h': LABEL.CAP_C.value,
        'Q discharge/mA.h': LABEL.CAP_D.value,
        'cycle number': LABEL.CYCLE_INDEX.value,
        'control/mA': LABEL.CONTROL_CURR.value,
    }
    ARBIN = {
        'Test_Time': LABEL.TEST_TIME.value,
        'Step_Time': LABEL.STEP_TIME.value,
        'DateTime': LABEL.DATE_TIME.value,
        'Voltage': LABEL.VOLTAGE.value,
        'Current': LABEL.CURRENT.value,
        'Charge_Energy': LABEL.ENERGY_C.value,
        'Discharge_Energy': LABEL.ENERGY_D.value,
        'Charge_Capacity': LABEL.CAP_C.value,
        'Discharge_Capacity': LABEL.CAP_D.value,
        'Cycle_Index': LABEL.CYCLE_INDEX.value,
        'Step_Index': LABEL.STEP_INDEX.value,
    }
    NEWARE = {
        # 'Total Time': LABEL.TEST_TIME.value,
        'Time': LABEL.STEP_TIME.value,
        'Timestamp': LABEL.DATE_TIME.value,
        'Voltage': LABEL.VOLTAGE.value,
        'Current(mA)': LABEL.CURRENT.value,
        # 'Capacity(Ah)': LABEL.CAP.value,
        'Charge_Energy(mWh)': LABEL.ENERGY_C.value,
        'Discharge_Energy(mWh)': LABEL.ENERGY_D.value,
        'Charge_Capacity(mAh)': LABEL.CAP_C.value,
        'Discharge_Capacity(mAh)': LABEL.CAP_D.value,
        'Cycle': LABEL.CYCLE_INDEX.value,
        'Step': LABEL.STEP_INDEX.value,
        'Status': LABEL.STEP_TYPE.value,
    }

    # For Neware xlsx files
    # NEWARE = {
    #     'Total Time': LABEL.TEST_TIME.value,
    #     'Time': LABEL.STEP_TIME.value,
    #     'Date': LABEL.DATE_TIME.value,
    #     'Voltage(V)': LABEL.VOLTAGE.value,
    #     'Current(A)': LABEL.CURRENT.value,
    #     'Capacity(Ah)': LABEL.CAP.value,
    #     'Chg. Energy(Wh)': LABEL.ENERGY_C.value,
    #     'DChg. Energy(Wh)': LABEL.ENERGY_D.value,
    #     'Chg. Cap.(Ah)': LABEL.CAP_C.value,
    #     'DChg. Cap.(Ah)': LABEL.CAP_D.value,
    #     'Cycle Index': LABEL.CYCLE_INDEX.value,
    #     'Step Index': LABEL.STEP_INDEX.value,
    #     'Step Type': LABEL.STEP_TYPE.value,
    # }


class UNITS_TEMPLATE(Enum):
    # Standard units
    # Time: sec
    # Voltage: V
    # Current (I): A
    # dQ: mA
    # Capacity: mAh
    # Energy: W-h
    UNIV_CSV = {
        LABEL.TEST_TIME.value: 1,
        LABEL.VOLTAGE.value: 1,
        LABEL.CURRENT.value: 1,
        LABEL.DQ.value: 1,
        LABEL.ENERGY_C.value: 1,
        LABEL.ENERGY_D.value: 1,
        LABEL.CAP.value: 1,
        LABEL.CAP_C.value: 1,
        LABEL.CAP_D.value: 1,
        LABEL.CYCLE_INDEX.value: 1
    }
    BIOLOGIC = {
        LABEL.TEST_TIME.value: 1,
        LABEL.VOLTAGE.value: 1,
        LABEL.CURRENT.value: 0.001,
        LABEL.DQ.value: 1,
        LABEL.ENERGY_C.value: 1,
        LABEL.ENERGY_D.value: 1,
        LABEL.CAP.value: 1,
        LABEL.CAP_C.value: 1,
        LABEL.CAP_D.value: 1,
        LABEL.CYCLE_INDEX.value: 1
    }
    ARBIN = {
        LABEL.TEST_TIME.value: 1,
        LABEL.VOLTAGE.value: 1,
        LABEL.CURRENT.value: 1,
        LABEL.DQ.value: 1,
        LABEL.ENERGY_C.value: 1,
        LABEL.ENERGY_D.value: 1,
        LABEL.CAP.value: 1000,
        LABEL.CAP_C.value: 1000,
        LABEL.CAP_D.value: 1000,
        LABEL.CYCLE_INDEX.value: 1
    }
    NEWARE = {
        LABEL.TEST_TIME.value: 1,
        LABEL.VOLTAGE.value: 1,
        LABEL.CURRENT.value: 1,
        LABEL.DQ.value: 1,
        LABEL.ENERGY_C.value: 1,
        LABEL.ENERGY_D.value: 1,
        LABEL.CAP.value: 1000,
        LABEL.CAP_C.value: 1000,
        LABEL.CAP_D.value: 1000,
        LABEL.CYCLE_INDEX.value: 1
    }


# TO ADD A NEW TEMPLATE:
# 1) Define a name for the file type: add an entry to the FTYPE enum
# 2) Define the extension type: add an entry to the EXTENSION enum. This is
#    only used to have a list of extensions, and does is not used to map an
#    extension to a filetype.
# 3) Create a mapping of the data table headers to standardized headers: add a
#    dictionary entry to the LABELS_TEMPLATE enum.
# 4) Create a unit conversion table to standardize units: add a dictionary
#    entry to the UNITS_TEMPLATE enum
# 5) Add a template with aggregated information: to the TEMPLATES dictionary,
#    add a dictionary entry containing data from LABELS_TEMPLATE and
#    UNITS_TEMPLATE defined in steps 3 and 4.
TEMPLATES = {
    FTYPE.BIOLOGIC.value: {
        'labels': LABELS_TEMPLATE.BIOLOGIC.value,
        'unit_conversion': UNITS_TEMPLATE.BIOLOGIC.value,
    },
    FTYPE.ARBIN.value: {
        'labels': LABELS_TEMPLATE.ARBIN.value,
        'unit_conversion': UNITS_TEMPLATE.ARBIN.value,
    },
    FTYPE.NEWARE.value: {
        'labels': LABELS_TEMPLATE.NEWARE.value,
        'unit_conversion': UNITS_TEMPLATE.NEWARE.value,
    },
    FTYPE.UNIV_CSV.value: {
        'labels': LABELS_TEMPLATE.UNIV_CSV.value,
        'unit_conversion': UNITS_TEMPLATE.UNIV_CSV.value,
    }
}

METADATA_UNITS = {
    MLABEL.CELL_NAME.value: None,
    MLABEL.MOLAR_MASS.value: 'g/mol',
    MLABEL.IONS_PFU.value: None,
    MLABEL.CHAR_MASS.value: 'mg',
    MLABEL.CHAR_VOL.value: 'L',
    MLABEL.ACTIVE_RATIO.value: None,
    MLABEL.CARBON_RATIO.value: None,
    MLABEL.BINDER_RATO.value: None,
    MLABEL.CATHODE_MAT.value: None,
    MLABEL.CATHODE_DIA.value: 'mm',
    MLABEL.CATHODE_SA.value: None,
    MLABEL.CATHODE_GEOMETRY.value: None,
    MLABEL.ANODE_MAT.value: None,
    MLABEL.ANODE_DIA.value: 'cm',
    MLABEL.ANODE_SA.value: 'cm^2',
    MLABEL.ANODE_GEOMETRY.value: None,
    MLABEL.BINDER.value: None,
    MLABEL.ION.value: None,
    MLABEL.ELY.value: None,
    MLABEL.TEMP.value: 'C',
    MLABEL.CELL_TYPE.value: None,
    MLABEL.EXPERIMENT.value: None,
    MLABEL.THEORETICAL_CAP.value: 'mAh/g',
}

MLABEL = MLABEL
