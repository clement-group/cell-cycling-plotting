import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.axes as axes
import seaborn as sns
import os
import re
import copy
import NewareNDA
import electrochem as echem
from formatting import format_plot
from pathlib import Path
from constants import MLABEL, CLABEL, LABEL, METADATA_UNITS, FTYPE, EXTENSION, LABELS_TEMPLATE, UNITS_TEMPLATE, TEMPLATES

data_dir = '/Users/tylerpennebaker/Library/CloudStorage/Box-Box/Elias-Raphaële shared folder/LGES project/WP6/Cell cycling/data/'
save_dir = '/Users/tylerpennebaker/Library/CloudStorage/Box-Box/Elias-Raphaële shared folder/LGES project/WP6/Cell cycling/data/'
fig_save_dir = '/Users/tylerpennebaker/Library/CloudStorage/Box-Box/Elias-Raphaële shared folder/LGES project/WP6/Cell cycling/figures/'
summary_save_dir = '/Users/tylerpennebaker/Library/CloudStorage/Box-Box/Elias-Raphaële shared folder/LGES project/WP6/Cell cycling/data/'

def get_active_mass(cell_name):
    """
    DESCRIPTION: Given a cell name, loads the shared echem excel sheet and finds the entered active mass (in g) of the cell.
    PARAMETERS:
        cell_name: string
            string of the cell name (e.g. LM9-LPR-E1-1)
    RETURNS: float
        Returns a float object of the cell active mass
    """
    active_mass = 1
    df = pd.read_excel('/Users/tylerpennebaker/Library/CloudStorage/Box-Box/Elias-Raphaële shared folder/LGES project/WP6/Cell cycling/cell_cycling_new/cyclingdatalog.xlsx')
    sample_info = df[df['Sample'] == cell_name]
    if sample_info.empty:
        # Handle the case when there are no rows matching the condition
        print(f"No information found for {cell_name}")
        #exit()
    else:
        active_mass = (sample_info['active mass'].values[0])
        active_mass = active_mass/1000
        # Proceed with further processing or use of active_mass
    return(active_mass)

def get_molar_mass(cell_name):
    """
    DESCRIPTION: Given a cell name, loads the shared echem excel sheet and finds the entered molar mass (in g) of the cell.
    PARAMETERS:
        cell_name: string
            string of the cell name (e.g. LM9-LPR-E1-1)
    RETURNS: float
        Returns a float object of the cell molar mass
    """
    molar_mass = 1
    df = pd.read_excel('/Users/erick/Library/CloudStorage/Box-Box/Electrochem/shared echem notebook.xlsx')
    sample_info = df[df['Sample'] == cell_name]
    if sample_info.empty:
        # Handle the case when there are no rows matching the condition
        print(f"No information found for {cell_name}")
        #exit()
    else:
        molar_mass = (sample_info['Molar Mass'].values[0])
        #molar_mass = molar_mass/1000
        # Proceed with further processing or use of molar_mass
    return(molar_mass)

def get_voltage_window(df: pd.DataFrame):
    """
    DESCRIPTION: Given a standardized dataframe, get the upper and lower voltage cutoffs of the first cycle,
                rounded to 2 decimal points.
    PARAMETERS:
        df: a pandas Dataframe object
            a dataframe containing standardized battery cycling data
    RETURNS: tuple of floats, (lower_cutoff, upper_cutoff)
        the upper and lower voltage cutoffs of the first cycle, (upper_cutoff, lower_cutoff)
    """
    first_cycle_df = df[df[LABEL.CYCLE_INDEX.value] == 1]
    charge_df = first_cycle_df[first_cycle_df[LABEL.STEP_TYPE.value] == 'charge']
    discharge_df = first_cycle_df[first_cycle_df[LABEL.STEP_TYPE.value] == 'discharge']
    if charge_df.empty or discharge_df.empty:
        return (None, None)
    upper_cutoff = np.round(charge_df[LABEL.V.value].iloc[-1],2)
    lower_cutoff = np.round(discharge_df[LABEL.V.value].iloc[-1], 2)
    # print('voltage window: {}'.format((lower_cutoff, upper_cutoff)))
    return (lower_cutoff, upper_cutoff)

def get_combined_capacity(df: pd.DataFrame):
    """
    DESCRIPTION: Given a data row of battery cycling data, determine the value of the combined capacity column
    PARAMETERS:
        series: a pandas Series object
            row containing battery cycling data
    RETURNS: float
        the value of the combined capacity column, to be used to calculate a new column
    """
    if df[LABEL.STEP_TYPE.value] == 'charge':
        return df[LABEL.CAP_C.value]
    elif df[LABEL.STEP_TYPE.value] == 'discharge':
        return df[LABEL.CAP_D.value]*-1
    elif df[LABEL.STEP_TYPE.value] == 'unsure':
        return df[LABEL.CAP_C.value]

def get_step_type(series):
    """
    DESCRIPTION: Given a data row of battery cycling data, determine the step type of the row.
    PARAMETERS:
        series: a pandas Series object
            row containing battery cycling data
    RETURNS: string
        A string describing the step type of the row
    """
    if LABEL.CONTROL_CURR.value in series.index:
        current = series[LABEL.CONTROL_CURR.value]
    else:
        current = series[LABEL.I.value]
    if LABEL.CAP.value in series.index:
        capacity = series[LABEL.CAP.value]
        if current == 0:
            return 'rest'
        elif current > 0 and capacity > 0:
            return 'charge'
        elif current < 0 and capacity < 0:
            return 'discharge'
        else:
            return 'unsure'
    else:
        if current == 0:
            return 'rest'
        elif current > 0:
            return 'charge'
        elif current < 0:
            return 'discharge'

def get_cycle_status(df: pd.DataFrame, voltage_window):
    """
    DESCRIPTION: Given the charge or discharge step of a single cycle as a dataframe, determine whether that charge
                or discharge step has completed based on the voltage window
    PARAMETERS:
        df: a pandas Dataframe object
            a subset dataframe of standardized battery cycling data containing charge or discharge data for a single cycle
        voltage window: a tuple of floats (lower_cutoff, upper_cutoff)
            A tuple containing the voltage window of the cycling experiment
    RETURNS: str
        A string of either 'not started', 'complete', or 'incomplete', based on the status of the cycle step
    """
    lower_cutoff, upper_cutoff = voltage_window
    buffer = 0.01
    if df.empty:
        return 'not started'
    elif df[LABEL.STEP_TYPE.value].iloc[0] == 'charge':
        max_V = df[LABEL.V.value].max()
        if max_V >= upper_cutoff - buffer:
            return 'complete'
        else:
            return 'incomplete'
    elif df[LABEL.STEP_TYPE.value].iloc[0] == 'discharge':
        min_V = df[LABEL.V.value].min()
        if min_V <= lower_cutoff + buffer:
            return 'complete'
        else:
            return 'incomplete'

def standardize(df: pd.DataFrame, filetype, metadata_df: pd.DataFrame=None):
    """
    DESCRIPTION: Given a parsed dataframe containing battery cycling data, reads, processes, and standardizes the data to a universal format.
    PARAMETERS:
        df: pandas Dataframe object
            dataframe containing cycling data
        filetype: str, default None
            The type of data file data originated from. Must match a defined filetype ('arbin', 'biologic', 'universal_csv')
    RETURNS: pandas Dataframe
        Returns a dataframe object containing data in a standardized format (i.e. column names/ units standardized, certain
        columns added if necessary)
    """
    # standardize column labels
    df.rename(columns=TEMPLATES[filetype]['labels'], inplace=True)
    # standardize units
    for key in TEMPLATES[filetype]['unit_conversion']:
        if key in df.columns:
            df[key] *= TEMPLATES[filetype]['unit_conversion'][key]
    # set cycle index column to 0-indexing
    df[LABEL.CYCLE_INDEX.value] -= df[LABEL.CYCLE_INDEX.value].iloc[0]
    # generate column with step type
    df[LABEL.STEP_TYPE.value] = df.apply(lambda x: get_step_type(x), axis=1)
    # generate combined capacity column if necessary
    if not LABEL.CAP.value in df.columns:
        df[LABEL.CAP.value] = df.apply(lambda x: get_combined_capacity(x), axis=1)
    # calculate dV
    df[LABEL.DV.value] = df[LABEL.V.value].diff()
    df[LABEL.DV.value].replace(0, np.nan, inplace= True)
    # calculate dQ
    if LABEL.DQ.value not in df.columns:
        df[LABEL.DQ.value] = df[LABEL.CAP.value].diff()
    # calculate dQ/dV
    df[LABEL.DQ_DV.value] = df[LABEL.DQ.value]/df[LABEL.DV.value]
    # calculate values that are "specific"
    if isinstance(metadata_df, pd.DataFrame) and MLABEL.CHAR_MASS.value in metadata_df.index:
        char_mass = metadata_df.loc[MLABEL.CHAR_MASS.value, 'value']
        # print('mass: ', char_mass)
        if char_mass:
            # mAh/g
            df[LABEL.GRAV_CAP_C.value] = df[LABEL.CAP_C.value]/char_mass
            df[LABEL.GRAV_CAP_D.value] = df[LABEL.CAP_D.value]/char_mass
            # Wh/kg
            df[LABEL.GRAV_ENERGY_C.value] = df[LABEL.ENERGY_C.value]/char_mass*1e6
            df[LABEL.GRAV_ENERGY_D.value] = df[LABEL.ENERGY_D.value]/char_mass*1e6
            # dQ/dV, mAh/g/V
            df[LABEL.GRAV_DQ_DV.value] = df[LABEL.DQ_DV.value]/char_mass*1000
            # rate, mA/g
            df[LABEL.RATE.value] = np.abs(df[LABEL.I.value])*1000/(char_mass/1000)
            if MLABEL.THEORETICAL_CAP.value in metadata_df.index:
                theor_cap = metadata_df.loc[MLABEL.THEORETICAL_CAP.value, 'value']
                if theor_cap:
                    df[LABEL.C_RATE.value] = 1e6*np.abs(df[LABEL.I.value])/theor_cap/char_mass
    if isinstance(metadata_df, pd.DataFrame) and MLABEL.CHAR_VOL.value in metadata_df.index:
        char_vol = metadata_df.loc[MLABEL.CHAR_VOL.value, 'value']
        # mAh/L
        df[LABEL.VOL_CAP_C.value] = df[LABEL.CAP_C.value]/char_vol
        df[LABEL.VOL_CAP_D.value] = df[LABEL.CAP_D.value]/char_vol
        # Wh/L
        df[LABEL.VOL_ENERGY_C.value] = df[LABEL.ENERGY_C.value]/char_vol
        df[LABEL.VOL_ENERGY_D.value] = df[LABEL.ENERGY_D.value]/char_vol
        # dQ/dV, mAh/L/V
        df[LABEL.VOL_DQ_DV.value] = df[LABEL.DQ_DV.value]/char_vol
    return df

def get_csv(cell_name, filetype=None, metadata=None, save_csv=None):

    csv_path = save_dir+cell_name+'.csv'


    # determine filetype and define filepath
    extensions = ['.mpt','.res','.ndax','.csv']
    for extension in extensions:
        if os.path.exists(data_dir+cell_name+extension):
            filepath = data_dir+cell_name+extension

    filepath = data_dir+cell_name+'.mpt'
    print(filepath)


    if Path(filepath).suffix == '.mpt':
        filetype = FTYPE.BIOLOGIC.value
    elif Path(filepath).suffix == '.res':
        filetype = FTYPE.ARBIN.value
    elif Path(filepath).suffix == '.ndax':
        filetype = FTYPE.NEWARE.value
    elif Path(filepath).suffix == '.csv':
        filetype = FTYPE.UNIV_CSV.value
    else:
        raise ValueError('{} does not have a recognized filetype. List of valid filetypes: {}'.\
            format(Path(filepath).name, [x.value for x in EXTENSION]))
#    if isinstance(metadata, pd.DataFrame):
#        metadata_df = metadata
#    elif metadata:
#        metadata_df = create_metadata_df(metadata, metadata_savepath)
#    else:
#        metadata_df = None

    # create metadata_df
    char_mass = [get_active_mass(cell_name)]
    metadata_df = pd.DataFrame({MLABEL.CHAR_MASS.value:char_mass})
    metadata_df = metadata_df.transpose()
    metadata_df.columns = ['value']
    # Case for Biologic .mpt text file
    if filetype == FTYPE.BIOLOGIC.value:
        # do a first parse to find where the data starts
        start_line = None
        lines = open(filepath, 'r', encoding='ISO-8859-1').readlines()
        for i, line in enumerate(lines):
            # data starts in the line that contains the phrase "charge/discharge"
            if re.search('charge/discharge', line):
                start_line = i
        df = pd.read_csv(filepath, skiprows=start_line, header=0, sep="\t", encoding = 'ISO-8859-1')
        df = df.rename(columns={'<I>/mA': 'I/mA'})
        df = df.astype({'cycle number': int})
        # print(df.columns)
        standardize(df, filetype, metadata_df=metadata_df)
    # Case for Arbin .res file
    elif filetype == FTYPE.ARBIN.value:
        # This won't work becasue i don't know what "echem" is (from vincent somehow)
        df = echem.parseArbin(filepath)
        standardize(df, filetype, metadata_df=metadata_df)
    # case for Neware .ndax file
    elif filetype == FTYPE.NEWARE.value:
        df = NewareNDA.read(filepath)
        df.to_csv(save_dir+'test2.csv', index=False)
        standardize(df, filetype, metadata_df=metadata_df)
    # case for universal .csv file
    elif filetype == FTYPE.UNIV_CSV.value:
        df = pd.read_csv(filepath)
        standardize(df, filetype, metadata_df=metadata_df)

    df.to_csv(csv_path, index=False)

    return(df)

def get_cell_summary(cell_name, c_savepath=None):
    """
    DESCRIPTION: Given a cell name, calculates summary data for each cycle
    PARAMETERS:
        cell_name: string, name of a cell
            string containing the name of a cell
        savepath: str, path object, file-like object, default None
            Path to save cycle summary csv file to. If None, no csv file is saved.
    RETURNS: pandas Dataframe object, c_df
        Returns a dataframe object containing summarized cycle data
    """

    df = get_csv(cell_name)
    num_cycles = df[LABEL.CYCLE_INDEX.value].iloc[-1]-df[LABEL.CYCLE_INDEX.value].iloc[0]+1
    lower_cutoff, upper_cutoff = get_voltage_window(df)
    cycle_number = []
    coulombic_efficiency = []
    charge_capacity = []
    discharge_capacity = []
    charge_energy = []
    discharge_energy = []
    avg_charge_voltage = []
    avg_discharge_voltage = []
    voltage_hysteresis = []
    energy_efficiency = []
    charge_cycle_status = []
    discharge_cycle_status = []
    grav_charge_capacity = []
    grav_discharge_capacity = []
    grav_charge_energy = []
    grav_discharge_energy = []
    charge_rate = []
    discharge_rate = []
    charge_C_rate = []
    discharge_C_rate = []
    for i in range(num_cycles):
        cycle_number.append(i)
        cycle_df = df[df[LABEL.CYCLE_INDEX.value] == i]
        cycle_c_df = cycle_df[cycle_df[LABEL.STEP_TYPE.value] == 'charge']
        cycle_d_df = cycle_df[cycle_df[LABEL.STEP_TYPE.value] == 'discharge']
        # get status of cycle, if lower and upper voltages are defined
        if lower_cutoff and upper_cutoff:
            cycle_status_c = get_cycle_status(cycle_c_df, (lower_cutoff, upper_cutoff))
            cycle_status_d = get_cycle_status(cycle_d_df, (lower_cutoff, upper_cutoff))
            charge_cycle_status.append(cycle_status_c)
            discharge_cycle_status.append(cycle_status_d)
        # charge step values
        if not cycle_c_df.empty:
            global last_c_row
            last_c_row = cycle_c_df.iloc[-1]
            charge_capacity.append(last_c_row[LABEL.CAP_C.value])
            charge_energy.append(last_c_row[LABEL.ENERGY_C.value])
            global voltage_c
            voltage_c = last_c_row[LABEL.ENERGY_C.value]/last_c_row[LABEL.CAP_C.value]*1000
            avg_charge_voltage.append(voltage_c)

        # discharge step values
        if not cycle_d_df.empty:
            global last_d_row
            last_d_row = cycle_d_df.iloc[-1]
            discharge_capacity.append(last_d_row[LABEL.CAP_D.value])
            discharge_energy.append(last_d_row[LABEL.ENERGY_D.value])
            global voltage_d
            voltage_d = last_d_row[LABEL.ENERGY_D.value]/last_d_row[LABEL.CAP_D.value]*1000
            avg_discharge_voltage.append(voltage_d)

        # specific values, if valid
        if LABEL.GRAV_CAP_C.value in df.columns:
            if not cycle_c_df.empty:
                grav_charge_capacity.append(last_c_row[LABEL.GRAV_CAP_C.value])
                grav_charge_energy.append(last_c_row[LABEL.GRAV_ENERGY_C.value])
            if not cycle_d_df.empty:
                grav_discharge_capacity.append(last_d_row[LABEL.GRAV_CAP_D.value])
                grav_discharge_energy.append(last_d_row[LABEL.GRAV_ENERGY_D.value])
        if LABEL.RATE.value in df.columns:
            if not cycle_c_df.empty:
                charge_rate.append(last_c_row[LABEL.RATE.value])
            if not cycle_d_df.empty:
                discharge_rate.append(last_d_row[LABEL.RATE.value])
        if LABEL.C_RATE.value in df.columns:
            if not cycle_c_df.empty:
                charge_C_rate.append(last_c_row[LABEL.C_RATE.value])
            if not cycle_d_df.empty:
                discharge_C_rate.append(last_d_row[LABEL.C_RATE.value])

        # charge + discharge values
        if lower_cutoff and upper_cutoff:
            if cycle_status_c == 'complete' and cycle_status_d == 'complete':
                coulombic_efficiency.append(last_d_row[LABEL.CAP_D.value]/last_c_row[LABEL.CAP_C.value]*100)
                voltage_hysteresis.append(voltage_c-voltage_d)
                energy_efficiency.append(last_d_row[LABEL.ENERGY_D.value]/last_c_row[LABEL.ENERGY_C.value]*100)
            else:
                coulombic_efficiency.append(last_d_row[LABEL.CAP_D.value]/last_c_row[LABEL.CAP_C.value]*100)
                voltage_hysteresis.append(voltage_c-voltage_d)
                energy_efficiency.append(last_d_row[LABEL.ENERGY_D.value]/last_c_row[LABEL.ENERGY_C.value]*100)
    # buffer end of list with 'None' to make all lists the same length
    target_len = len(cycle_number)
    coulombic_efficiency += [None]*(target_len-len(coulombic_efficiency))
    charge_capacity += [None]*(target_len-len(charge_capacity))
    discharge_capacity += [None]*(target_len-len(discharge_capacity))
    charge_energy += [None]*(target_len-len(charge_energy))
    discharge_energy += [None]*(target_len-len(discharge_energy))
    avg_charge_voltage += [None]*(target_len-len(avg_charge_voltage))
    avg_discharge_voltage += [None]*(target_len-len(avg_discharge_voltage))
    voltage_hysteresis += [None]*(target_len-len(voltage_hysteresis))
    energy_efficiency += [None]*(target_len-len(energy_efficiency))
    charge_cycle_status += [None]*(target_len-len(charge_cycle_status))
    discharge_cycle_status += [None]*(target_len-len(discharge_cycle_status))
    grav_charge_capacity += [None]*(target_len-len(grav_charge_capacity))
    grav_discharge_capacity += [None]*(target_len-len(grav_discharge_capacity))
    grav_charge_energy += [None]*(target_len-len(grav_charge_energy))
    grav_discharge_energy += [None]*(target_len-len(grav_discharge_energy))
    charge_rate += [None]*(target_len-len(charge_rate))
    discharge_rate += [None]*(target_len-len(discharge_rate))
    charge_C_rate += [None]*(target_len-len(charge_C_rate))
    discharge_C_rate += [None]*(target_len-len(discharge_C_rate))

    # add data to dictionary
    cycle_summary_data = {
        CLABEL.CYCLE_INDEX.value: cycle_number,
        CLABEL.AVG_V_C.value: avg_charge_voltage,
        CLABEL.AVG_V_D.value: avg_discharge_voltage,
        CLABEL.CE.value: coulombic_efficiency,
        CLABEL.V_HYS.value: voltage_hysteresis,
        CLABEL.ENERGY_EFF.value: energy_efficiency,
        CLABEL.CAP_C.value: charge_capacity,
        CLABEL.CAP_D.value: discharge_capacity,
        CLABEL.ENERGY_C.value: charge_energy,
        CLABEL.ENERGY_D.value: discharge_energy,
        CLABEL.CHARGE_STATUS.value: charge_cycle_status,
        CLABEL.DISCHARGE_STATUS.value: discharge_cycle_status,
        CLABEL.GRAV_CAP_C.value: grav_charge_capacity,
        CLABEL.GRAV_CAP_D.value: grav_discharge_capacity,
        CLABEL.GRAV_ENERGY_C.value: grav_charge_energy,
        CLABEL.GRAV_ENERGY_D.value: grav_discharge_energy,
        CLABEL.CHARGE_RATE.value: charge_rate,
        CLABEL.DISCHARGE_RATE.value: discharge_rate,
        CLABEL.CHARGE_C_RATE.value: charge_C_rate,
        CLABEL.DISCHARGE_C_RATE.value: discharge_C_rate,
    }
    # convert dictionary to dataframe
    c_df = pd.DataFrame(data=cycle_summary_data)

    if c_savepath:
        c_df.to_csv(c_savepath)
    return c_df

def plot_VoltProfile(cell_name, plt: pyplot = None, ax: axes.Axes = None,cycles=None, continuous=False, li_per_fu=True, \
        cmap=None, palette='flare', xlabel=None, ylabel='Voltage (V)', xlim=None, ylim=None, line_width=1.5, show_lgd=True, lgd_fsize=20, \
        show_fig=True, savefile=None, fsize=(8,8)):

    df = get_csv(cell_name)

    if not plt:
        plt = pyplot
        # set font size
        plt.rcParams['font.size'] = lgd_fsize
        plt.close()
        fig, ax = plt.subplots(figsize=fsize)
    else:
        fig, ax = plt.gcf(), plt.gca()

    # set color palette
    num_cycles = df[LABEL.CYCLE_INDEX.value].iloc[-1]+1
    if not cycles:
        cycles = list(range(num_cycles-1))
    else:
        cycles = list(range(cycles[0], cycles[1]))
    clean_cycles = [x for x in cycles if 0 <= x < num_cycles]
    first_cycle = clean_cycles[0]
    last_cycle = clean_cycles[-1]
    num_cycles_plot = len(cycles)+3
    colors = list(sns.color_palette(palette, n_colors=num_cycles_plot))

    # plot voltage-capacity cycles
    for i in clean_cycles:
        cycle_df = df[df[LABEL.CYCLE_INDEX.value] == i]
        cycle_c_df = cycle_df[cycle_df[LABEL.STEP_TYPE.value] == 'charge']
        cycle_d_df = cycle_df[cycle_df[LABEL.STEP_TYPE.value] == 'discharge']
        discharge_cap = cycle_d_df[LABEL.GRAV_CAP_D.value]
        charge_cap = cycle_c_df[LABEL.GRAV_CAP_C.value]
        #charge_cap can have some zero values that screw up the plot. This replaces 0s with NaNs to fix.
        charge_cap.replace(0, np.nan, inplace=True)
        if continuous:
            discharge_cap = charge_cap.iloc[-1] - discharge_cap
        color = colors[i + 1]
        label = 'Cycle {}'.format(i+1) if i == first_cycle or i == last_cycle else None
        ax.plot(charge_cap, cycle_c_df[LABEL.V.value], '-', color=color, lw=line_width, label=label)
        ax.plot(discharge_cap, cycle_d_df[LABEL.V.value], '-', color=color, lw=line_width)

    xlabel1 = 'Capacity (mAh/g)'
    ax.set_xlabel(xlabel1)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(str(cell_name))
    plt.tight_layout(pad=2)

    if li_per_fu:
        molar_mass = get_molar_mass(cell_name)
        ion = 'Li'
        single_ion_capacity = 1 / molar_mass * 96500 * 0.2777
        # function to convert active material specific capacity to ion ions inserted/ extracted
        def active_to_ion(capacity):
            return capacity / single_ion_capacity
        # function to convert alkali ions inserted/ extracted to active material specific capacity
        def ion_to_active(capacity):
            return capacity * single_ion_capacity

        if ax:
            # secondary x-axis settings
            # get settings from primary x-axis
            xax = ax.get_xaxis()
            minor_locator = copy.copy(xax.get_minor_locator())
            tick_len = ax.xaxis.majorTicks[0].tick1line.get_markersize()
            tick_width = ax.xaxis.majorTicks[0].tick1line.get_linewidth()
            direction = 'in'
            ax2 = ax.secondary_xaxis('top', functions=(active_to_ion, ion_to_active))
            ax2.tick_params(direction=direction, width=tick_width, length=tick_len)
            # secondary minor axis settings
            ax2.xaxis.set_minor_locator(minor_locator)
            ax2.tick_params(which='minor', direction=direction, width=tick_width, length=tick_len/2)
            twinxlabel = "ions inserted/ extracted"
            if ion:
                twinxlabel = "{} ions inserted/ extracted".format(ion)
            ax2.set_xlabel(twinxlabel)

    if show_lgd:
        ax.legend(prop={'size': lgd_fsize}, frameon=False).set_draggable(True)
    if savefile:
        # save the plot as a .png file.
        save_path = fig_save_dir + cell_name + "_VoltProfile.png"
        fig.savefig(save_path, dpi=300)
        print("Saved to {}".format(save_path))
    if show_fig:
        plt.show()
        plt.close()

def plot_CycvsCap(cell_names, plt: pyplot = None, ax: axes.Axes = None, plot_CE=False, cycles=None, continuous=False, label_names=None,
                  cmap=None, palette='tab10', xlabel=None, ylabel='Specific Capacity (mAh/g)', ce_ylabel='Coulombic Efficiency (%)', xlim=None, ylim=None, ce_ylim=None,
                  line_width=1.5, show_lgd=True, lgd_fsize=20, show_fig=True, savefile=None, save_name=None, fsize=(6,6)):
    if not plt:
        plt = pyplot
        # set font size
        plt.rcParams['font.size'] = lgd_fsize
        plt.close()
        plt.figure(figsize=fsize)
        ax = plt.gca()

    fig, ax1 = plt.subplots()
    colors = list(sns.color_palette(palette, n_colors=len(cell_names)))

    # If only one cell name is provided, convert it to a list to make it interact with the rest of the code
    if isinstance(cell_names, str):
        cell_names = [cell_names]

    for index in range(len(cell_names)):
        cell_name = cell_names[index]
        df = get_cell_summary(cell_name)
        capacity = df[CLABEL.GRAV_CAP_D.value]
        cycle_number = df[CLABEL.CYCLE_INDEX.value] + 1
        ax1.plot(cycle_number, capacity, linestyle='-', marker='o', lw=line_width, label=cell_name, color=colors[index])

    # Set labels and legend for primary axis
    ax1.set_xlabel('Cycle Number')
    ax1.set_ylabel('Capacity')

    # Plot CE data on the secondary axis
    if plot_CE:
        # Create secondary axis for CE data on the right
        ax2 = ax1.twinx()

        for index in range(len(cell_names)):
            cell_name = cell_names[index]
            df = get_cell_summary(cell_name)
            CE = df[CLABEL.CE.value]
            cycle_number = df[CLABEL.CYCLE_INDEX.value] + 1
            ax2.plot(cycle_number, CE, linestyle='-', marker='o', lw=line_width, label=None, color=colors[index], markerfacecolor='w')

        ax2.set_ylabel(ce_ylabel)
        ax2.set_ylim(ce_ylim)
        plt.tight_layout(pad=0.5)

    xlabel = 'Cycle #'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    plt.xlim(xlim)
    ax1.set_ylim(ylim)
    plt.subplots_adjust(left=0.2)
    plt.subplots_adjust(bottom=0.15)

    if show_lgd:
        plt.legend(prop={'size': lgd_fsize}, frameon=False).set_draggable(True)
    if savefile:
        # save the plot as a .png file.
        save_path = fig_save_dir + save_name + "_CycvsCap.png"
        plt.savefig(save_path)
        print("Saved to {}".format(save_path))
    if show_fig:
        plt.show()
    plt.close()

def plot_Efficiency(cell_names, plt: pyplot = None, ax: axes.Axes = None, cycles=None, continuous=False, label_names=None,
                  cmap=None, palette=None, xlabel=None, ylabel='Coulombic Efficiency (%)', xlim=None, ylim=None,
                  line_width=1.5, show_lgd=True, lgd_fsize=20, show_fig=True, savefile=None, save_name=None, fsize=(6,6)):
    if not plt:
        plt = pyplot
        # set font size
        plt.rcParams['font.size'] = lgd_fsize
        plt.close()
        plt.figure(figsize=fsize)
        ax = plt.gca()

    # If only one cell name is provided, convert it to a list to make it interact with the rest of the code
    if isinstance(cell_names, str):
        cell_names = [cell_names]

    # Set color palette and plot
    if not palette:
        for cell_name in cell_names:
            df = get_cell_summary(cell_name)
            CE = df[CLABEL.CE.value]
            cycle_number = df[CLABEL.CYCLE_INDEX.value] + 1
            plt.plot(cycle_number, CE, linestyle='-', marker='o', lw=line_width, label=cell_name, markerfacecolor='w')
    else:
        colors = list(sns.color_palette(palette, n_colors=len(cell_names)))

        for index in range(0, len(cell_names)):
            cell_name = cell_names[index]
            print(cell_name)
            df = get_cell_summary(cell_name)
            CE = df[CLABEL.CE.value]
            cycle_number = df[CLABEL.CYCLE_INDEX.value] + 1
            plt.plot(cycle_number, CE, linestyle='-', marker='o', lw=line_width, label=cell_name, color=colors[index])


    xlabel = 'Cycle #'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(cell_name)
    plt.subplots_adjust(left=0.2)
    plt.subplots_adjust(bottom=0.15)
    if show_lgd:
        plt.legend(prop={'size': lgd_fsize}, frameon=False).set_draggable(True)
    if savefile:
        # save the plot as a .png file.
        save_path = fig_save_dir + save_name + "_CE.png"
        plt.savefig(save_path)
        print("Saved to {}".format(save_path))
    if show_fig:
        plt.show()
    plt.close()

def plot_VoltTime(cell_name,plt: pyplot=None, ax: axes.Axes=None,cycles=None, continuous=False,\
        cmap=None, palette='flare', xlabel=None, ylabel='Voltage (V)', xlim=None, ylim=None, line_width=1.5, show_lgd=True, lgd_fsize=20,\
        show_fig=True, savefile=None, fsize=(6,6)):

    df = get_csv(cell_name)


    if not plt:
        plt = pyplot
        # set font size
        plt.rcParams['font.size'] = lgd_fsize
        plt.close()
        plt.figure(figsize=fsize)
        ax = plt.gca()



    # plot voltage-capacity cycles
    df_time = df[LABEL.TEST_TIME.value]/3600
    plt.plot(df_time, df[LABEL.V.value], '-', color='black', lw=line_width)

    xlabel = 'Time (hours)'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(cell_name)
    plt.subplots_adjust(bottom=0.15)

    if show_lgd:
        plt.legend(prop={'size': lgd_fsize}, frameon=False).set_draggable(True)
    if savefile:
        # save the plot as a .png file.
        save_path = fig_save_dir + cell_name + "_VoltTime.png"
        plt.savefig(save_path, dpi=300)
        print("Saved to {}".format(save_path))
    if show_fig:
        plt.show()
    plt.close()

def plot_dQdV(cell_name,plt: pyplot=None, ax: axes.Axes=None,cycles=None, continuous=False,\
        cmap=None, palette='flare', xlabel=None, ylabel='dQ/dV (mAh/gV)', xlim=None, ylim=None, line_width=1.5, show_lgd=True, lgd_fsize=20,\
        show_fig=True, savefile=None, fsize=(6,6)):

    df = get_csv(cell_name)


    if not plt:
        plt = pyplot
        # set font size
        plt.rcParams['font.size'] = lgd_fsize
        plt.close()
        plt.figure(figsize=fsize)
        ax = plt.gca()

    plt, ax = format_plot(
        fig_size=(8,8)
    )
    # set color palette
    num_cycles = df[LABEL.CYCLE_INDEX.value].iloc[-1]+1
    if not cycles:
        cycles = list(range(num_cycles-1))
    else:
        cycles =  list(range(cycles[0],cycles[1]))
    clean_cycles = [x for x in cycles if x >= 0 and x < num_cycles]
    first_cycle = clean_cycles[0]
    last_cycle = clean_cycles[-1]
    num_cycles_plot = len(cycles)+3
    colors = list(sns.color_palette(palette, n_colors=num_cycles_plot))


    # plot dQ/dV cycles

    for i in clean_cycles:
        cycle_df = df[df[LABEL.CYCLE_INDEX.value] == i]
        cycle_c_df = cycle_df[cycle_df[LABEL.STEP_TYPE.value] == 'charge']
        cycle_d_df = cycle_df[cycle_df[LABEL.STEP_TYPE.value] == 'discharge']
        cycle_d_df[LABEL.DQ_DV.value] = cycle_d_df[LABEL.DQ_DV.value]* -1
        cycle_df = pd.concat([cycle_c_df,cycle_d_df], ignore_index=True)
        color=colors[i + 1]
        if i == first_cycle or i == last_cycle:
            label = 'Cycle {}'.format(i+1)
        else:
            label = None
        plt.plot(cycle_df[LABEL.V.value], cycle_df[LABEL.DQ_DV.value], '-', color=color, lw=line_width, label=label)


    xlabel = 'Voltage (V)'
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(cell_name)
    plt.subplots_adjust(bottom=0.15)

    if show_lgd:
        plt.legend(prop={'size': lgd_fsize}, frameon=False).set_draggable(True)
    if savefile:
        # save the plot as a .png file.
        save_path = fig_save_dir + cell_name + "_VoltProfile.png"
        plt.savefig(save_path, dpi=300)
        print("Saved to {}".format(save_path))
    if show_fig:
        plt.show()
    plt.close()

# cells=['Test1_20240304','Test2_20240304','Test3_20240304','Test4_20240304']
#plot_VoltProfile(cell_names[0])
#plot_VoltProfile('LM9-LPR-E4-4a')
#plot_VoltTime('LM9-LPR-E4-4F', savefile='yes')
#plot_dQdV('Test1_20240304',cycles=[1,101])
#plot_CycvsCap(cells,plot_CE=True,savefile=True,save_name="")

#for cell in cell_names1:
    #plot_VoltProfile(cell)

#for cell in cell_names_2:
    #plot_VoltProfile(cell)


cells=["LPSC_LZC_bilayer1_20240401",
"LPSC_LZC_bilayer2_20240401",
"LPSC_LZC_bilayer3_20240401",
"LPSC_LZC_bilayer4_20240401",
"Test1-4_20240219_75MPa_C20",
"Test2-4_20240219_75MPa_C20",
"Test3-4_20240219_75MPa_C20",
"Test4-4_20240219_75MPa_C20"]

# plot_CycvsCap("LPSC_LZC_bilayer3_20240401",plot_CE=True,savefile=True,save_name="LPSC_LZC_bilayer3_20240401",cycles=[1,100])
# plot_CycvsCap("LPSC_LZC_bilayer4_20240401",plot_CE=True,savefile=True,save_name="LPSC_LZC_bilayer4_20240401",cycles=[1,126])
plot_CycvsCap("Test1-4_20240219_75MPa_C20",plot_CE=True)






# def get_cell_summary(cell_name):
#     df = get_csv(cell_name)
#     num_cycles = df[LABEL.CYCLE_INDEX.value].iloc[-1]+1
#     cycles = list(range(num_cycles-1))
#     clean_cycles = [x for x in cycles if x >= 0 and x < num_cycles]

#     # Define the columns of the cell summary
#     cycle_index = []
#     c_cap = []
#     c_gravcap = []
#     c_energy = []
#     c_gravenergy = []
#     d_cap = []
#     d_gravcap = []
#     d_energy = []
#     d_gravenergy = []
#     for i in clean_cycles:
#         cycle_df = df[df[LABEL.CYCLE_INDEX.value] == i]
#         #cycle_c_df = cycle_df[cycle_df['step type'] == 'charge']
#         #cycle_d_df = cycle_df[cycle_df['step type'] == 'discharge']

#         target_chg_columns = ['Q charge/mA.h','gravimetric Q charge (mAh/g)','Energy charge/W.h','gravimetric energy charge (Wh/g)']
#         target_dchg_columns = ['Q discharge/mA.h','gravimetric Q discharge (mAh/g)','Energy discharge/W.h','gravimetric energy discharge (Wh/g)']
#         # find the maximum or average value for various metrics for charge and discharge of each cycle
#         cycle_index.append(cycle_df[LABEL.CYCLE_INDEX.value].max())
#         #Chg
#         c_cap.append(cycle_df[LABEL.CAP_C.value].max())
#         c_gravcap.append(cycle_df[LABEL.GRAV_CAP_C.value].max())
#         c_energy.append(cycle_df[LABEL.ENERGY_C.value].max())
#         c_gravenergy.append(cycle_df[LABEL.GRAV_ENERGY_C.value].max())
#         #DChg
#         d_cap.append(cycle_df[LABEL.CAP_D.value].max())
#         d_gravcap.append(cycle_df[LABEL.GRAV_CAP_D.value].max())
#         d_energy.append(cycle_df[LABEL.ENERGY_D.value].max())
#         d_gravenergy.append(cycle_df[LABEL.GRAV_ENERGY_D.value].max())

#     # Combine all the target columns into a dataframe
#     summary_df = pd.DataFrame({LABEL.CYCLE_INDEX.value:cycle_index,
#                        LABEL.CAP_C.value:c_cap,
#                        LABEL.GRAV_CAP_C.value:c_gravcap,
#                        LABEL.ENERGY_C.value:c_energy,
#                        LABEL.GRAV_ENERGY_C.value:c_gravenergy,
#                        LABEL.CAP_D.value:d_cap,
#                        LABEL.GRAV_CAP_D.value:d_gravcap,
#                        LABEL.ENERGY_D.value:d_energy,
#                        LABEL.GRAV_ENERGY_D.value:d_gravenergy})

#     summary_df.to_csv(save_dir+cell_name+'_summary.csv', index=False)
#     return(summary_df)
