import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator


def format_plot(fig_size=(8, 8), fsize=24, x_tick_spacing=None,
                y_tick_spacing=None, minor_tick_interval=2, tick_len=10,
                palette=None, num_palette_colors=None, minor_x=True,
                x_rot=None, set_integer_x_spacing=False, border_thickness=1.5,
                secondary_x=False, secondary_y=False, hide_y=False,
                forward_func=None, reverse_func=None, secondary_color='black'):
    # plot parameters
    if palette:
        if num_palette_colors:
            sns.set_palette(palette, num_palette_colors)
        else:
            sns.set_palette(palette)
    plt.figure(figsize=fig_size, dpi=100)
    plt.rcParams.update({'font.size': fsize})
    plt.rcParams.update({'font.family': 'Arial'})
    ax = plt.gca()
    # major axis settings
    border_thickness = 1.5
    ax.xaxis.set_tick_params(width=border_thickness)
    ax.yaxis.set_tick_params(width=border_thickness)
    tick_len = 10
    ax.tick_params(direction='in', width=border_thickness, length=tick_len)
    [x.set_linewidth(border_thickness) for x in ax.spines.values()]
    if x_tick_spacing:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_spacing))
    if y_tick_spacing:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_spacing))
    if set_integer_x_spacing:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # minor axis settings
    if minor_x:
        ax.xaxis.set_minor_locator(AutoMinorLocator(minor_tick_interval))
    ax.yaxis.set_minor_locator(AutoMinorLocator(minor_tick_interval))
    ax.tick_params(which='minor', direction='in', width=border_thickness,
                   length=tick_len/2)
    # minor axis settings
    if minor_x:
        ax.xaxis.set_minor_locator(AutoMinorLocator(minor_tick_interval))
    ax.yaxis.set_minor_locator(AutoMinorLocator(minor_tick_interval))
    ax.tick_params(which='minor', direction='in', width=border_thickness,
                   length=tick_len/2)
    if secondary_x:
        # secondary x-axis settings
        ax2 = ax.secondary_xaxis('top', functions=(forward_func, reverse_func))
        ax2.tick_params(direction="in", width=border_thickness,
                        length=tick_len)
        # secondary minor axis settings
        ax2.xaxis.set_minor_locator(AutoMinorLocator(minor_tick_interval))
        ax2.tick_params(which='minor', direction='in', width=border_thickness,
                        length=tick_len/2)
        return plt, ax, ax2
    if secondary_y:
        # secondary y-axis settings
        ax2 = ax.twinx()
        ax2.tick_params(direction="in", width=border_thickness,
                        length=tick_len)
        ax2.tick_params(axis='y', labelcolor=secondary_color)
        # secondary minor axis settings
        ax2.xaxis.set_minor_locator(AutoMinorLocator(minor_tick_interval))
        ax2.tick_params(which='minor', direction='in', width=border_thickness,
                        length=tick_len/2)
        return plt, ax, ax2
    if hide_y:
        ax.get_yaxis().set_visible(False)
    return plt, ax
