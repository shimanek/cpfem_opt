"""
Compare the full geometry model to the 100-element chain model for the PAN model.
1. Shipin full geometry model vs 100-element chain model
2. Change in chain model as a function of length?
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import AutoMinorLocator

## i/o folders
data_dir = os.path.join(os.getcwd(), 'data')
figure_dir = os.path.join(os.getcwd(), 'figures')

def main():
    """import each dataset and save plot"""
    ## data:
    old235 = get_strain_stress('old100EL_235.csv', length=100, area=1)
    new235 = get_strain_stress('new_235.csv', length=1, area=1)
    old128 = get_strain_stress('old_128.csv', length=100, area=1)
    new128 = get_strain_stress('new_128.csv', length=1, area=1)
    fullGeo128 = get_strain_stress('out_Ni_fullGeo_1b-2-8.csv', length=71.12, area=3.94)

    ## plotting:
    plot_SS([old235, new235], ['Old B.C., 100 Elements', 'New B.C., 1 Element'], 
        'Model and B.C. Comparison', 'fig_ModelCompare_235')
    plot_SS([old128, new128, fullGeo128], ['Old B.C., 100 Elements', 'New B.C., 1 Element', 'Full Mesh, $2\cdot10^{4}$ Elements'], 
        'Model Comparison [128]', 'fig_ModelCompare_128_fullGeo')


## functions
def get_strain_stress(filename, length, area):
    """get stress and strain for each filename"""
    # data_folder = os.path.join(os.getcwd(), 'data')
    temp_path = os.path.join(data_dir, filename)
    temp_data = np.loadtxt( temp_path, delimiter=',', skiprows=1 )
    strain = temp_data[:,1] / length
    stress = temp_data[:,2] / area
    return np.stack(( strain.transpose(), stress.transpose() ), axis=1)

def plot_SS(strain_stress, labels, title, filename):
    """plot stress vs strain and apply common settings"""
    fig, ax = plt.subplots()
    colors = ['black', 'brown'] if len(labels)==2 else ['black','brown','blue']
    styles = ['solid']*10
    markers = ['s', 'x'] if len(labels)==2 else ['s', 'x', 'o']
    for i, ss in enumerate(strain_stress):
        ax.plot(ss[:,0],ss[:,1], marker=markers[i], label=labels[i], color=colors[i], linestyle=styles[i], markerfacecolor='None', linewidth=1)

    # legend:
    handles, labels = plt.gca().get_legend_handles_labels()
    order = np.argsort( [max(ss[:,1]) for ss in strain_stress] )[::-1]
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], 
        loc='lower right', fancybox=False, edgecolor='black', borderaxespad=1.0)
    # axis settings:
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Engineering Strain [m/m]')
    ax.set_ylabel('Engineering Stress [MPa]')
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_title(title)
    # save and close:
    fig.savefig(os.path.join(figure_dir,filename), dpi=600, bbox_inches='tight')
    plt.close()


## folder structure: 
# .
# ├── data
# │   ├── out_100el_1b-5-10.csv
# │   └── out_Ni_fullGeo_1b-5-10.csv
# ├── figures
# ├── pics
# │   ├── res_Ni-1510_100el_stress_noEdges.png
# │   ├── res_Ni-1510_100el_stress_yesEdges.png
# │   ├── res_Ni-1510_fullGeo_stress_noEdges.png
# │   ├── res_NiAl_fullGeo_stress.png
# │   └── res_NiAl_fullGeo_stress_noEdges.png
# ├── plot2.py
# └── plot_nix_SS.py



if __name__ == '__main__':
    main()
