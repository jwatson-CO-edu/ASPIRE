########## INIT ####################################################################################

import os
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from utils import get_merged_logs_in_dir_with_prefix



########## DATA AGGREGATION & PROCESSING ###########################################################

def collect_pass_fail_makespans( data ):
    """ Get the makespans for pass and fail separately so that they can be compared """
    msPass = []
    msFail = []
    for trial in data['trials']:
        if trial['result']:
            msPass.append( trial['makespan'] )
        else:
            msFail.append( trial['makespan'] )
    return msPass, msFail


def collect_multiple_makespan_datasets( dirDct, prefix = "TAMP-Loop" ):
    """ Gather a collection of named datasets for analysis """
    dataDct = {}
    for name, path in dirDct.items():
        data_i = get_merged_logs_in_dir_with_prefix( path, prefix )
        msPass, msFail = collect_pass_fail_makespans( data_i )
        print( f"Series {name}: Success Rate = {len(msPass) / (len(msPass)+len(msFail))}" )
        dataDct[ name ] = {
            'data'  : data_i,
            'msPass': msPass,
            'msFail': msFail,
        }
    return dataDct



########## PLOTTING ################################################################################

def plot_pass_fail_makespans( msPass, msFail, plotName ):
    plt.boxplot( [msPass, msFail],
                 vert=True,  # vertical box alignment
                 patch_artist=True,  # fill with color
                 labels=["Pass","Fail"],# will be used to label x-ticks
                 showfliers=False)  
    plt.ylabel('Makespan [s]')
    plt.savefig( str( plotName ) + "_pass-fail-makespans" + '.pdf' )

    plt.show()
    plt.clf()


def plot_pass_fail_histo( msPass, msFail, Nbins, plotName ):
    plt.hist( [msPass, msFail], Nbins, histtype='bar', label=["Success", "Failure"] )

    plt.legend(); plt.xlabel('Episode Makespan'); plt.ylabel('Count')
    plt.savefig( str( plotName ) + "_pass-fail-histogram" + '.pdf' )

    plt.show()
    plt.clf()


def plot_sweep_pass_makespans( data, plotName ):
    """  """
    datNames = list( data.keys() )
    datNames.sort()
    pltSeries = []
    for datName in datNames:
        pltSeries.append( data[ datName ]['msPass'] )
    plt.boxplot( pltSeries,
                 vert=True,  # vertical box alignment
                 patch_artist=True,  # fill with color
                 labels=datNames,# will be used to label x-ticks
                 showfliers=False)  
    plt.title('Expected Block Tower Makespan -vs- Confusion')
    plt.xlabel('Total Confusion Probability')
    plt.ylabel('Makespan [s]')
    plt.savefig( str( plotName ) + "_sweep-makespans" + '.pdf' )
    plt.show()
    plt.clf()
        




########## INIT ####################################################################################

if __name__ == "__main__":
    data = collect_multiple_makespan_datasets( 
        {
            f"{np.round(0.001*6,3):.3f}" : "./data/",
            f"{np.round(0.010*6,3):.3f}" : "./132a_sweep/data/",
            f"{np.round(0.025*6,3):.3f}" : "./132b_sweep/data/",
            f"{np.round(0.050*6,3):.3f}" : "./132c_sweep/data/",
            f"{np.round(0.075*6,3):.3f}" : "./132d_sweep/data/",
        }, 
        prefix = "TAMP-Loop" 
    )
    plot_sweep_pass_makespans( data, "TAMP-Baseline-Conf" )
    
    
    # get_merged_logs_in_dir_with_prefix( drctry, prefix )
    # print( f"There are {data['N']} trials." )
    # print( f"Success Rate: {data['pass']/data['N']}" )
    
    # msPass, msFail = collect_pass_fail_makespans( data )
    # plot_pass_fail_makespans( msPass, msFail, prefix )
    # plot_pass_fail_histo( msPass, msFail, 10, prefix )
