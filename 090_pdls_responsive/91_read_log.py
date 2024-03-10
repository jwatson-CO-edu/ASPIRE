import os
from pprint import pprint

import matplotlib.pyplot as plt

from utils import get_merged_logs_in_dir_with_prefix




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


if __name__ == "__main__":
    # drctry = "./data/spec-domain-stoch/"
    drctry = "./93c_Alt_FD_Params_2/data/spec-domain-stoch-deathwatch/"
    prefix = "TAMP-Loop"
    data   = get_merged_logs_in_dir_with_prefix( drctry, prefix )
    print( f"There are {data['N']} trials." )
    print( f"Success Rate: {data['pass']/data['N']}" )
    
    msPass, msFail = collect_pass_fail_makespans( data )
    plot_pass_fail_makespans( msPass, msFail, prefix )
    plot_pass_fail_histo( msPass, msFail, 10, prefix )
