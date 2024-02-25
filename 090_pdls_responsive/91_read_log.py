import os
from pprint import pprint

import matplotlib.pyplot as plt

from utils import DataLogger

def get_paths_in_dir_with_prefix( directory, prefix ):
    """ Get only paths in the `directory` that contain the `prefix` """
    fPaths = [os.path.join(directory, f) for f in os.listdir( directory ) if os.path.isfile( os.path.join(directory, f))]
    return [path for path in fPaths if (prefix in str(path))]

def get_merged_logs_in_dir_with_prefix( directory, prefix ):
    """ Merge all logs into one that can be analized easily """
    pklPaths = get_paths_in_dir_with_prefix( directory, prefix )
    logMain  = DataLogger()
    for path in pklPaths:
        log_i = DataLogger()
        log_i.load( path )
        # pprint( log_i.metrics )
        # print( '\n' )
        logMain.merge_from( log_i )
    return logMain.get_snapshot()


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
    prefix = "TAMP-Loop__2024-02-2"
    data   = get_merged_logs_in_dir_with_prefix( "./93g_Alt_FD_Params_6/data/dijkstra/", prefix )
    print( f"There are {data['N']} trials." )
    print( f"Success Rate: {data['pass']/data['N']}" )
    
    msPass, msFail = collect_pass_fail_makespans( data )
    plot_pass_fail_makespans( msPass, msFail, prefix )
    plot_pass_fail_histo( msPass, msFail, 10, prefix )
