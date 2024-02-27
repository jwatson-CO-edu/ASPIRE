import os
from pprint import pprint

import matplotlib.pyplot as plt

from utils import get_merged_logs_in_dir_with_prefix

if __name__ == "__main__":
    drctry = "./93d_Alt_FD_Params_3/data/cea-wastar1_bt-short"
    prefix = "TAMP-Loop__2024-02-2"
    data   = get_merged_logs_in_dir_with_prefix( drctry, prefix )
    print( f"There are {data['N']} trials." )
    print( f"Success Rate: {data['pass']/data['N']}" )
    

    print( "\n\n" )

    ep = data['trials'][51]
    print( ep['result'] )
    for event in ep['events']:
        print( event[:2] )