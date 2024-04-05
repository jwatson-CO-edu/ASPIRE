########## INIT ####################################################################################

import os
from pprint import pprint

import matplotlib.pyplot as plt

from utils import get_merged_logs_in_dir_with_prefix



########## MAIN ####################################################################################

if __name__ == "__main__":
    drctry = "./data/"
    prefix = "TAMP-Loop"
    data   = get_merged_logs_in_dir_with_prefix( drctry, prefix )
    print( f"There are {data['N']} trials." )
    print( f"Success Rate: {data['pass']/data['N']}" )

    confCount = 0
    msngCount = 0
    dethCount = 0
    # confFound = False

    for trial in data['trials']:
        # confFound = False
        for event in trial['events']:
            # print( event )
            evn = event[1]
            msg = event[2]
            if ("CONFUSION" in evn) or ("CONFUSION" in msg):
                confCount += 1
            if ("missing" in evn) or ("missing" in msg):
                msngCount += 1
            if ("BRAINDEATH" in evn) or ("BRAINDEATH" in msg):
                dethCount += 1
        # print( '\n' )

    for event in data['trials'][50]['events']:
        print( f"\t{event}" )
                    
    print( f"Counted {confCount} confusion events in {data['N']} trials, Average: {confCount/data['N']}" )
    print( f"Counted {msngCount} missing events in {data['N']} trials, Average: {msngCount/data['N']}" )
    print( f"Counted {dethCount} braindeath events in {data['N']} trials, Average: {dethCount/data['N']}" )
    