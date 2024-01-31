########## INIT ####################################################################################

import pickle
import matplotlib.pyplot as plt



########## MAIN ####################################################################################
if __name__ == "__main__":

    data   = None
    msPass = []
    msFail = []

    with open( 'fullDemo250_2024-01-26.pkl', 'rb' ) as handle:
        metrics = pickle.load( handle )

    with open( 'fullDemo250_2024-01-26_msPass.pkl', 'rb' ) as handle:
        msPass = pickle.load( handle )

    with open( 'fullDemo250_2024-01-26_msFail.pkl', 'rb' ) as handle:
        msFail = pickle.load( handle )

    ### Analyze ###
    
    plt.boxplot( [msPass, msFail],
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=["Pass","Fail"],# will be used to label x-ticks
                     showfliers=False)  
    plt.ylabel('Makespan [s]')
    plt.savefig( 'fullDemo_Whisker.pdf' )

    plt.show()


    print( len( metrics["trials"] ) )