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

    trialData = metrics['trials']
    failTypes = set([])
    print( len( trialData ) )
    print( f"Success Rate: {len(msPass)/len(metrics['trials'])}" )
    
    for ep in trialData:
        for k in ep.keys():
            if k not in ('result','makespan',):
                failTypes.add(k)
    
    failTypes = list( failTypes )
    fSeries   = [[] for _ in failTypes]

    for ep in trialData:
        for typ in failTypes:
            if (typ in ep):
                fSeries[ failTypes.index( typ ) ].append( ep[ typ ] )
            else:
                fSeries[ failTypes.index( typ ) ].append( 0 )

    ### Analyze ###
    
    plt.boxplot( [msPass, msFail],
                 vert=True,  # vertical box alignment
                 patch_artist=True,  # fill with color
                 labels=["Pass","Fail"],# will be used to label x-ticks
                 showfliers=False)  
    plt.ylabel('Makespan [s]')
    plt.savefig( 'fullDemo_Whisker.pdf' )

    plt.show()
    plt.clf()

    for i, typ in enumerate( failTypes ):
        print( f"{typ}: {fSeries[i]}" )

    plt.rcParams['figure.figsize'] = [20, 8]

    plt.boxplot( fSeries,
                 vert=True,  # vertical box alignment
                 patch_artist=True,  # fill with color
                 labels=failTypes,# will be used to label x-ticks
                 showfliers=False)  
    plt.ylabel('Occurrences')
    plt.savefig( 'fullDemo_faultsWhisker.pdf' )

    plt.show()
    plt.clf()
