########## INIT ####################################################################################

import time, sys, pickle



########## MAIN ####################################################################################
if __name__ == "__main__":

    data = None

    with open( 'fullDemo250_2024-01-26.pkl', 'rb' ) as handle:
        metrics = pickle.load( handle )

    print( len( metrics["trials"] ) )