# Adapted from code by Eric Appelt: https://gist.github.com/appeltel/fd3ddeeed6c330c7208502462639d2c9

########## DEV PLAN ################################################################################


########## INIT ####################################################################################

import asyncio
import random



########## UTILITY CLASSES #########################################################################

class Hub():
    """ Distributes each message to the queue of each all subscription """
    # FIXME: ROUTING / SORTING SHOULD TAKE PLACE HERE

    def __init__( self ):
        """ Maintain subscriptions as a set """
        self.subscriptions = set()

    def publish( self, message ):
        """ For each queue referenced, push message """
        for queue in self.subscriptions:
            queue.put_nowait( message )