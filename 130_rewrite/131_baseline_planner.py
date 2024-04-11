########## DEV PLAN ################################################################################
"""
[ ] Combine `ObjectBelief` and `ObjectMemory` --> `ObjectMemory`, SIMPLIFY!
"""

########## INIT ####################################################################################

from symbols import GraspObj, ObjectReading



########## BELIEFS #################################################################################

class ObjectMemory:
    """ Attempt to maintain recent and constistent object beliefs based on readings from the vision system """

    def reset_beliefs( self ):
        """ Remove all references to the beliefs, then erase the beliefs """
        self.beliefs = []

    def __init__( self ):
        """ Set belief containers """
        self.reset_beliefs()



########## BASELINE PLANNER ########################################################################

class BaselineTAMP:
    """ Basic TAMP loop against which the Method is compared """

    ##### Init ############################################################

    pass