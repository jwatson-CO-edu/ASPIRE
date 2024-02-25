########## DEV PLAN ################################################################################
"""
[ ] Simulate an event sequence
[ ] Take symbol likelihood into account
[ ] Frame the task-wide Trial-and-Error problem
[ ] Q: How to plan for a specified likelihood of a pose distribution?

[ ] Q: How do we use the projected duration and result?
    * Possible scenarios
        - Robot is given a time limit for a task
        - Robot has to choose between tasks
        - Conmod install vs restart decisions

* Predict failure, but from what?
    - FT sequence?
    - Time sequence of scene graphs?

* https://github.com/correlllab/Factor-Graphs-for-Failure-Analysis/blob/main/zc_WeBots_PDDL/80_Beta-Dist.ipynb
"""
########## INIT ####################################################################################

from uuid import uuid5

import numpy as np
from scipy.stats import beta


########## PLAN ROLLOUTS ###########################################################################

class Event:
    """ Something that happens with probabilistic duration and success """

    def __init__( self ):
        """ Set params """
        self.ID       = uuid5()
        self.name     = "Event_" + str( self.ID )
        self.durMean  = 1.0
        self.durStdv  = 0.5
        self.Npass    = 0
        self.Nfail    = 0
        self.duration = 0.0
        self.result   = False
        self.prev     = None
        self.next     = None

    def obs_pass( self ):
        """ Observe a success """
        self.Npass += 1

    def obs_fail( self ):
        """ Observe a failure """
        self.Nfail += 1

    def simulate( self ):
        """ Simulate duration and result, Return a cumulative duration and result from all following events """
        # NOTE: It is assumed that the client code should call this from the head node
        if (self.prev is not None) and (not self.prev.result):
            self.duration = 0.0
            self.result   = False
        else:
            self.duration = np.random.normal( self.durMean, self.durStdv )
            self.result   = (beta.rvs( self.Npass, self.Nfail ) >= 0.5)
        rtnDur = self.duration
        rtnRes = self.result
        if (self.next is not None):
            nxtDur, nxtRes = self.next.simulate()
            rtnDur += nxtDur
            rtnRes =  (rtnRes and nxtRes)
        return rtnDur, rtnRes

    def append( self, other ):
        """ Attach the event that follows this one """
        self.next  = other
        other.prev = self

    def prepend( self, other ):
        """ Attach the event that preceeds this one """
        other.next = self
        self.prev  = other





