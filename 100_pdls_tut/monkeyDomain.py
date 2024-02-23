########## INIT ####################################################################################

from py2pddl import Domain, create_type
from py2pddl import predicate, action, goal, init

########## DOMAIN ##################################################################################


class MonkeyDomain( Domain ):
    """ Simple PDDL Problem """


    ##### Symbol Types ###########################

    # Monkey  = create_type( "Monkey"  ) # There is only one agent
    Fruit   = create_type( "Fruit"   ) # - Allow multiple fruit
    Loc     = create_type( "Loc"    ) # -- There are several places the chair could be


    ##### Predicates #############################

    @predicate()
    def monkey_hungry( self ):
        """ <HUNGRY> state """

    @predicate( Loc )
    def monkey_at( self, loc ):
        """ <MONKEY @ LOC, ELEV> state """

    @predicate( Fruit, Loc )
    def fruit_at( self, fruit, loc ):
        """ <MONKEY @ ELEVATION> state """

    @predicate( Loc )
    def chair_at( self,  loc ):
        """ <CHAIR @ LOC> state """


    ##### Actions ################################

    @action( Loc, Loc )
    def move_chair( self, locBgn, locEnd ):

        precond = [ self.chair_at( locBgn ),
                    self.monkey_at( locBgn ), ]

        effect  = [ self.monkey_at( locEnd ), 
                   ~self.monkey_at( locBgn ),
                    self.chair_at( locEnd ), 
                   ~self.chair_at( locBgn ),  ]
        
        return precond, effect
    

    @action( Loc, Loc )
    def go_from_to( self, locBgn, locEnd ):

        precond = [ self.monkey_at( locBgn ) ]
        
        effect  = [ self.monkey_at( locEnd ), 
                   ~self.monkey_at( locBgn ), ]
        
        return precond, effect
    

    @action( Fruit, Loc )
    def eat( self, fruit, loc ):

        precond = [ self.monkey_at( loc ), 
                    self.chair_at( loc ),
                    self.fruit_at( fruit, loc ), ]
        
        effect  = [ ~self.monkey_hungry(),
                    ~self.fruit_at( fruit, loc ) ]

        return precond, effect



########## PROBLEM SPECIFICATION ###################################################################


class MonkeyProblem( MonkeyDomain ):

    def __init__( self, locations, fruitNames ):
        super().__init__()
        self.locs = MonkeyDomain.Loc.create_objs(     locations , prefix = "loc"     )
        self.frut = MonkeyDomain.Fruit.create_objs(   fruitNames, prefix = "fruit"   )

    @init
    def init( self, initDict ) -> list:
        rtnStates = []
        if 'chair_at' in initDict:
            for loc in initDict['chair_at']:
                rtnStates.append(
                    self.chair_at(  self.locs[ loc ] )
                )
        if 'monkey_at' in initDict:
            for loc in initDict['monkey_at']:
                rtnStates.append(
                    self.monkey_at( self.locs[ loc ] )
                )
        if 'fruit_at' in initDict:
            for (fruit, loc) in initDict['fruit_at']:
                rtnStates.append(
                    self.fruit_at( self.frut[ fruit ], self.locs[ loc ] )
                )
        rtnStates.append( self.monkey_hungry() ) # The monkey always begins hungry
        return rtnStates
    
    @goal
    def goal( self, goalDict ) -> list:
        rtnGoals = []
        if 'hungry' in goalDict:
            if not goalDict['hungry']:
                rtnGoals.append(  ~self.monkey_hungry()  )
            else:
                rtnGoals.append(  self.monkey_hungry()  )
        return rtnGoals