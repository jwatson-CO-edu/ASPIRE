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
    Support = create_type( "Support" ) # - There are several places the monkey could stand
    Elev    = create_type( "Elev"    ) # - Relative elevation


    ##### Predicates #############################

    @predicate()
    def monkey_hungry( self ):
        """ <MONKEY @ STATUS> state """

    @predicate( Loc, Elev )
    def monkey_at( self, loc, elev ):
        """ <MONKEY @ LOC, ELEV> state """

    @predicate( Fruit, Loc, Elev )
    def fruit_at( self, fruit, loc, elev ):
        """ <MONKEY @ ELEVATION> state """

    @predicate( Support, Loc )
    def support_at( self, support, loc ):
        """ <MONKEY @ LOC, ELEV> state """

    @predicate( Support, Elev )
    def support_height( self, support, elev ):
        """ <MONKEY @ LOC, ELEV> state """


    ##### Actions ################################

    @action( Support, Loc, Loc )
    def move_support( self, support, locBgn, locEnd ):

        precond = [ self.support_at( support, locBgn ), ]

        effect  = [ self.support_at( support, locEnd ), 
                   ~self.support_at( support, locBgn ),  ]
        
        return precond, effect
    

    @action( Loc, Elev, Support, Loc, Elev )
    def go_from_to( self, locBgn, elevBgn, support, locEnd, elevEnd ):

        precond = [ self.monkey_at( locBgn, elevBgn ), 
                    self.support_at( support, locEnd ), 
                    self.support_height( support, elevEnd ) ]
        
        effect  = [ self.monkey_at( locEnd, elevEnd ), 
                   ~self.monkey_at( locBgn, elevBgn ), ]
        
        return precond, effect
    

    @action( Fruit, Loc, Elev )
    def eat( self, fruit, loc, elev ):

        precond = [ self.monkey_at( loc, elev ), 
                    self.fruit_at( fruit, loc, elev ), ]
        
        effect  = [ ~self.monkey_hungry(), ]

        return precond, effect



########## PROBLEM SPECIFICATION ###################################################################


class MonkeyProblem( MonkeyDomain ):

    def __init__( self, supports, locations, elevations, fruitNames ):
        super().__init__()
        self.sppt = MonkeyDomain.Support.create_objs( supports  , prefix = "support" )
        self.locs = MonkeyDomain.Loc.create_objs(     locations , prefix = "loc"     )
        self.elev = MonkeyDomain.Elev.create_objs(    elevations, prefix = "elev"    )
        self.frut = MonkeyDomain.Fruit.create_objs(   fruitNames, prefix = "fruit"   )

    @init
    def init( self, initDict ) -> list:
        rtnStates = []
        if 'support_at' in initDict:
            for (support, loc) in initDict['support_at']:
                rtnStates.append(
                    self.support_at( self.sppt[ support ], self.locs[ loc ] )
                )
        if 'monkey_at' in initDict:
            for (loc, elev) in initDict['monkey_at']:
                rtnStates.append(
                    self.monkey_at( self.locs[ loc ], self.elev[ elev ] )
                )
        if 'fruit_at' in initDict:
            for (fruit, loc, elev) in initDict['fruit_at']:
                rtnStates.append(
                    self.fruit_at( self.frut[ fruit ], self.locs[ loc ], self.elev[ elev ] )
                )
        if 'support_height' in initDict:
            for (support, elev) in initDict['support_height']:
                rtnStates.append(
                    self.support_height( self.sppt[ support ], self.elev[ elev ] )
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