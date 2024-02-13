import subprocess
from monkeyDomain import MonkeyProblem

########## PROBLEM CONSTRUCTION: Python --> PDDL ###################################################

def get_PDDL_problem():
    """ Generate all the parts of a PDDL problem """  
    
    mp = MonkeyProblem( 
        supports   = ['FLOOR', 'CHAIR'],
        locations  = ['NORTH', 'EAST', 'SOUTH', 'WEST', 'CENTER'],
        elevations = ['LO', 'HI'],
        fruitNames = ['BANANA']
    )

    mp.generate_domain_pddl()

    mp.generate_problem_pddl(
        init = { 'initDict':{
            'support_at':[
                ('CHAIR', 'NORTH', ),
                ('FLOOR', 'NORTH', ),
                ('FLOOR', 'EAST',  ),
                ('FLOOR', 'SOUTH', ),
                ('FLOOR', 'WEST',  ),
                ('FLOOR', 'CENTER',),
            ],
            'monkey_at':[
                ('SOUTH', 'LO', ),
            ],
            'fruit_at':[
                ('BANANA', 'CENTER', 'HI',),
            ],
            'support_height':[
                ('FLOOR', 'LO',),
                ('CHAIR', 'HI',),
            ],}
        },
        goal = { 'goalDict':{ 'hungry':False, } }
    )
    return mp

def solve_PDDL_problem():
    """ Ask Fast Downward to solve the problem """
    cmd = """./downward/fast-downward.py domain.pddl problem.pddl --evaluator "hff=ff()" --search "lazy_greedy([hff], preferred=[hff])" """
    subprocess.run( 
        cmd, 
        shell  = True, 
        # stdout = subprocess.DEVNULL 
    )



########## MAIN ####################################################################################

if __name__ == "__main__":
    
    prob = get_PDDL_problem()
    solve_PDDL_problem()