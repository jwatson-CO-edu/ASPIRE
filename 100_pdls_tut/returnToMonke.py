import subprocess
from monkeyDomain import MonkeyProblem

########## PROBLEM CONSTRUCTION: Python --> PDDL ###################################################

def get_PDDL_problem():
    """ Generate all the parts of a PDDL problem """  
    
    mp = MonkeyProblem( 
        locations  = ['NORTH', 'EAST', 'SOUTH', 'WEST', 'CENTER'],
        fruitNames = ['BANANA']
    )

    mp.generate_domain_pddl()

    mp.generate_problem_pddl(
        init = { 'initDict':{
            'chair_at':[
                'NORTH',
            ],
            'monkey_at':[
                'SOUTH',
            ],
            'fruit_at':[
                ('BANANA', 'CENTER',),
            ],
        },},
        goal = { 'goalDict':{ 'hungry':False, } }
    )
    return mp

def solve_PDDL_problem():
    """ Ask Fast Downward to solve the problem """
    cmd = """./downward/fast-downward.py domain.pddl problem.pddl --evaluator "hff=ff()" --search "eager_greedy([hff], preferred=[hff])" """
    subprocess.run( 
        cmd, 
        shell  = True, 
    )



########## MAIN ####################################################################################

if __name__ == "__main__":
    
    prob = get_PDDL_problem()
    solve_PDDL_problem()