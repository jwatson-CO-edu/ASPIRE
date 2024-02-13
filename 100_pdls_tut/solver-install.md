# PDDL Interface + Solver

## PDDL
1. `python3.9 -m pip install git+https://github.com/remykarem/py2pddl#egg=py2pddl --user`
1. `python3.9 -m py2pddl.init <domain name>.py`
1. Fill in the domain, init conditions, and the problem to solve
1. `python3.9 -m py2pddl.parse <domain name>.py`

## Fast Downward
1. `git clone https://github.com/aibasel/downward.git`
1. `cd downward/`
1. `python3.9 ./build.py`
1. `./downward/fast-downward.py domain.pddl problem.pddl \
--evaluator "hff=ff()" \
--search "lazy_greedy([hff], preferred=[hff])"`