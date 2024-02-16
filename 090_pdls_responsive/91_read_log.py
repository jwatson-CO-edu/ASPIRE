import pickle
from pprint import pprint

with open( 'statistics/py3/magpie-tamp.pkl', 'rb' ) as f:
    dct = pickle.load( f )
# pprint( dct )
for k,v in dct.items():
    print( k, '\t', v['successes']/v['calls'],'\t', v['successes']/v['overhead'] )