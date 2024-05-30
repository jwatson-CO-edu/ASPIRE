########## INIT ####################################################################################

from utils import get_merged_logs_in_dir_with_prefix


########## GET DATA ################################################################################

baseDir = "data/"
prefix  = "TAMP-Loop"
data_i  = get_merged_logs_in_dir_with_prefix( baseDir, prefix )

print( f"There are {len(data_i['trials'])} recorded trials in ./{baseDir}" )

# print( data_i['trials'][0].keys() )

# for trial in data_i['trials']:
trial = data_i['trials'][0]
for e in trial['events']:
    print( e )
# break


