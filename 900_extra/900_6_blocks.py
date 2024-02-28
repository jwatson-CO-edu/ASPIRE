########## DEV PLAN ################################################################################
"""
Goal: Demonstrate and Illustrate Object Beliefs
[ ] Gather N readings
[ ] Compute stats on readings
[ ] Render Beliefs vs Ground Truth
"""
########## INIT ####################################################################################

##### Imports #####

### Special ### 
import trimesh

### Local ###
from components import Volume
from env_config import _BLOCK_SCALE
from PLOVER import PLOVER
from PB_BlocksWorld import PB_BlocksWorld



########## HELPER FUNCTIONS ########################################################################

def get_cube_Volume( sideLen_m ):
    """ Get a mesh representing a block """
    rtnVol = Volume()
    rtnVol.mesh = trimesh.creation.box( (sideLen_m, sideLen_m, sideLen_m,) )
    return rtnVol


def color_mesh( mesh, color4i ):
    """ Color all the faces of the `mesh` uniformly """
    mesh.visual.face_colors = color4i


def get_color_cube_spawner( sideLen_m, color4i ):
    """ Return a closure that returns a cube mesh of a certain size and color """
    def rtnFunc():
        rtnCube = get_cube_Volume( sideLen_m )
        color_mesh( rtnCube.mesh, color4i )
        return rtnCube
    return rtnFunc


########## OBJECT VOLUME LOOKUP ####################################################################

meshLookup = { # Objects for the 6-block world
    'redBlock': get_color_cube_spawner( _BLOCK_SCALE, [255,  0,  0,255] ),
    'ylwBlock': get_color_cube_spawner( _BLOCK_SCALE, [255,255,  0,255] ),
    'bluBlock': get_color_cube_spawner( _BLOCK_SCALE, [  0,  0,255,255] ),
    'grnBlock': get_color_cube_spawner( _BLOCK_SCALE, [  0,255,  0,255] ),
    'ornBlock': get_color_cube_spawner( _BLOCK_SCALE, [255,128,  0,255] ),
    'vioBlock': get_color_cube_spawner( _BLOCK_SCALE, [128,  0,255,255] ),
}

def get_volume_by_label( label, lookup ):
    """ Call the `Volume` spawner function that matches the known label, otherwise return empty `Volume` """
    if label in lookup:
        return lookup[ label ]()
    else:
        return Volume()
    

########## MAIN ####################################################################################
if __name__ == "__main__":

    world  = PB_BlocksWorld()
    plover = PLOVER()

    for i in range( 100 ):
        plover.belief_update( world.full_scan_noisy() )

    objs = plover.sample_all()
    for obj in objs:
        print( obj )