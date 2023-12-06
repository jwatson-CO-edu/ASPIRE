########## INIT ####################################################################################

##### Imports #####
import sys, time, pickle # Python 3 pickle uses cPickle
from time import sleep
now = time.time
sys.path.append( "../" )
from magpie.Camera import DepthCam, PointCloudTransmissionFormat, RGBDTransmissionFormat, PCD_JSON, RGBD_JSON
from magpie.interprocess import set_non_blocking, non_block_read, PBJSON_IO
from magpie.utils import HeartRate

##### Communication #####
set_non_blocking( sys.stdin ) # REQUIRED STEP
pbj = PBJSON_IO()
sleep( 0.100 )



########## CAMERA START ############################################################################
# Average per-cloud capture time is 0.03377411365509033 seconds!
# 2023-12-04: Conservatively, attempt to get a cloud every 50ms (0.050s)

camera = None
try:
    camera = DepthCam()
    sleep(1)
except:
    raise RuntimeError( "CANNOT connect to camera!" )



########## UTILITY FUNCTIONS #######################################################################

def check_shutdown():
    """ Return True if SHUTDOWN message was received """
    # WARNING: CONSUMES ALL MESSAGES
    msgs = pbj.recv_and_unpack( sys.stdin.buffer )
    for msg in msgs:
        # raise ValueError( str(msg) )
        if ('message' in msg) and (msg['message'] == "SHUTDOWN"):
            return True
    return False


def get_frame_message():
    """ Pickle a point cloud and the associated image """
    pc, rgbd = camera.get_PCD()
    # raise ValueError( str( dir( rgbd ) ) )
    return {
        "origin" : "camera",
        # "pc"     : pickle.dumps( PointCloudTransmissionFormat( pc ) ),
        # "rgbd"   : pickle.dumps( RGBDTransmissionFormat( rgbd )     ),
        "pc"     : PCD_JSON(  pc   ),
        "rgbd"   : RGBD_JSON( rgbd ),
    }



########## MAIN LOOP ###############################################################################

hr = HeartRate( 20.0 ) # Try to mainain 20Hz

while True:

    ### INPUT ###
    # `check_shutdown` consumes all incoming messages, but we won't talk to this process except to shut it down
    if check_shutdown():
        break

    ### OUTPUT ###
    # Pack PC + depth image and immediately send
    sys.stdout.buffer.write( pbj.pack( get_frame_message() ) )
    sys.stdout.flush()

    ### PAUSE ###
    hr.sleep()


### CLEANUP ###
camera.stop() # Release camera
exit(0)
