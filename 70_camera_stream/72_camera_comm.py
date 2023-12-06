########## INIT ####################################################################################

##### Imports #####
import time, subprocess, sys
from threading import Thread
from time import sleep
now = time.time
from random import choice, randrange

sys.path.append( "../" )
from magpie.interprocess import set_non_blocking, non_block_read, PBJSON_IO
from magpie.utils import HeartRate




########## THREADS #################################################################################
# https://chase-seibert.github.io/blog/2012/11/16/python-subprocess-asynchronous-read-stdout.html

def cam_connection( stdin, stdout, timeout = 10.0 ):
    ''' needs to be in a thread so we can read the stdout w/o blocking '''

    set_non_blocking( stdout )
    bgn = now()
    pbj = PBJSON_IO()
    hr  = HeartRate( 40.0 )
    Nfr = 0

    print( f"About to run for {timeout} seconds ..." )

    while (now()-bgn) < timeout:
        
        ### INPUT ###
        msgs = pbj.recv_and_unpack( stdout )
        Nfr += len( msgs )

        ### PAUSE ###
        hr.sleep()

    end = now()
        
    print( f"Got {Nfr} in {end-bgn} seconds! Rate: {Nfr*1.0/(end-bgn)}" )

    for _ in range(3):
        cObj = pbj.pack( {"message" : "SHUTDOWN"} )
        stdin.write( cObj )
        stdin.flush()
        sleep( 0.005 )
    
    return None



########## MAIN ####################################################################################

if __name__ == '__main__':

    process = subprocess.Popen(
        ['python3.9', 'camera_process.py'],
        stdin  = subprocess.PIPE,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE
    )
    sleep( 0.050 )
    print( f"Process {process.pid} started with status: {process.returncode}" )

    thread = Thread( target = cam_connection, 
                     args   = [process.stdin, process.stdout, 20.0] )
    thread.daemon = True
    thread.start()

    thread.join()
    print( "Thread ENDED!" )

    process.kill()
    process.wait()
    print( f"Process {process.pid} ended with status: {process.returncode}" )
    if process.stderr.read(1):
        print( f"Errors: {process.stderr.read()}" )
    










# Average per-cloud capture time is 0.03377411365509033 seconds!
# 2023-12-04: Conservatively, attempt to get a cloud every 50ms (0.050s)