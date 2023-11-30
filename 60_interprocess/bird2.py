import sys
from time import sleep
from random import random

import os, time, fcntl, subprocess
from threading import Thread
from time import sleep
now = time.time

def non_block_read( output ):
    ''' even in a thread, a normal read with block until the buffer is full '''
    fd = output.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    try:
        out = output.read()
        if out is None:
            return ''
        return out
    except:
        return ''

for _ in range(100):
    inpt = non_block_read( sys.stdin )
    if len( inpt ):
        if (int( inpt ) % 2 == 0):
            print( "Squawk!" )
            sys.stdout.flush()
        else:
            print( "Chirp!" )
            sys.stdout.flush()
    else:
        print( "NO DATA!" )
        sys.stdout.flush()
    sleep( 0.050 )

exit(0)