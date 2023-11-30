# https://chase-seibert.github.io/blog/2012/11/16/python-subprocess-asynchronous-read-stdout.html

import sys, os, socket
import datetime
import fcntl
import subprocess
from threading import Thread

def non_block_read( output ):
    ''' even in a thread, a normal read with block until the buffer is full '''
    fd = output.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    try:
        return output.read()
    except:
        return ''

def log_worker(stdout):
    ''' needs to be in a thread so we can read the stdout w/o blocking '''
    while True:
        output = non_block_read(stdout).strip()
        if output:
            # FIXME: READ OUTPUT
            pass


if __name__ == '__main__':

    mysql_process = subprocess.Popen(
        ['mysql', '--user=%s' % sys.argv[1], '--password=%s' % sys.argv[2], '--batch', '--skip-tee', '--skip-pager', '--unbuffered'],
        stdin  = subprocess.PIPE,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE
    )

    thread = Thread(target=log_worker, args=[mysql_process.stdout])
    thread.daemon = True
    thread.start()

    mysql_process.wait()
    thread.join(timeout=1)





