# Adapted from code by Eric Appelt: https://gist.github.com/appeltel/fd3ddeeed6c330c7208502462639d2c9

########## INIT ####################################################################################

import asyncio
import random
from collections import namedtuple



########## UTILITY CLASSES #########################################################################
Msg = namedtuple( "origin", "message" )

class Hub:
    """ Distributes each message to the queue of each all subscription """

    def __init__( self ):
        """ Maintain subscriptions as a set """
        self.queue = asyncio.Queue() # Input from nodes
        self.nodes = {} # ------------ All nodes on the network

    def publish( self, message ):
        """ For each queue referenced, push message """
        for name, node in self.nodes.items():
            if message.origin != name:
                node.queue.put_nowait( message )

    def process_input( self ):
        while not self.queue.empty():
            msg = self.queue.get_nowait()
            self.publish( msg )


class Node:
    """ Both publishes and consumes """

    def __init__( self, name, hub ):
        """ Connect to `hub` and init `queue` """
        self.name  = name
        self.hub   = hub # ----------- Reference to message distributor
        self.queue = asyncio.Queue() # Personal message queue

    def __enter__( self ):
        """ Add own queue to the hub when this subscription enters _____ """
        # FIXME: WHEN IS THIS CALLED?
        self.hub.nodes.add( self.queue )
        return self.queue

    def __exit__( self, type, value, traceback ):
        """ Remove own queue to the hub when this subscription exits """
        self.hub.nodes.remove( self.queue )
        print( "Node has exited!\n", type, '\n', value, '\n', traceback, '\n' )

    def send( self, msgStr ):
        """ Send message text back to the hub """
        message = Msg( self.name, msgStr )
        self.hub.queue.put_nowait( message )

    async def recv( self ):
        """ Wait for one message text from the hub """
        msg    = await self.queue.get()
        return msg.message



########## PUBLISHER & SUBSCRIBER PROCESSES ########################################################

async def manage_shell_subprocess( name, hub, cmd ): #, stdout_cb, stderr_cb ):
    """ Kick off a subprocess and handle messages to and from that porcess """
    process = await asyncio.create_subprocess_shell( # FIXME: IF THIS HANGS, DOES EXEC NOT HANG?
        cmd,
        stdout = asyncio.subprocess.PIPE, 
        stderr = asyncio.subprocess.PIPE
    )
    prcNode = Node( name, hub )
    msgText = ""
    while True:
        # FIXME, START HERE: MANAGE THE PROCESS
        # Wait for a message and hand it to the process
        # Check for a message from the process
        # If the process has died, then quit
        # PAUSE?
        pass
    # FIXME: REAP THE PROCESS & RETURN EXIT STATUS


async def reader( name, hub ):
    """ Function implements a Subscriber, Manages a `Subscription` """

    # Wait some random amount of time before beginning ...
    await asyncio.sleep( random.random() * 15 )
    print( f'Reader {name} has decided to subscribe now!' )

    ## Subscriber ##
    msg = ""
    # Init queue object and connect it to the `hub` object
    with Subscription( hub ) as queue:
        # While the subscriber has not been sent the poison pill
        while msg != 'SHUTDOWN':
            # Wait for a message and print it
            msg = await queue.get()
            print(f'Reader {name} got message: {msg}')
            # With a 10% chance, stop looking for messages
            if random.random() < 0.1:
                print(f'Reader {name} has read enough')
                break
    # Notify user that the subscriber has stopped
    print( f'Reader {name} is shutting down' )


async def writer( iterations, hub ):
    """ Function implements a Publisher, Sends messages to the `hub` """
    for x in range( iterations ):
        # Report number of current subscribers
        print( f'Writer: I have {len(hub.nodes)} subscribers now' )
        # Push a message with the current itertion
        hub.publish(f'Hello world - {x}')
        # Let other things happen
        await asyncio.sleep(3)
    # Poison Pill: Tell all subscribers to exit
    hub.publish('SHUTDOWN')



########## MAIN ####################################################################################
# This has the live reporting behavior that we need!

if __name__ == '__main__':
    loop    = asyncio.get_event_loop() # ---------- Init event tracker
    hub     = Hub() # ----------------------------- Init message distributor
    readers = [reader(x, hub) for x in range(10)] # Init 10 Subscribers
    # Start the event loop
    loop.run_until_complete(
        # Send a collection of jobs to the event loop
        asyncio.gather(
            writer( 10, hub ), # Ask Publisher to run 10 iterations and connect it to the hub
            *readers # --------- Add readers to the event loop collection
        )
    )