########## INIT ####################################################################################

##### Imports #####
import multiprocessing, logging, os, sys, socket, time, signal, glob
from time import sleep
from random import randrange, random
from multiprocessing import Process, Array, Value, Pool
import numpy as np
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler

import tensorflow


########## XML-RPC Server ##########################################################################
        
class RequestHandler( SimpleXMLRPCRequestHandler ):
    """ Restrict to a particular path? """
    rpc_paths = ( '/RPC2', )


class UnreliableNumberServer:
    """ Manage XMLRPC interactions in a loop """

    ## Class Values ##
    num   = Value( 'd', -100.0 )
    lo    =   1
    hi    = 100
    tMin  =   0.5
    tMax  =   3.0
    pHang =   0.5

    @classmethod
    def wait_random( cls ):
        """ Randomly either hang indefinitely, or stall some random period in the specified range """
        if random() < cls.pHang:
            while 1:
                print( "!HANG!" )
                sleep( 1.0 )
        else:
            pause = cls.tMin + random()*(cls.tMax - cls.tMin)
            print( f"Pause for {pause} seconds!" )
            sleep( pause )

    @classmethod
    def give_number( cls ):
        """ Give a random number in the given range """
        cls.wait_random()
        return randrange( cls.lo, cls.hi+1 )
    

def create_unreliable_xmlrpc_server( configTuple ):
    """ Create an XML-RPC server that is either local or remote """
    
    # 0. Unpack config
    print( "\n Config Tuple:" , configTuple , '\n' )
    ipad = configTuple['ip']
    port = configTuple['port']
    
    # 1. Create the XML-RPC object
    server = SimpleXMLRPCServer( ( ipad, port )                    ,
                                 requestHandler = RequestHandler ,
                                 logRequests    = False          )
    
    # 2. Register the object and server query functions
    instance = UnreliableNumberServer()
    server.register_function( instance.give_number )
    server.register_introspection_functions()

    # 3. Run the server's main loop (This will be done in its own process)
    print( "XML-RPC serving randomly hanging service from", ( ipad, port ) )
    
    server.serve_forever()



########## MAIN: PROCESS MANAGEMENT + XML-RPC CLIENT ###############################################

def start_and_return_server_process():
    """ Start XML-RPC in a separate process """
    srv = multiprocessing.Process( target = create_unreliable_xmlrpc_server )
    srv.start()
    return srv.start()


def request_with_timeout( t = 10.0 ):
    


if __name__ == "__main__":

    ## Server Process ##
    srv = start_and_return_server_process()

    ## Server Process ##