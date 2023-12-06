########## INIT ####################################################################################

##### Imports #####
### Standard ###
import os, fcntl, sys
from queue import Queue
from time import sleep
### Special ###
import pbjson



########## STREAM COMMUNICATION ####################################################################

def set_non_blocking( output ):
    ''' even in a thread, a normal read with block until the buffer is full '''
    # Original Author, Chase Seibert: https://chase-seibert.github.io/blog/2012/11/16/python-subprocess-asynchronous-read-stdout.html
    fd = output.fileno()
    fl = fcntl.fcntl( fd, fcntl.F_GETFL )
    fcntl.fcntl( fd, fcntl.F_SETFL, fl | os.O_NONBLOCK )

def non_block_read( output ):
    """ Attempt a read from an un-blocked stream, If nothing received, then return empty string """
    # Original Author, Chase Seibert: https://chase-seibert.github.io/blog/2012/11/16/python-subprocess-asynchronous-read-stdout.html
    try:
        out = output.read()
        if out is None:
            return bytes()
        return bytes( out )
    except:
        return bytes()



########## BINARY DATA #############################################################################
_CHUNK_LIM = 873816 / 2

class PBJSON_IO:
    """ Pack and Unpack Binary JSON (PBJSON) """
    _BGN_BYTES = bytearray( [42,42] )
    _END_BYTES = bytearray( [86,86] )

    def erase( self ):
        """ Init buffer """
        self.buf = bytearray()
        
    def __init__( self ):
        """ Init buffer """
        self.erase()
        self.Ique = Queue()
        self.Oque = Queue()

    def chunkify( self, packet, chunkSize = _CHUNK_LIM ):
        """ Break the `packet` into pieces of `chunkSize` or smaller """
        Nbyte = len( packet )
        if Nbyte > chunkSize:
            pktLst = []
            i      = 0
            bgn    = 0
            while bgn < Nbyte:
                i += 1
                end = min( i*chunkSize, Nbyte )
                pktLst.append( packet[ bgn : end ] )
                bgn = end
            return pktLst
        else:
            return [packet,]

    def pack( self, inpt ):
        packet = PBJSON_IO._BGN_BYTES[:]
        packet.extend( pbjson.dumps( inpt ) )
        packet.extend( PBJSON_IO._END_BYTES )
        return packet
        # return self.chunkify( packet, chunkSize )

    def output_bytes( self, *objs, **kwargs ):
        """ Pack objects into chunks and enqueue """
        if "chunkSize" in kwargs:
            chunkSize = kwargs["chunkSize"]
        else:
            chunkSize = _CHUNK_LIM
        for obj in objs:
            pckt = self.pack( obj )
            pkts = self.chunkify( pckt, chunkSize )
            for pkt in pkts:
                self.Oque.put( pkt )

    def mass_pack_and_send( self, stream, *objs, **kwargs ):
        """ Pack (possibly) many objects into packets and send each packet as (possibly) many chunks """
        if "pause" in kwargs:
            pause = kwargs["pause"]
        else:
            pause = False
        self.output_bytes( *objs, **kwargs )
        while not self.Oque.empty():
            stream.write( self.Oque.get() )
            stream.flush()
            if pause:
                sleep( pause )
    
    def input_bytes( self, inByteArr ):
        """ Copy bytes to the buffer """
        self.buf.extend( inByteArr )

    def unpack( self ):
        """ Unpack and enqueue every complete packet, Assumes that `read` operations can chop up packets """
        #  0. Init vars
        Nbytes = len( self.buf )
        #  1. If the buffer does not have enough data for a complete message, return `False`
        if Nbytes < 4:
            return False
        bx42   = 42
        bx86   = 86
        accum  = False
        packet = bytearray()
        skip   = False
        index  = -1
        found  = False
        
        #  2. For each byte in the array, Read and handle
        for i, byte in enumerate( self.buf ):
            #  3. Fetch the next byte if it exists
            if (i+1) < Nbytes:
                byte_ip1 = self.buf[i+1]
            else:
                byte_ip1 = None
            #  4. If we found the beginning of a special word last iteration, then skip this byte
            if skip:
                skip = False
            #  5. If we found the packet start in a prev iter
            elif accum:
                #  6. If the terminator word is found, Then reset packet search and enque packet
                if [byte, byte_ip1] == [bx86,bx86]:
                    accum = False
                    index = i+1
                    found = True
                    try:
                        self.Ique.put( pbjson.loads( packet ) )
                        packet = bytearray()
                    except:
                        pass
                #  7. Else continue accumulating
                else:
                    packet.append( byte )
            #  8. If the beginning word is found, then accumulate and skip
            elif [byte, byte_ip1] == [bx42,bx42]:
                accum = True
                skip  = True
        #  9. If the last packet terminated before the end of the buffer, Then retain packet fragment only
        if ((index+1) < Nbytes):
            self.buf = self.buf[ index+1: ]
        # 10. Else erase the buffer
        else:
            self.erase()
        # 11. Return whether or not new packets were found
        return found
    
    def pop( self ):
        """ Get one unpacked object, Return None if empty """
        if not self.Ique.empty():
            return self.Ique.get()
        else:
            return None
        
    def __len__( self ):
        """ Get the number of unpacked objects """
        return self.Ique.qsize()
        
    def get_all( self ):
        """ Get all unpacked objects in a list """
        rtnLst = []
        while not self.Ique.empty():
            rtnLst.append( self.Ique.get() )
        return rtnLst
    
    def recv_and_unpack( self, stream ):
        """ Perform a non-blocking read, Unpack messages, Return all messages (if any) """
        inpt = non_block_read( stream )
        self.input_bytes( inpt )
        self.unpack()
        return self.get_all()
    
    def mass_recv_and_unpack( self, stream ):
        """ Perform many reads, Unpack messages, Return all messages (if any) """
        inpt = [1,2]
        while len( inpt ):
            inpt = non_block_read( stream )
            self.input_bytes( inpt )
        self.unpack()
        return self.get_all()
