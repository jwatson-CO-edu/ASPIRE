import pbjson


########## BINARY DATA #############################################################################

class PBJSON_IO:
    _BGN_BYTES = bytearray( [42,42] )
    _END_BYTES = bytearray( [86,86] )
    """ Pack and Unpack Binary JSON (PBJSON) """

    def erase( self ):
        """ Init buffer """
        self.buf = bytearray()
        
    def __init__( self ):
        """ Init buffer """
        self.erase()
        self.que = Queue()

    def pack( self, inpt ):
        packet = PBJSON_IO._BGN_BYTES[:]
        packet.extend( pbjson.dumps( inpt ) )
        packet.extend( PBJSON_IO._END_BYTES )
        return packet
    
    def write( self, inByteArr ):
        """ Copy bytes to the buffer """
        self.buf.extend( inByteArr )

    def unpack( self ):
        """ Unpack and enqueue every complete packet, Assumes that `read` operations can chop up packets """
        Nbytes = len( self.buf )
        if Nbytes < 4:
            return False
        packet = bytearray()
        begun  = False
        bx42   = 42
        bx86   = 86
        index  = -1
        found  = False
        skip   = False
        for i, byte in enumerate( self.buf ):
            if skip:
                skip = False
            elif begun:
                if (byte == bx86) and ((index+1) < Nbytes) and (self.buf[index+1] == bx86):
                    try:
                        obj = pbjson.loads( packet )
                        self.que.put( obj )
                    except:
                        pass
                    packet = bytearray()
                    begun  = False
                    index  = i+1
                    found  = True
                    skip   = True
                else:
                    packet.append( byte )
            elif (byte == bx42) and ((index+1) < Nbytes) and (self.buf[index+1] == bx42):
                begun = True
                skip  = True
        if ((index+1) < Nbytes):
            self.buf = self.buf[ index+1: ]
        else:
            self.erase()
        return found
    
    def pop( self ):
        """ Get one unpacked object, Return None if empty """
        if not self.que.empty():
            return self.que.get()
        else:
            return None
        
    def __len__( self ):
        """ Get the number of unpacked objects """
        return self.que.qsize()
        
    def get_all( self ):
        """ Get all unpacked objects in a list """
        rtnLst = []
        while not self.que.empty():
            rtnLst.append( self.que.get() )
        return rtnLst