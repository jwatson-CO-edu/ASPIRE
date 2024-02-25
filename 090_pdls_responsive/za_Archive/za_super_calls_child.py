class A:
    def __init__( self ):
        self.phrase = "SUPER"
    def copy( self ):
        return self.__class__()
    def say( self ):
        print( self.phrase )
        
class B( A ):
    def __init__( self ):
        super().__init__()
        self.phrase = "CHILD"
        
a = A()
b = B()
c = b.copy()

for thing in [a,b,c]:
    thing.say()