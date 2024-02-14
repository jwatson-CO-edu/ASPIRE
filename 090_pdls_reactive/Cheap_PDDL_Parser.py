def read( filename ):
    """ Get all the text of a file """
    with open(filename, 'r') as f:
        return f.read()
    
    
def split_on_any( input, anyLst, ws = True ):
    """ Split a string on any of the chars in `anyLst`, and also include those chars in the split list """
    tokens = []
    currToken = ""

    def cache_word():
        """ Store token and clear the current token """
        nonlocal tokens, currToken
        if len( currToken ):
            tokens.append( currToken )
        currToken = ""

    for char in input:
        if ws and char.isspace():
            cache_word()
        elif char in anyLst:
            cache_word()
            tokens.append( char )
        else:
            currToken += char

    # Store remainder after the last split
    cache_word()

    return tokens


def strip_pddl_comments( text ):
    """ Remove everything between a leading ';' and the next '\n' """
    ignore = False
    rtnStr = ""
    for char in text:
        if char == '\n':
            ignore = False
        if not ignore:
            if char == ';':
                ignore = True
            else:
                rtnStr += char
    return rtnStr


def nestify_token_list( tokens, index ):
    """ Transform the token list to a list of lists """
    rtnLst = []
    N      = len( tokens )
    while index < N:
        if tokens[ index ] == '(':
            item, index = nestify_token_list( tokens, index+1 )
            rtnLst.append( item )
        elif tokens[ index ] == ')':
            return rtnLst, index+1
        else:
            rtnLst.append( tokens[ index ] )
            index += 1
    return rtnLst, index
    

def pddl_as_list( path ):
    """ Thinnest PDDL Parser Possible """
    # NOTE: This function assumes that the first non-comment character in the PDDL file is a '('
    text   = strip_pddl_comments( read( path ) )
    tokens = split_on_any( text, ['(',')'] )
    # print( tokens, '\n\n' )
    rtnLst, _ = nestify_token_list( tokens, 1 )
    return rtnLst


def get_action_defn( nestList, actName ):
    """ Fetch an action definition from the "parsed" PDDL, otherwise `None` """
    # NOTE: This function assumes the action is defined at the root level
    for elem in nestList:
        if isinstance( elem, list ):
            if elem[0] == ':action':
                if elem[1] == actName:
                    return elem[2:]
    return None