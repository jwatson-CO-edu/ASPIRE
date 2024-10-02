########## INIT ####################################################################################
import subprocess




########## FAST DOWNWARD ###########################################################################
def build_FD():
    process = subprocess.Popen( f"python3.10 ./pddlstream/downward/build.py", 
                                shell  = True,
                                stdout = subprocess.PIPE, 
                                stderr = subprocess.PIPE )

    # wait for the process to terminate
    out, err = process.communicate()
    out = out.decode()
    err = err.decode()
    errcode = process.returncode
    print( f"Installed FastDownward, Output: {out},\nErrors: {err},\nWith Code: {errcode}\n" )

