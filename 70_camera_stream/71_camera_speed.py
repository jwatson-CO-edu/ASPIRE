import sys, time
from time import sleep
now = time.time
sys.path.append( "../" )
from magpie.Camera import DepthCam

if __name__ == "__main__":
    camera = DepthCam()
    Nframe = 50
    sleep(1)
    bgn = now()
    for _ in range( Nframe ):
        pc, rgbd = camera.get_PCD()
    end = now()
    print( f"Average per-cloud capture time is {(end-bgn)/Nframe} seconds!" )
    camera.stop()


""" ##### OUTPUT #####
Depth stream started!
Color stream started!
Average per-cloud capture time is 0.03377411365509033 seconds!
"""