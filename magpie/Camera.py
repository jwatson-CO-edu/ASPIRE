########## INIT ####################################################################################

##### Imports #####

### Special ###
import time
import pyrealsense2 as rs
import numpy as np
from PIL import Image
import open3d as o3d



########## PICKLING UTILITIES ######################################################################

class PointCloudTransmissionFormat:
    """ PCD that is able to be pickled """
    # Original Author: Florian Bruggisser, https://github.com/isl-org/Open3D/issues/218#issuecomment-923016145
    def __init__( self, pointcloud ):
        self.points  = np.array( pointcloud.points  )
        self.colors  = np.array( pointcloud.colors  )
        self.normals = np.array( pointcloud.normals )

    def create_pointcloud( self ):
        pointcloud = o3d.PointCloud()
        pointcloud.points  = o3d.utility.Vector3dVector(self.points)
        pointcloud.colors  = o3d.utility.Vector3dVector(self.colors)
        pointcloud.normals = o3d.utility.Vector3dVector(self.normals)
        return pointcloud

def PCD_JSON( pcd ):
    return {
        "points" : np.array( pcd.points  ).tobytes(),
        "colors" : np.array( pcd.colors  ).tobytes(),
        "normals": np.array( pcd.normals ).tobytes(),
    }

def RGBD_JSON( rgbd ):
    return {
        "color": np.array( rgbd.color ).tobytes(),
        "depth": np.array( rgbd.depth ).tobytes(),
    }

class RGBDTransmissionFormat:
    """ PCD that is able to be pickled """
    # Original Author: Florian Bruggisser, https://github.com/isl-org/Open3D/issues/218#issuecomment-923016145
    def __init__( self, rgbd ):
        # self.dimension  = np.array( rgbd.dimension  )
        self.color      = np.array( rgbd.color      )
        self.depth      = np.array( rgbd.depth      )

    def create_rgbd( self ):
        rgbdImg = o3d.RGBDImage()
        # rgbdImg.dimension  = o3d.utility.Vector3dVector( self.dimension )
        rgbdImg.color      = o3d.utility.Vector3dVector( self.color )
        rgbdImg.depth      = o3d.utility.Vector3dVector( self.depth )
        return rgbdImg


########## INTEL REALSENSE 2 #######################################################################

class DepthCam:
    """ Simplest possible wrapper around D405 """
    
    def __init__( self, width = 1280, height = 720, zMax=0.5 ):
        """ Open and store a stream """

        self.rows = height
        self.cols = width
        self.zMax = zMax  # max distance for objects in depth images (m)
        self.extrinsics = np.eye(4)  # extrinsic parameters of the camera frame 4 x 4 numpy array
        
        # Create a context object. This object owns the handles to all connected realsense devices
        rs.align( rs.stream.color )
        self.pipeline = rs.pipeline()

        # Configure streams
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
        print( "Depth stream started!" )
        self.config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)
        print( "Color stream started!" )

        # Getting information about the connected realsense model (device object) - D405
        pipeProfile = self.config.resolve( rs.pipeline_wrapper( self.pipeline ) )
        device = pipeProfile.get_device()
        depth_sensor = device.first_depth_sensor()
        self.depthScale = depth_sensor.get_depth_scale()
        
        # Start streaming
        self.pipeline.start( self.config )

        
    def get_frames( self ):
        frames = None
        while not frames:
            frames = self.pipeline.wait_for_frames()
        return frames

    def get_depth_image( self ):
        """ Get one depth frame """
        return np.asanyarray( self.get_frames().get_depth_frame().get_data() )

    def get_pinhole_instrinsics(self, frame):
        # frame is a subclass of pyrealsense2.video_frame (depth_frame,etc)
        intrinsics = frame.profile.as_video_stream_profile().intrinsics
        return o3d.camera.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx,
                                                 intrinsics.fy, intrinsics.ppx,
                                                 intrinsics.ppy)
    
    def take_images( self, save = False ):
        # Takes RGBD Image using Realsense
        # intrinsic and extrinsic parameters are NOT applied only in get_PCD()
        # out: Open3D RGBDImage
        pipe, config = self.pipeline, self.config

        frames = pipe.wait_for_frames()
        depthFrame = frames.get_depth_frame()  # pyrealsense2.depth_frame
        colorFrame = frames.get_color_frame()

        # Sets class value for intrinsic pinhole parameters
        self.pinholeInstrinsics = self.get_pinhole_instrinsics(colorFrame)
        # asign extrinsics here if the camera pose is known
        # alignOperator maps depth frames to color frames
        alignOperator = rs.align(rs.stream.color)
        alignOperator.process(frames)
        alignedDepthFrame, alignedColorFrame = frames.get_depth_frame(), frames.get_color_frame()

        # unmodified rgb and z images as numpy arrays of 3 and 1 channels
        rawColorImage = np.array(alignedColorFrame.get_data())
        rawDepthImage = np.asarray(alignedDepthFrame.get_data())

        rawRGBDImage = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rawColorImage),
            o3d.geometry.Image(rawDepthImage.astype('uint16')),
            depth_scale=1.0 / self.depthScale,
            depth_trunc=self.zMax,
            convert_rgb_to_intensity=False)

        if save:
            subFix = str(time.time())
            np.save(f"depthImage{subFix}", rawRGBDImage.depth)
            np.save(f"colorImage{subFix}", rawRGBDImage.color)
            colorIM = Image.fromarray(rawColorImage)
            colorIM.save(f"colorImage{subFix}.jpeg")
        return rawRGBDImage

    def get_PCD( self, save = False ):
        # Takes images and returns a PCD and RGBD Image
        # Applies extrinsics and zMax
        # Downsamples PCD based on self.voxelSize
        # :save boolean that toggles whether to save data
        # out: tuple of (open3d point cloud (o3d.geometry.PointCloud),RGBDImage)
        rawRGBDImage = self.take_images()
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rawRGBDImage,
            self.pinholeInstrinsics,
            project_valid_depth_only=True,
            extrinsic=self.extrinsics
        )

        # Don't downsample
        # downsampledPCD = pcd.voxel_down_sample(voxel_size=self.voxelSize)
        if save:
            subFix = time.time()
            np.save(f"colorImage{subFix}", np.array(rawRGBDImage.color))
            np.save(f"depthImage{subFix}", np.array(rawRGBDImage.depth))
            o3d.io.write_point_cloud(f"pcd{subFix}.pcd", downsampledPCD)
        return pcd, rawRGBDImage
        # return (downsampledPCD,rawRGBDImage)

    def get_color_pc( self ):
        frames = self.get_frames()
        depFrm = frames.get_depth_frame()
        colFrm = frames.get_color_frame()
        colImg = np.asarray( colFrm.get_data() )
        print( colImg.shape )
        print( colImg[0,0,:] )
        
        pc = rs.pointcloud()
        pc.map_to( colFrm )
        points = pc.calculate( depFrm )
        vtx = np.asarray( points.get_vertices(2) )
        tex = np.asarray( points.get_texture_coordinates(2) )
        col = np.zeros( (vtx.shape[0], 3), dtype = int )

        for k, uvCoord in enumerate( tex ):
            i = int( min( uvCoord[0], 1.0) * (self.cols-1) )
            j = int( -uvCoord[1] * (self.rows-1) )
            print( uvCoord, i ,j )
            col[k,:] = colImg[j,i,:]

        return vtx, col

    def stop( self ):
        """ Stop camera and release resources """
        self.pipeline.stop()