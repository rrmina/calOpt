import os
import cv2
import time
import imageio

VIDEO_NAME = "video.mp4"
FRAME_SAVE_PATH = "frames/"
FRAME_BASE_FILE_NAME = "frame"
FRAME_BASE_FILE_TYPE = ".png"

def getInfo( video_path ):
    """
    Extracts the height, width
    and fps of a video
    """

    vidcap = cv2.VideoCapture( video_path )
    width = vidcap.get( cv2.CAP_PROP_FRAME_WIDTH )
    height = vidcap.get( cv2.CAP_PROP_FRAME_HEIGHT )
    fps = vidcap.get( cv2.CAP_PROP_FPS )

    return height, width, fps

def getFrames( video_path, frame_save_path ):
    """
    Extracts the frames of a videos
    and saves in the specified path
    """

    vidcap = cv2.VideoCapture(video_path)

    success, image = vidcap.read()
    count = 1
    success = True
    while success:
        cv2.imwrite(
            "{}{}{}{}".format(frame_save_path, FRAME_BASE_FILE_NAME, count, FRAME_BASE_FILE_TYPE),
            image
        )
        success, image = vidcap.read()
        count += 1

    print("Done extracting all frames")

def _make_dir( path ):
    if not os.path.exists(path):
        os.makedirs( path )

def main():
    # Print Video Info
    print( getInfo( VIDEO_NAME ) )

    # Make sure frame save path exists
    _make_dir( FRAME_SAVE_PATH )

    # Extract the frames
    getFrames( VIDEO_NAME, FRAME_SAVE_PATH )

main()  