import sys, os
import cv2


# set the video reader
video_path = 0 # camera number index
# video_path = "/home/pacific/Documents/Work/Projects/Workflows/server/PycharmProjects/Pacific_AvatarGame_Host/humanpose_2d/LiveCamera/test.mp4" # real video file
if type(video_path).__name__ == "str":
    videoReader = cv2.VideoCapture(video_path)
    print("Load live video from file...")
elif type(video_path).__name__ == "int":
    videoReader = cv2.VideoCapture(video_path)
    print("Get live video from camera...")

if videoReader.isOpened():
    print("Camera staus ready...")
else:
    print("Camera status fault...")
    exit()

video_fps = videoReader.get(cv2.CAP_PROP_FPS)
print("Live Video FPS: ", video_fps)

video_width = videoReader.get(cv2.CAP_PROP_FRAME_WIDTH)
video_height = videoReader.get(cv2.CAP_PROP_FRAME_HEIGHT)
video_size = (int(video_width), int(video_height))
print("Live Video Size: ", video_size)

# set the video writer
videoWriter = cv2.VideoWriter('./save.avi', cv2.VideoWriter_fourcc('M', 'P', '4', '2'), int(video_fps), video_size)

# read and write the video frame
while videoReader.isOpened():
    success, frame = videoReader.read()
    if success:
        # show the video frame
        print("Live Video Frame Shape: {}".format(frame.shape))
        cv2.putText(frame, "Live Camera", (470,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.namedWindow("Live Video", 0)
        cv2.imshow("Live Video", frame)

        # save the video frame
        videoWriter.write(frame)
        cv2.waitKey(20) # wait 20 ms for next frame of the live video

        # check whether manual exit command entered
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
            continue

videoReader.release()
videoWriter.release()
cv2.destroyAllWindows()
print("Live Video Done.")
