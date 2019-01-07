from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
from datetime import datetime

path = "./"

fourcc = VideoWriter_fourcc("AVI")
vid = None

outvid = "./video.avi"
fps=10
is_color=True

for day in range(10):
    for hour in range(24):
        image = path+datetime(2015,1,day,hour).strftime("%Y%m%d%h.png")

        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
