import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer
import time
import threading

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")

    # Run inference
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net.setInput(blob)
    output = net.forward()

    h, w = image.shape[:2]

    # Convert the output array into a structured form.
    output = output.squeeze()  # (1, 1, N, 7) -> (N, 7)
    output = output[output[:, 2] >= score_threshold]
    detections = [
        Detection(
            class_id=int(detection[1]),
            label=CLASSES[int(detection[1])],
            score=float(detection[2]),
            box=(detection[3:7] * np.array([w, h, w, h])),
        )
        for detection in output
    ]

    # Render bounding boxes and captions
    for detection in detections:
        caption = f"{detection.label}: {round(detection.score * 100, 2)}%"
        color = COLORS[detection.class_id]
        xmin, ymin, xmax, ymax = detection.box.astype("int")

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(
            image,
            caption,
            (xmin, ymin - 15 if ymin - 15 > 15 else ymin + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    result_queue.put(detections)

    return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="web",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": get_ice_servers(),
        "iceTransportPolicy": "relay",
    },
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if st.checkbox("Show the detected labels", value=True):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        # NOTE: The video transformation with object detection and
        # this loop displaying the result labels are running
        # in different threads asynchronously.
        # Then the rendered video frames and the labels displayed here
        # are not strictly synchronized.
        while True:
            result = result_queue.get()
            labels_placeholder.table(result


webrtc_streamer(key="web", video_frame_callback=callback)

class VideoStream:
    # Opens a video with OpenCV from file in a thread
    def __init__(self, src, name="VideoStream", real_time=True):
        """Initialize the video stream from a video

        Args:
            src (str): Video file to process.
            name (str, default='VideoStream'): Name for the thread.
            real_time (bool, default='True'): Defines if the video is going to 
                be read at full speed or adjusted to the original frame rate.

        Attributes:
            name (str, default='VideoStream'): Name for the thread.
            stream (cv2.VideoCapture): Video file stream.
            real_time (bool, default='True'): Defines if the video is going to 
                be read at full speed or adjusted to the original frame rate.
            frame_rate (float): Frame rate of the video.
            grabbed (bool): Tells if the current frame's been correctly read.
            frame (nparray): OpenCV image containing the current frame.
            lock (_thread.lock): Lock to avoid race condition.
            _stop_event (threading.Event): Event used to gently stop the thread.

        """
        self.name = name
        self.stream = webrtc_streamer(key="web")
        self.real_time = real_time
        self.frame_rate = self.stream.get(cv2.CAP_PROP_FPS)
        self.grabbed, self.frame = self.stream.read()
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
    
    def read(self):
        if self.stopped():
            print("Video ended")
        return self.frame
        
    def start(self):
        # Start the thread to read frames from the video stream with target function update
        threading.Thread(target=self.update, daemon=True, name=self.name).start()
        return self

    def update(self):
        # Continuosly iterate through the video stream until stopped
        while self.stream.isOpened():
            if not self.stopped():
                if self.real_time:
                    self.grabbed, self.frame = self.stream.read()
                    # Wait to match the original video frame rate
                    time.sleep(1.0/self.frame_rate)
                else:
                    self.grabbed, self.frame = self.stream.read()
            else:
                return
        self.stop()

    def stop(self):
        self.lock.acquire()
        self.stream.release()
        self._stop_event.set()
        self.lock.release()

    def stopped(self):
        return self._stop_event.is_set()

class WebcamVideoStream:
    # Opens a video stream with OpenCV from a wired webcam in a thread
    def __init__(self, src, shape=None, name="WebcamVideoStream"):
        """Initialize the video stream from a video

        Args:
            src (int): ID of the camera to use. From 0 to N.
            name (str, default='WebcamVideoStream'): Name for the thread.

        Attributes:
            name (str, default='WebcamVideoStream'): Name for the thread.
            stream (cv2.VideoCapture): Webcam video stream.
            real_time (bool, default='True'): Defines if the video is going to 
                be read at full speed or adjusted to the original frame rate.
            frame_rate (float): Frame rate of the video.
            grabbed (bool): Tells if the current frame's been correctly read.
            frame (nparray): OpenCV image containing the current frame.
            lock (_thread.lock): Lock to avoid race condition.
            _stop_event (threading.Event): Event used to gently stop the thread.

        """
        self.name = name
        self.stream = webrtc_streamer(key="web")
        self.shape = shape
        if self.shape is not None:
            self.stream.set(3, shape[0])
            self.stream.set(4, shape[1])
        self.grabbed, self.frame = self.stream.read()
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
    
    def read(self):
        return self.frame
        
    def start(self):
        # Start the thread to read frames from the video stream
        threading.Thread(target=self.update, daemon=True, name=self.name).start()
        return self

    def update(self):
        # Continuosly iterate through the video stream until stopped
        while self.stream.isOpened():
            if not self.stopped():
                self.grabbed, self.frame = self.stream.read()
            else:
                return
        self.stopped

    def stop(self):
        self.lock.acquire()
        self.stream.release()
        self._stop_event.set()
        self.lock.release()

    def stopped(self):
        return self._stop_event.is_set()
