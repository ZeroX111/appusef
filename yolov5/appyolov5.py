import streamlit as st
import numpy as np
from streamlit_webrtc import (
    AudioProcessorBase,
    ClientSettings,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
import av
import queue
from typing import List, NamedTuple
import torch
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box,save_one_box2
from utils.torch_utils import select_device, smart_inference_mode

WEBRTC_CLIENT_SETTINGS = ClientSettings(     # webrtc_streamer配置
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},   #配置云服务器
    media_stream_constraints={"video": True, "audio": True},
)

# Official Model
# MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Custom Model
# MODEL = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/ubuntu/yolov5/best.pt')
weights='runs/train/exp/weights/best.pt'  # model path or triton URL
device = select_device('')
imgsz=(640, 640)


#需要换掉
MODEL = DetectMultiBackend(weights, device=device, dnn=False, data='data/coco128.yaml', fp16=False)



stride, names, pt = MODEL.stride, MODEL.names, MODEL.pt
imgsz = check_img_size(imgsz, s=stride)  # check image size
MODEL.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup


def main():
    """Object detection demo with PyTorch
    based on https://github.com/whitphx/streamlit-webrtc/blob/master/app.py
    """
    class Detection(NamedTuple):
        name: str
        prob: float

    class YOLOv5VideoProcessor(VideoProcessorBase):
        result_queue: "queue.Queue[List[Detection]]"

        def __init__(self) -> None:
            global MODEL
            self._model = MODEL
            self.result_queue = queue.Queue()

        def _annotate_image(self, image, results):  #结果存储函数
            # loop over the detections
            (h, w) = image.shape[:2]
            result: List[Detection] = []

            for _, s in results.pandas().xyxy[0].iterrows():
                result.append(Detection(name=s['name'], prob=s['confidence']))

            results.render()
            image = results.imgs[0]
            
            return image, result

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="rgb24")
            results = self._model(image)
            annotated_image, result = self._annotate_image(image, results)

            # NOTE: This `recv` method is called in another thread,
            # so it must be thread-safe.
            self.result_queue.put(result)

            return av.VideoFrame.from_ndarray(annotated_image, format="rgb24")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
        video_processor_factory=YOLOv5VideoProcessor,
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
                if webrtc_ctx.video_processor:
                    try:
                        result = webrtc_ctx.video_processor.result_queue.get(
                            timeout=1.0
                        )
                    except queue.Empty:
                        result = None
                    labels_placeholder.table(result)
                else:
                    break

    st.markdown(
        "This demo uses code from "
        "https://github.com/whitphx/streamlit-webrtc/blob/master/app.py. "
        "Many thanks to the project."
    )

if __name__ == "__main__":
    main()