from django.http import StreamingHttpResponse
from django.shortcuts import render
import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO

def index(request):
    return render(request, 'index.html')

def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def detect_defects(frame):
    
    weights = str(Path(__file__).resolve().parent.parent / 'yolov8' / 'best.pt')
    model = YOLO(weights)
    results = model(frame)
    # Process results
    # for result in results:
    #     frame = result.render()
    if results is not None:

        annotated_image = results[0].plot()
        return annotated_image
    
    return frame


def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detect_defects(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
