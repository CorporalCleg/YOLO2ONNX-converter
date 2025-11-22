## Simple YOLO2ONNX converter
It suports either detection or segmentation models

## install
```
uv sync
```

## export
```
 uv run export.py --model_path yolov8n.pt
 ```

 ## infer
 ```
 uv run inference.py   
 ```

 ## python snippet

 detection
 ```
    from inference import YOLO_ONNX

    model = YOLO_ONNX("model.onnx", mode='detection')
    image_path = "example.jpg"
    outputs = model.predict(image_path)
    model.visualize(image_path, outputs, "output.png")
    print(f"Found {len(outputs)} objects")
 ```

segmentation
```
    from inference import YOLO_ONNX

    model = YOLO_ONNX("model.onnx", mode='segment')
    image_path = "example.jpg"
    outputs = model.predict(image_path)
    model.visualize(image_path, outputs, "output.png")
    print(f"Found {len(outputs)} objects")
```