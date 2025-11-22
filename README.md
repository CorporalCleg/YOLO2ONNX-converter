## Simple YOLO2ONNX converter

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
 ```
    from inference import YOLO_ONNX
    import PIL.Image as Image


    model = YOLO_ONNX("yolov8n.onnx")
    image = Image.open("example.jpg")
    outputs = model.predict(image)
    model.visualize(image, outputs, "output.png")
 ```