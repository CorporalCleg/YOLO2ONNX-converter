from ultralytics import YOLO
from argparse import ArgumentParser



def export(model_path: str, batch: int = 4):
    model = YOLO(model_path)
    print('model resolution: ', type(model.args['imgsz']))

    print(f"Exporting model ONNX ...")
    model.export(format="onnx", imgsz=model.args['imgsz'], dynamic=False, simplify=True, nms=True, batch=batch, opset=13)

    print(f"Model exported successfully to {model_path.replace('.pt', '.onnx')}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    export(args.model_path, args.batch)