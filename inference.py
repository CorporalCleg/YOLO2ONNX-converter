from onnxruntime import InferenceSession
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw

class YOLO_ONNX:
    MEANS = [0.485, 0.456, 0.406]
    STDS = [0.229, 0.224, 0.225]
    
    def __init__(self, model_path: str, threshold: float = 0.2):
        self.session = InferenceSession(model_path)
        input_shape = self.session.get_inputs()[0].shape
        self.batch_size = int(input_shape[0]) if input_shape[0] is not None else 1
        self.input_height = int(input_shape[2]) if input_shape[2] is not None else 640
        self.input_width = int(input_shape[3]) if input_shape[3] is not None else 640
        self.threshold = threshold
        # print('batch_size: ', self.batch_size)
        print('input_height: ', self.input_height)
        print('input_width: ', self.input_width)

    def _preprocess(self, image: Image.Image):
        image = image.resize((self.input_width, self.input_height))
        image = np.array(image).astype(np.float32) / 255.0
        image = ((image - self.MEANS) / self.STDS).astype(np.float32)
        image = np.transpose(image, (2, 0, 1))
        # Expand to batch dimension and pad/repeat to match expected batch size
        image = np.expand_dims(image, axis=0)
        if self.batch_size > 1:
            # Repeat the image to fill the batch
            image = np.repeat(image, self.batch_size, axis=0)
        return image

    def _postprocess(self, outputs: np.ndarray):
        # outputs cx, cy, w, h, confidence, class
        predictions = []
        # print('outputs shape: ', np.array(outputs).shape)
        for output in outputs:
            # print('output: ', output)
            x_min, y_min, x_max, y_max, confidence, cls_id = output
            # filter by confidence
            confidence = confidence > self.threshold
            # filter by class
            if confidence:
                predictions.append({"x_min": int(x_min), "y_min": int(y_min), "x_max": int(x_max), "y_max": int(y_max), "confidence": confidence, "class_id": int(cls_id)})
        return predictions

    def visualize(self, image: Image.Image, predictions: list, save_path: str="output.png"):

        for prediction in predictions:
            x_min, y_min, x_max, y_max, confidence, cls_id = prediction.values()
            draw = ImageDraw.Draw(image)
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
            draw.text((x_min, y_min - 10), f"{confidence:.2f}", fill="red")
        image.save(save_path)

    def predict(self, image: Image.Image):
        input_image = self._preprocess(image)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_image})
        # Return only the first batch item
        # print('outputs shape: ', outputs[0])
        return self._postprocess(outputs[0][0])


if __name__ == "__main__":
    model = YOLO_ONNX("yolov8n.onnx")
    image = Image.open("example.jpg")
    outputs = model.predict(image)
    model.visualize(image, outputs, "output.png")
    print(outputs)