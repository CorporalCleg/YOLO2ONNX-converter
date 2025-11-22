import cv2
import numpy as np
from onnxruntime import InferenceSession

class YOLO_ONNX:
    def __init__(self, model_path: str, mode: str = 'detection', threshold: float = 0.25, iou: float = 0.7):
        self.session = InferenceSession(model_path)
        input_shape = self.session.get_inputs()[0].shape
        self.batch_size = int(input_shape[0]) if input_shape[0] is not None else 1
        self.input_height = int(input_shape[2]) if input_shape[2] is not None else 640
        self.input_width = int(input_shape[3]) if input_shape[3] is not None else 640
        self.threshold = threshold
        self.iou = iou
        self.mode = mode
        print(f"Model Input: {self.input_width}x{self.input_height}")

    def letterbox(self, img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw, dh = dw / 2, dh / 2  # divide padding into 2 sides

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, (r, (dw, dh))

    def preprocess(self, img: np.ndarray):
        img, (ratio, (dw, dh)) = self.letterbox(img, new_shape=(self.input_height, self.input_width))
        
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255.0
        img = img[None] # add batch dimension
        
        if self.batch_size > 1:
             # Repeat the image to fill the batch
             img = np.repeat(img, self.batch_size, axis=0)

        return img, ratio, (dw, dh)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None):
        # Rescale boxes (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0]
            pad = ratio_pad[1]

        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        
        # Clip boxes
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, img0_shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, img0_shape[0])  # y1, y2
        return boxes

    def crop_mask(self, masks, boxes):
        # masks: (N, H, W), boxes: (N, 4)
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=np.float32)[None, None, :]
        c = np.arange(h, dtype=np.float32)[None, :, None]
        
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    def postprocess(self, outputs, img0_shape, ratio, pad):
        # outputs[0]: (Batch, 300, 38) -> Detections
        # outputs[1]: (Batch, 32, 160, 160) -> Proto Masks
        
        detections = outputs[0][0] # (300, 38)
        proto_masks = outputs[1][0] # (32, 160, 160)
        
        # Filter by confidence
        mask = detections[:, 4] > self.threshold
        detections = detections[mask]
        
        if len(detections) == 0:
            return []

        # Extract boxes, classes, mask coefficients
        boxes = detections[:, :4]
        scores = detections[:, 4]
        class_ids = detections[:, 5]
        mask_coeffs = detections[:, 6:]

        # Rescale boxes to original image
        boxes = self.scale_boxes((self.input_height, self.input_width), boxes, img0_shape, (ratio, pad))

        # Process masks
        # Matrix multiplication: (N, 32) @ (32, 160*160) -> (N, 160*160)
        c, mh, mw = proto_masks.shape
        masks = (mask_coeffs @ proto_masks.reshape(c, -1)).reshape(-1, mh, mw)
        masks = self.sigmoid(masks)

        # Resize masks to original image size
        # We do this by resizing to input size first (implicit in letterbox logic usually) 
        # but here we need to scale up.
        # OpenCV resize doesn't handle N channels batch well, loop or use transposition
        
        full_masks = []
        for i in range(len(masks)):
            # Resize to input size (640x640) first (since masks are 160x160)
            m = cv2.resize(masks[i], (self.input_width, self.input_height))
            
            # Crop to remove padding
            # Remove padding logic reverse of letterbox
            top, left = int(pad[1]), int(pad[0])
            bottom = int(self.input_height - pad[1])
            right = int(self.input_width - pad[0])
            
            if bottom > top and right > left:
                m = m[top:bottom, left:right]
            
            # Resize to original image size
            m = cv2.resize(m, (img0_shape[1], img0_shape[0]))
            full_masks.append(m)
            
        full_masks = np.array(full_masks)
        
        # Crop masks to bounding boxes
        if len(full_masks) > 0:
             full_masks = self.crop_mask(full_masks, boxes)
        
        # Threshold masks
        full_masks = (full_masks > 0.5).astype(np.uint8)
        
        results = []
        for i in range(len(boxes)):
            # Get polygons
            contours, _ = cv2.findContours(full_masks[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons = [c.reshape(-1, 2).tolist() for c in contours if c.size >= 6]
            
            results.append({
                "bbox": {
                    "x_min": int(boxes[i][0]), "y_min": int(boxes[i][1]), 
                    "x_max": int(boxes[i][2]), "y_max": int(boxes[i][3]), 
                    "confidence": float(scores[i]), 
                    "class_id": int(class_ids[i])
                },
                "polygon": polygons
            })
            
        return results

    def predict(self, image_path: str):
        img0 = cv2.imread(image_path)
        if img0 is None:
             raise ValueError(f"Could not load image {image_path}")
             
        img, ratio, pad = self.preprocess(img0)
        
        input_name = self.session.get_inputs()[0].name
        output_names = [x.name for x in self.session.get_outputs()]
        
        outputs = self.session.run(output_names, {input_name: img})
        
        if self.mode == 'segment':
             return self.postprocess(outputs, img0.shape, ratio, pad)
        else:
             # Just return boxes part if detection mode (simplified)
             # Would need separate simple postprocess for pure detection
             pass

    def visualize(self, image_path: str, predictions: list, save_path: str="output.png"):
        img = cv2.imread(image_path)
        
        for pred in predictions:
            bbox = pred['bbox']
            polygons = pred['polygon']
            
            x1, y1, x2, y2 = bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']
            label = f"{bbox['class_id']}: {bbox['confidence']:.2f}"
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw mask/polygons
            for poly in polygons:
                pts = np.array(poly, dtype=np.int32)
                cv2.polylines(img, [pts], True, (255, 0, 0), 2)
                
                # Optional: Fill mask with transparency
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], (255, 0, 0))
                cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

        cv2.imwrite(save_path, img)

if __name__ == "__main__":
    model = YOLO_ONNX("model.onnx", mode='segment')
    image_path = "example.jpg"
    outputs = model.predict(image_path)
    model.visualize(image_path, outputs, "output.png")
    print(f"Found {len(outputs)} objects")