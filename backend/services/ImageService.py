import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import gdown
# from ultralytics import YOLO
import uuid
from dotenv import load_dotenv

load_dotenv() 

class ImageService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = 152

        self.model_dir = os.path.join(os.getcwd(), "weights")
        self.model_path_1 = os.path.join(self.model_dir, "EfficientNetV2_Gridmask.pth")
        self.model_path_2 = os.path.join(self.model_dir, "EfficientNetV2_mix_cut_grid.pth")
        self.class_names_path = os.path.join(os.getcwd(), "classes.txt")
        self.model_url_1 = os.getenv("MODEL_URL_1")
        self.model_url_2 = os.getenv("MODEL_URL_2")
        # print(f"Model URL: {self.model_url}")

        os.makedirs(self.model_dir, exist_ok=True)

        # ====== TẢI MODEL ======
        if not os.path.exists(self.model_path_1) or not os.path.exists(self.model_path_2):
            print("Đang tải model từ Google Drive...")
            gdown.download(self.model_url_1, self.model_path_1, quiet=False)
            gdown.download(self.model_url_2, self.model_path_2, quiet=False)
            print("Tải model thành công!")

        # ====== LOAD CLASS NAMES ======
        if not os.path.exists(self.class_names_path):
            raise FileNotFoundError("Không tìm thấy file classes.txt!")

        with open(self.class_names_path, "r") as f:
            self.class_names = [line.strip() for line in f]

        # ====== KHỞI TẠO MODEL EfficientNetV2_Gridmask ======
        print("Đang khởi tạo EfficientNetV2_Gridmask ...")
        self.model_1 = efficientnet_v2_s(weights=None)
        self.model_1.classifier[1] = nn.Linear(
            self.model_1.classifier[1].in_features, self.num_classes
        )
        # ====== LOAD WEIGHTS ======
        print("Đang load trọng số...")
        state_dict = torch.load(self.model_path_1, map_location=self.device)
        self.model_1.load_state_dict(state_dict["model_state_dict"], strict=False)
        self.model_1 = self.model_1.to(self.device)
        self.model_1.eval()
        print("EfficientNetV2_Gridmask sẵn sàng!")

        # ====== KHỞI TẠO MODEL EfficientNetV2_mix_cut_grid ======
        print("Đang khởi tạo EfficientNetV2_Gridmask ...")
        self.model_2 = efficientnet_v2_s(weights=None)
        self.model_2.classifier[1] = nn.Linear(
            self.model_2.classifier[1].in_features, self.num_classes
        )
        # ====== LOAD WEIGHTS ======
        print("Đang load trọng số...")
        state_dict = torch.load(self.model_path_2, map_location=self.device)
        self.model_2.load_state_dict(state_dict["model_state_dict"], strict=False)
        self.model_2 = self.model_2.to(self.device)
        self.model_2.eval()
        print("EfficientNetV2_mix_cut_grid sẵn sàng!")

        # ====== TRANSFORM ======
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # ====== LOAD YOLO ======
        # print("Đang load YOLOv8 nano...")
        # self.yolo = YOLO("yolov8n.pt")
        # print("YOLO sẵn sàng!")

    # ---------------------------------------------------------
    # ========== PHÂN LOẠI ẢNH LẺ  ==========
    # ---------------------------------------------------------
    # async def detect_image(self, file_bytes: bytes):
    #     img = Image.open(BytesIO(file_bytes)).convert("RGB")
    #     img_tensor = self.transform(img).unsqueeze(0).to(self.device)

    #     with torch.no_grad():
    #         outputs = self.model(img_tensor)
    #         probs = torch.softmax(outputs, dim=1)
    #         pred_idx = torch.argmax(probs, dim=1).item()

    #     return {
    #         "predicted_class": self.class_names[pred_idx],
    #         "probability": float(probs[0][pred_idx])
    #     }

    # ---------------------------------------------------------
    # ========== PHÂN LOẠI ẢNH LẺ Ensemble  ==========
    # ---------------------------------------------------------
    async def detect_image(self, file_bytes: bytes):

        img = Image.open(BytesIO(file_bytes)).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        models = [self.model_1, self.model_2]
 
        with torch.no_grad():
            prods_sum = None
            for m in models:
                m.eval()
                logits = m(img_tensor)
                prods = torch.softmax(logits, dim=1)
                prods_sum = prods if prods_sum is None else prods_sum + prods
            # print("prods_sum:", prods_sum)
            final_probs = prods_sum / len(models)
            conf, pred_idx = torch.max(final_probs, dim=1)
        predicted_class_name = self.class_names[pred_idx]
        confidence_value = conf.item()*100

        return {
            "predicted_class": predicted_class_name,
            "probability": confidence_value
        }

    # ---------------------------------------------------------
    # ========== YOLO + PHÂN LOẠI ==========
    # ---------------------------------------------------------
    # async def detect_and_classify(self, file_bytes: bytes):

    #     # ===== Decode ảnh =====
    #     np_arr = np.frombuffer(file_bytes, np.uint8)
    #     img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    #     if img is None:
    #         raise RuntimeError("Không thể decode ảnh!")

    #     results = self.yolo(img)[0]

    #     # ===== Lặp qua từng detection =====
    #     for box in results.boxes:

    #         cls_id = int(box.cls[0])
    #         label_name = results.names[cls_id]

    #         if label_name.lower() != "bird":
    #             continue

    #         x1, y1, x2, y2 = map(int, box.xyxy[0])

    #         crop = img[y1:y2, x1:x2]
    #         crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    #         buf = BytesIO()
    #         crop_pil.save(buf, format="JPEG")
    #         buf.seek(0)

    #         result = await self.detect_image(buf.getvalue())

    #         prob = result["probability"]
    #         pred_class = result["predicted_class"]

    #         # if prob < 0.5:
    #         #     continue

    #         text = f"{pred_class} {prob*100:.1f}%"

    #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv2.putText(img, text, (x1, y1 - 10),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    #     # ===== LƯU ẢNH OUTPUT =====
    #     output_dir = "static"
    #     os.makedirs(output_dir, exist_ok=True)
    #     random_filename = f"{uuid.uuid4().hex}.jpg"
    #     output_path = os.path.join(output_dir, random_filename)
    #     cv2.imwrite(output_path, img)

    #     print(f"Ảnh kết quả đã lưu tại: {output_path}")

    #     return {
    #         "output_image_path": output_path
    #     }
