import os
import cv2
import torch
from PIL import Image
from torchvision import models, transforms


def load_age_model(model_path: str):
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    ckpt = torch.load(model_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict_age(model, tfm, face_crop_bgr, device):
    img_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    x = tfm(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).squeeze().item()
    age = max(0.0, pred)
    age_group = int(age // 10) * 10
    return age, f"{age_group}대"


def main():
    model_path = os.getenv("AGE_MODEL_PATH", os.path.join(os.path.dirname(__file__), "age_model_batch_8_epochs_10.pt"))
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일 없음: {model_path}")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"✅ device: {device}")
    model = load_age_model(model_path).to(device)
    tfm = build_transform()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라 열기 실패")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces[:1]:
            crop = frame[y:y+h, x:x+w]
            age_val, age_group = predict_age(model, tfm, crop, device)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{age_group} ({age_val:.1f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            print(f"🧓 [Age] {age_group} (예측 {age_val:.1f})")
            break

        cv2.imshow("Age Check", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
