import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset


DATA_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DATA_DIR.parent
CLASS_NAMES = ("awake", "sleepy")
IMAGE_SIZE = 80
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


class EyeDataset(Dataset):
    def __init__(self, split_dir, training=False, samples=None):
        self.samples = list(samples) if samples is not None else []
        self.training = training

        if samples is None:
            for label, class_name in enumerate(CLASS_NAMES):
                class_dir = split_dir / class_name
                if not class_dir.is_dir():
                    raise FileNotFoundError(f"Missing class folder: {class_dir}")
                self.samples.extend(
                    (path, label)
                    for path in sorted(class_dir.iterdir())
                    if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
                )

        if not self.samples:
            raise ValueError(f"No supported images found in {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("L")
            if self.training and random.random() < 0.5:
                image = ImageOps.mirror(image)
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
            pixels = np.asarray(image, dtype=np.float32) / 255.0

        tensor = torch.from_numpy(pixels).unsqueeze(0)
        return tensor, label


class EyeStateCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, len(CLASS_NAMES)),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def evaluate(model, loader, device, loss_function, temperature=1.0):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros(len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.int64)

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images) / temperature
            loss = loss_function(outputs, labels)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            for actual, predicted in zip(labels.cpu(), predictions.cpu()):
                confusion[actual, predicted] += 1

    return total_loss / total, correct / total, confusion


def fit_temperature(model, loader, device):
    logits_batches = []
    label_batches = []
    model.eval()

    with torch.inference_mode():
        for images, labels in loader:
            logits_batches.append(model(images.to(device)).cpu())
            label_batches.append(labels.cpu())

    logits = torch.cat(logits_batches)
    labels = torch.cat(label_batches)
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = nn.functional.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


def metrics_from_confusion(confusion):
    metrics = []
    for index, class_name in enumerate(CLASS_NAMES):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum()) - true_positive
        false_negative = int(confusion[index, :].sum()) - true_positive
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        support = int(confusion[index, :].sum())
        metrics.append((class_name, precision, recall, f1_score, support))
    return metrics


def build_subject_splits(data_dir, seed):
    all_samples = []
    for split_name in ("train", "val", "test"):
        all_samples.extend(EyeDataset(data_dir / split_name).samples)

    names = [path.name for path, _ in all_samples]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate image filenames found across the existing splits.")

    grouped_samples = defaultdict(list)
    grouped_counts = defaultdict(Counter)
    for sample in all_samples:
        path, label = sample
        subject_id = path.stem.split("_", 1)[0]
        grouped_samples[subject_id].append(sample)
        grouped_counts[subject_id][label] += 1

    subject_ids = sorted(grouped_samples)
    if len(subject_ids) < 3:
        raise ValueError("At least three distinct subjects are required for grouped splits.")

    totals_by_class = [
        sum(grouped_counts[subject_id][label] for subject_id in subject_ids)
        for label in range(len(CLASS_NAMES))
    ]
    ratios = (0.70, 0.15, 0.15)
    target_counts = [
        [total * ratio for total in totals_by_class]
        for ratio in ratios
    ]
    train_subject_count = round(len(subject_ids) * ratios[0])
    val_subject_count = round(len(subject_ids) * ratios[1])
    rng = random.Random(seed)
    best_score = float("inf")
    best_groups = None

    for _ in range(5000):
        shuffled = subject_ids.copy()
        rng.shuffle(shuffled)
        groups = (
            shuffled[:train_subject_count],
            shuffled[train_subject_count : train_subject_count + val_subject_count],
            shuffled[train_subject_count + val_subject_count :],
        )
        counts = [
            [
                sum(grouped_counts[subject_id][label] for subject_id in group)
                for label in range(len(CLASS_NAMES))
            ]
            for group in groups
        ]
        if any(count == 0 for split_counts in counts for count in split_counts):
            continue
        score = sum(
            ((counts[split_index][label] - target_counts[split_index][label])
             / target_counts[split_index][label]) ** 2
            for split_index in range(len(ratios))
            for label in range(len(CLASS_NAMES))
        )
        if score < best_score:
            best_score = score
            best_groups = groups

    if best_groups is None:
        raise ValueError("Could not create subject-disjoint splits with both classes.")

    datasets = []
    for split_name, group in zip(("train", "val", "test"), best_groups):
        samples = [
            sample
            for subject_id in group
            for sample in grouped_samples[subject_id]
        ]
        datasets.append(
            EyeDataset(data_dir / split_name, training=split_name == "train", samples=samples)
        )

    return (*datasets, best_groups)


def load_classifier(checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    class_names = checkpoint["class_names"]
    model = EyeStateCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    temperature = float(checkpoint.get("temperature", 1.0))
    return model, class_names, checkpoint["image_size"], temperature


def classify_pil_image(image, model, class_names, image_size, device, temperature=1.0):
    image = image.convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor) / temperature, dim=1)[0]

    predicted_index = int(probabilities.argmax().item())
    confidence = float(probabilities[predicted_index].item())
    return class_names[predicted_index], confidence


def classify_image(image_path, model, class_names, image_size, device, temperature=1.0):
    with Image.open(image_path) as image:
        return classify_pil_image(
            image, model, class_names, image_size, device, temperature
        )


def detect_eye_regions(image):
    grayscale = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_detector.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )
    if len(faces) == 0:
        return []

    face_x, face_y, face_width, face_height = max(
        faces, key=lambda face: int(face[2]) * int(face[3])
    )
    eye_width = max(24, int(face_width * 0.27))
    eye_height = max(18, int(face_height * 0.12))
    eye_center_y = face_y + int(face_height * 0.41)
    regions = []

    for side, center_ratio in (("Mắt trái (theo ảnh)", 0.32), ("Mắt phải (theo ảnh)", 0.68)):
        center_x = face_x + int(face_width * center_ratio)
        left = max(face_x, center_x - eye_width // 2)
        right = min(face_x + face_width, center_x + eye_width // 2)
        top = max(face_y, eye_center_y - eye_height // 2)
        bottom = min(face_y + face_height, eye_center_y + eye_height // 2)
        if right > left and bottom > top:
            regions.append((side, (left, top, right, bottom)))

    return regions


def predict_image(image_path, checkpoint_path, device):
    model, class_names, image_size, temperature = load_classifier(checkpoint_path, device)
    predicted_class, confidence = classify_image(
        image_path, model, class_names, image_size, device, temperature
    )
    display_names = {"awake": "Mat mo (awake)", "sleepy": "Mat nham (sleepy)"}
    print(f"Eye state: {display_names[predicted_class]}")
    print(f"Model confidence: {confidence:.1%}")


def run_image_demo(checkpoint_path, device):
    import tkinter as tk
    from tkinter import filedialog, messagebox
    from PIL import ImageOps, ImageTk

    root = tk.Tk()
    root.title("Phân loại trạng thái mắt")
    root.geometry("1040x700")
    root.minsize(900, 620)
    root.configure(bg="#edf1ec")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    try:
        model, class_names, image_size, temperature = load_classifier(
            checkpoint_path, device
        )
    except Exception as error:
        messagebox.showerror("Khong mo duoc mo hinh", str(error), parent=root)
        root.destroy()
        return

    header = tk.Frame(root, bg="#173b35", padx=28, pady=20)
    header.grid(row=0, column=0, sticky="ew")
    tk.Label(
        header,
        text="PHÂN LOẠI TRẠNG THÁI MẮT",
        font=("Segoe UI", 18, "bold"),
        fg="#ffffff",
        bg="#173b35",
    ).pack(anchor="w")
    tk.Label(
        header,
        text="Chọn ảnh, xem vùng mắt được khoanh và kết quả phân loại",
        font=("Segoe UI", 10),
        fg="#c5ddd3",
        bg="#173b35",
    ).pack(anchor="w", pady=(4, 0))

    body = tk.Frame(root, bg="#edf1ec", padx=24, pady=24)
    body.grid(row=1, column=0, sticky="nsew")
    body.columnconfigure(0, weight=3, uniform="panels")
    body.columnconfigure(1, weight=2, uniform="panels")
    body.rowconfigure(0, weight=1)

    preview_panel = tk.Frame(body, bg="#202b29", width=590, height=500)
    preview_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
    preview_panel.grid_propagate(False)
    preview_label = tk.Label(
        preview_panel,
        text="Ảnh sẽ hiển thị ở đây",
        font=("Segoe UI", 14),
        fg="#d4dfd9",
        bg="#202b29",
    )
    preview_label.place(relx=0.5, rely=0.5, anchor="center")

    result_panel = tk.Frame(
        body, bg="#ffffff", padx=24, pady=24, highlightbackground="#d8e1da", highlightthickness=1
    )
    result_panel.grid(row=0, column=1, sticky="nsew")
    tk.Label(
        result_panel,
        text="KẾT QUẢ",
        font=("Segoe UI", 10, "bold"),
        fg="#6a7c72",
        bg="#ffffff",
    ).pack(anchor="w")
    state_label = tk.Label(
        result_panel,
        text="Chưa có ảnh",
        font=("Segoe UI", 23, "bold"),
        fg="#173b35",
        bg="#ffffff",
        wraplength=300,
        justify="left",
    )
    state_label.pack(anchor="w", pady=(14, 8))
    left_eye_label = tk.Label(
        result_panel,
        text="Mắt trái (theo ảnh): --",
        font=("Segoe UI", 12),
        fg="#16735d",
        bg="#ffffff",
        wraplength=300,
        justify="left",
    )
    left_eye_label.pack(anchor="w", pady=(2, 0))
    right_eye_label = tk.Label(
        result_panel,
        text="Mắt phải (theo ảnh): --",
        font=("Segoe UI", 12),
        fg="#16735d",
        bg="#ffffff",
        wraplength=300,
        justify="left",
    )
    right_eye_label.pack(anchor="w", pady=(4, 0))
    filename_label = tk.Label(
        result_panel,
        text="Chưa chọn ảnh",
        font=("Segoe UI", 9),
        fg="#6a7c72",
        bg="#ffffff",
        wraplength=300,
        justify="left",
    )
    filename_label.pack(anchor="w", pady=(18, 0))
    tk.Label(
        result_panel,
        text="Kết quả tham khảo. Mô hình học từ ảnh mắt hồng ngoại; ảnh selfie có thể cho dự đoán sai.",
        font=("Segoe UI", 9),
        fg="#6a7c72",
        bg="#ffffff",
        wraplength=300,
        justify="left",
    ).pack(anchor="w", pady=(20, 0))

    def select_image():
        image_path = filedialog.askopenfilename(
            parent=root,
            title="Chọn ảnh cần phân loại",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")],
        )
        if not image_path:
            return

        try:
            with Image.open(image_path) as image:
                original = ImageOps.exif_transpose(image).convert("RGB")
            eye_regions = detect_eye_regions(original)
            annotated = original.copy()
            if not eye_regions:
                state_text = "Không tìm thấy khuôn mặt"
                result_color = "#bd583f"
                left_text = "Mắt trái (theo ảnh): chưa nhận diện"
                right_text = "Mắt phải (theo ảnh): chưa nhận diện"
                left_color = right_color = "#6a7c72"
            else:
                draw = ImageDraw.Draw(annotated)
                display_names = {"awake": "Mắt mở", "sleepy": "Mắt nhắm"}
                eye_results = {}
                for side, box in eye_regions:
                    predicted_class, confidence = classify_pil_image(
                        original.crop(box), model, class_names, image_size, device, temperature
                    )
                    color = "#18b887" if predicted_class == "awake" else "#f06b4f"
                    draw.ellipse(box, outline=color, width=max(3, original.width // 300))
                    eye_results[side] = predicted_class, confidence

                state_text = "Kết quả riêng cho từng mắt"
                result_color = "#173b35"
                left_class = next(
                    (result for side, result in eye_results.items() if side.startswith("Mắt trái")),
                    None,
                )
                right_class = next(
                    (result for side, result in eye_results.items() if side.startswith("Mắt phải")),
                    None,
                )
                left_text = (
                    f"Mắt trái (theo ảnh): {display_names[left_class[0]]} "
                    f"(tin cậy {left_class[1]:.0%})"
                    if left_class else "Mắt trái (theo ảnh): chưa nhận diện"
                )
                right_text = (
                    f"Mắt phải (theo ảnh): {display_names[right_class[0]]} "
                    f"(tin cậy {right_class[1]:.0%})"
                    if right_class else "Mắt phải (theo ảnh): chưa nhận diện"
                )
                left_color = "#16735d" if left_class and left_class[0] == "awake" else "#bd583f"
                right_color = "#16735d" if right_class and right_class[0] == "awake" else "#bd583f"

            preview = annotated
            preview.thumbnail((560, 460), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(preview)
        except Exception as error:
            messagebox.showerror("Không đọc được ảnh", str(error), parent=root)
            return

        preview_label.configure(image=photo, text="")
        preview_label.image = photo
        state_label.configure(text=state_text, fg=result_color)
        left_eye_label.configure(text=left_text, fg=left_color)
        right_eye_label.configure(text=right_text, fg=right_color)
        filename_label.configure(text=Path(image_path).name)

    footer = tk.Frame(root, bg="#edf1ec", padx=24, pady=0)
    footer.grid(row=2, column=0, sticky="ew", pady=(0, 22))
    tk.Button(
        footer,
        text="CHỌN ẢNH",
        command=select_image,
        font=("Segoe UI", 11, "bold"),
        fg="#ffffff",
        bg="#16735d",
        activebackground="#125b4b",
        activeforeground="#ffffff",
        relief="flat",
        padx=22,
        pady=11,
        cursor="hand2",
    ).pack(side="left")
    tk.Label(
        footer,
        text="Chọn ảnh khác để thử",
        font=("Segoe UI", 9),
        fg="#6a7c72",
        bg="#edf1ec",
    ).pack(side="left", padx=14)

    root.mainloop()


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data_dir = args.data_dir.resolve()
    train_dataset, val_dataset, test_dataset, subject_groups = build_subject_splits(
        data_dir, args.seed
    )
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    print(f"Device: {device}")
    print(
        f"Images: train={len(train_dataset)}, val={len(val_dataset)}, "
        f"test={len(test_dataset)}"
    )
    print(
        "Subject groups: "
        f"train={len(subject_groups[0])}, val={len(subject_groups[1])}, "
        f"test={len(subject_groups[2])} (disjoint)"
    )

    model = EyeStateCNN().to(device)
    if args.smoke_test:
        images, _ = next(iter(train_loader))
        with torch.inference_mode():
            outputs = model(images.to(device))
        print(f"Smoke test passed: images={tuple(images.shape)}, outputs={tuple(outputs.shape)}")
        return

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_accuracy = correct / total
        val_loss, val_accuracy, _ = evaluate(model, val_loader, device, loss_function)
        scheduler.step(val_loss)
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_accuracy:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": list(CLASS_NAMES),
                    "image_size": IMAGE_SIZE,
                    "val_accuracy": best_val_accuracy,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    temperature = fit_temperature(model, val_loader, device)
    checkpoint["temperature"] = temperature
    torch.save(checkpoint, checkpoint_path)
    test_loss, test_accuracy, confusion = evaluate(
        model, test_loader, device, loss_function, temperature
    )
    print(f"Best validation accuracy: {checkpoint['val_accuracy']:.4f}")
    print(f"Validation-calibrated temperature: {temperature:.4f}")
    print(f"Test loss: {test_loss:.4f}; test accuracy: {test_accuracy:.4f}")
    print(f"Confusion matrix (rows=true, columns=predicted):\n{confusion}")
    for class_name, precision, recall, f1_score, support in metrics_from_confusion(confusion):
        print(
            f"{class_name}: precision={precision:.4f}, recall={recall:.4f}, "
            f"f1={f1_score:.4f}, support={support}"
        )
    print(f"Checkpoint saved to: {checkpoint_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an awake/sleepy classifier on the existing eye-image folders."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--checkpoint", type=Path, default=PROJECT_DIR / "models" / "eye_state_cnn.pt"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Load one training batch and check the model output without training.",
    )
    parser.add_argument(
        "--predict",
        type=Path,
        help="Predict the eye state for one image using the saved checkpoint.",
    )
    parser.add_argument(
        "--choose-image",
        action="store_true",
        help="Choose an eye image from a file dialog and predict its state.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    selected_device = torch.device(
        "cuda" if torch.cuda.is_available() and not arguments.cpu else "cpu"
    )
    if arguments.choose_image:
        run_image_demo(arguments.checkpoint.resolve(), selected_device)
    elif arguments.predict:
        predict_image(arguments.predict, arguments.checkpoint.resolve(), selected_device)
    else:
        train(arguments)