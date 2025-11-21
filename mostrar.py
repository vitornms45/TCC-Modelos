import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image

model = YOLO("Modelo_Yolov11_Improve_Final.pt")
model.model.eval()

def preprocess(img_path, size=(224, 224)):
    img = Image.open(img_path).convert("RGB").resize(size)
    img_arr = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.tensor(img_arr).permute(2, 0, 1).unsqueeze(0)
    return img, img_tensor

img_path = "train\keratoconus\KCN_1_Sag_P.jpg"
img, input_tensor = preprocess(img_path)
    
last_conv_layer = None
for layer in reversed(list(model.model.modules())):
    if isinstance(layer, torch.nn.Conv2d):
        last_conv_layer = layer
        break
print("Última camada convolucional detectada:", last_conv_layer)

activations, gradients = {}, {}

def forward_hook(module, input, output):
    activations["value"] = output.detach()

def backward_hook(module, grad_input, grad_output):
    gradients["value"] = grad_output[0].detach()

handle_fwd = last_conv_layer.register_forward_hook(forward_hook)
handle_bwd = last_conv_layer.register_backward_hook(backward_hook)

model.model.train()
for param in model.model.parameters():
    param.requires_grad_(True)
input_tensor.requires_grad_(True)

output = model.model(input_tensor)
if isinstance(output, (list, tuple)):
    output = output[0]

pred = torch.softmax(output, dim=1)
pred_class = torch.argmax(pred, dim=1)
score = pred[0, pred_class]

print(f"Classe predita: {pred_class.item()} | Confiança: {score.item():.4f}")

model.model.zero_grad()
score.backward(retain_graph=True)

grads = gradients["value"]
acts = activations["value"]

weights = grads.mean(dim=(2, 3), keepdim=True)
cam = (weights * acts).sum(dim=1).squeeze()
cam = torch.relu(cam)

cam = cam - cam.min()
cam = cam / cam.max()
cam = cam.cpu().numpy()
cam = cv2.resize(cam, (224, 224))

threshold = 0.8
mask = (cam >= threshold).astype(np.uint8) * 255

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_cv = np.array(img)
highlighted = img_cv.copy()

if len(contours) > 0:
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    cv2.rectangle(highlighted, (x, y), (x + w, y + h), (255, 0, 0), 3)
else:
    print("Nenhuma região forte detectada — usando ponto máximo.")
    y, x = np.unravel_index(np.argmax(cam), cam.shape)
    box_size = 40
    x1, y1 = max(0, x - box_size // 2), max(0, y - box_size // 2)
    x2, y2 = min(224, x + box_size // 2), min(224, y + box_size // 2)
    cv2.rectangle(highlighted, (x1, y1), (x2, y2), (255, 0, 0), 3)

plt.figure(figsize=(6, 6))
plt.imshow(highlighted)
plt.title(f"Área mais importante (Classe {pred_class.item()}, Confiança {score.item():.2f})")
plt.axis("off")
plt.show()

handle_fwd.remove()
handle_bwd.remove()