import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm


class ModelWrapper:
    def __init__(self, model_name='vit_base_patch16_224', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = timm.create_model(model_name, pretrained=True)
        # replace head with 2-class classifier (untrained) - using pretrained features
        in_features = self.model.head.in_features if hasattr(self.model, 'head') else None
        if in_features is not None:
            self.model.head = torch.nn.Linear(in_features, 2)
        self.model.to(self.device).eval()
        # class order matches alphabetical folder names used during training
        self.class_names = ['fake', 'real']

        # Attempt to auto-load a fine-tuned checkpoint if present
        try:
            ckpt_path = os.path.join(os.path.dirname(__file__), 'model_finetuned.pth')
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=self.device)
                if 'model_state' in ckpt:
                    self.model.load_state_dict(ckpt['model_state'], strict=False)
                else:
                    self.model.load_state_dict(ckpt, strict=False)
                print(f'Loaded checkpoint from {ckpt_path}')
        except Exception as e:
            print('Warning: could not load checkpoint:', e)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def preprocess(self, pil_image: Image.Image):
        return self.transform(pil_image)

    def predict(self, pil_image: Image.Image):
        tensor = self.preprocess(pil_image).to(self.device)
        with torch.no_grad():
            out = self.model(tensor.unsqueeze(0))
            probs = F.softmax(out, dim=1)[0]
            score, label = torch.max(probs, dim=0)
            idx = int(label.item())
            return idx, float(score.item())

    def predict_with_tensor(self, pil_image: Image.Image):
        tensor = self.preprocess(pil_image).to(self.device)
        with torch.no_grad():
            out = self.model(tensor.unsqueeze(0))
            probs = F.softmax(out, dim=1)[0]
            score, label = torch.max(probs, dim=0)
        idx = int(label.item())
        return idx, float(score.item()), tensor.detach().cpu()
