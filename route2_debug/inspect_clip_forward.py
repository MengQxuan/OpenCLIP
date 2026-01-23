import torch
import open_clip
from PIL import Image

CKPT = "/home/ycj/ycj/mqx/weights/open_clip_pytorch_model.bin"
IMG  = "/home/ycj/ycj/mqx/open_clip/docs/CLIP.png"

def l2norm(x):
    return x / x.norm(dim=-1, keepdim=True)

def main():
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=CKPT)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    img = Image.open(IMG).convert("RGB")
    image = preprocess(img).unsqueeze(0).to(device)

    texts = ["a logo", "a photo of a cat"]
    tokens = tokenizer(texts).to(device)

    print("== Inputs ==")
    print("image:", tuple(image.shape), image.dtype)
    print("tokens:", tuple(tokens.shape), tokens.dtype)

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=(device=="cuda")):
        imf = model.encode_image(image)
        txf = model.encode_text(tokens)

    print("\n== Encoded ==")
    print("image_features:", tuple(imf.shape), imf.dtype)
    print("text_features :", tuple(txf.shape), txf.dtype)

    imn = l2norm(imf)
    txn = l2norm(txf)

    # CLIP: logits = exp(logit_scale) * imn @ txn^T
    scale = float(model.logit_scale.exp().detach().cpu())
    logits = scale * (imn @ txn.T)
    probs = logits.softmax(dim=-1)

    print("\n== Similarity ==")
    print("logit_scale(exp):", scale)
    print("logits:", logits.detach().cpu().numpy())
    print("probs :", probs.detach().cpu().numpy())

if __name__ == "__main__":
    main()