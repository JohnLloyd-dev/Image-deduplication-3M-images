import sys, pathlib, numpy as np, torch, clip

txt_file = pathlib.Path(sys.argv[1])
out_file = pathlib.Path(sys.argv[2])

names = [ln.strip() for ln in txt_file.read_text(encoding="utf-8").splitlines() if ln.strip()]

device = "cpu"                       # or "cuda:0"
model, _ = clip.load("ViT-B/32", device=device)

with torch.no_grad():
    tokens = clip.tokenize(names).to(device)          # [N,77]
    txt_emb = model.encode_text(tokens)               # [N,512]
    txt_emb /= txt_emb.norm(dim=-1, keepdim=True)     # cosine normalise

np.save(out_file, txt_emb.cpu().numpy())
print(f"saved {txt_emb.shape} to {out_file}")
