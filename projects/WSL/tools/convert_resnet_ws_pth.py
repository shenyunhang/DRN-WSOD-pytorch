import logging
import sys
import torch

logger = logging.getLogger(__name__)

in_path = sys.argv[1]
out_path = sys.argv[2]

in_pth = torch.load(in_path)
print(in_pth.keys())
in_pth = in_pth["state_dict"]
print(in_pth.keys())

out_pth = dict()
for k in list(in_pth.keys()):
    if "module.neck.fc" in k:
        k_new = k.replace("module.neck.fc", "roi_heads.box_head.fc")
    elif "module.backbone." in k:
        k_new = k.replace("module.backbone.", "backbone.")
    elif "module.neck" in k:
        k_new = k.replace("module.neck.", "roi_heads.box_head.")
    else:
        print("Unknown k pattern:", k)
        k_new = k
    out_pth[k_new] = in_pth[k]
    print("{} \t-->\t {}".format(k, k_new))
print(out_pth.keys())

torch.save(out_pth, out_path)
