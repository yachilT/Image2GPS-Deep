import torch
model = torch.hub.load("serizba/salad", "dinov2_salad")
model.eval()
model.cuda()