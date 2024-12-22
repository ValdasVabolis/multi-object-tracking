import torch

# # Path to the downloaded .pth.tar file
# pth_tar_path = '../models/shufflenet-dp.pth.tar'

# # Load the checkpoint
# checkpoint = torch.load(pth_tar_path, map_location=torch.device('cpu'))

# # Extract the model's state dictionary
# if 'state_dict' in checkpoint:
#     state_dict = checkpoint['state_dict']  # Some .pth.tar files use 'state_dict'
# else:
#     state_dict = checkpoint  # Directly use if it's just the weights

# # Save the state_dict as a .pt file
# torch.save(state_dict, 'reid_model.pt')

# print("Model successfully converted to reid_model.pt")
checkpoint = torch.load("../models/shufflenet-dp.pth.tar", map_location=torch.device("cpu"))
print(checkpoint.keys())

