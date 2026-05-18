"""
- take data from 2 modalities of the same instance
- embed into 2 vectors of same size
- (NT-Xent loss)
    - increase similarity of these 2 vectors
    - decrease similarity of these 2 vectors from other vectors in the batch (also affects similarity between vectors of same modality)
- symmetric cross entropy loss
    - zero shot classification by nearest neighbor in the embedding space 
"""


## for single sensor data (LoRa & RFID)
# class ConvolutionalEncoder(torch.nn.Module):
#     def __init__(self, input_dim=4, hidden_dim=16, out_dim=64, dropout_p=0.1, dtype=torch.bfloat16):
#         super().__init__()
#         self.conv1 = torch.nn.Conv1d(input_dim, hidden_dim, kernel_size=3)
#         self.maxpool1 = torch.nn.MaxPool1d(2)
#         self.conv2 = torch.nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3)
#         self.maxpool2 = torch.nn.MaxPool1d(2)
#         self.conv3 = torch.nn.Conv1d(hidden_dim * 2, out_dim, kernel_size=3)
#         self.type(dtype)

#     def forward(self, x):
#         if x.ndim == 2:
#             x = x.unsqueeze(0)

#         x = torch.nn.functional.silu(self.conv1(x))
#         x = self.maxpool1(x)
#         x = torch.nn.functional.silu(self.conv2(x))
#         x = self.maxpool2(x)
#         return x

