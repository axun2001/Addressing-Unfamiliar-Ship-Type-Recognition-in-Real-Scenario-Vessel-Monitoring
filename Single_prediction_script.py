import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import os
from torchvision.models import resnet34
import random
import builtins
import matplotlib.pyplot as plt

# 定义Siamese网络
class Siamese(nn.Module):
    def __init__(self, embedding_dim):
        super(Siamese, self).__init__()
        self.embedding_dim = embedding_dim
        self.resnet = resnet34(pretrained=False)
        self.resnet.fc = nn.Linear(512, embedding_dim)

    def forward_once(self, x):
        output = self.resnet(x)
        return output

    def share_weight(self):
        # 将两个相同的网络的权重共享
        layers = [self.base.layer1, self.base.layer2, self.base.layer3, self.base.layer4]
        for i in range(len(layers)):
            layers[i] = nn.Sequential(*list(layers[i].children()))
        self.base.layer1, self.base.layer2, self.base.layer3, self.base.layer4 = nn.ModuleList(layers)

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        return output1, output2, output3

model = Siamese(embedding_dim=128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load('./Siamese_0.5_acc_ship_54.0777.pth', map_location=torch.device('cuda')))

model.eval()

from sklearn.metrics.pairwise import cosine_similarity

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

support_dir = 'D:/PyTorch/OSL-pt/dataset_total/support'
support_dataset = []
for class_name in os.listdir(support_dir):
    class_dir = os.path.join(support_dir, class_name)
    if os.path.isdir(class_dir):
        support_images = []
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            support_image = Image.open(image_path).convert('RGB')
            support_tensor = transform(support_image).unsqueeze(0)
            support_tensor = support_tensor.to('cuda')
            support_images.append(support_tensor)
        # 计算支持集类别平均特征向量
        support_embeddings = [model.forward_once(support_image).detach().cpu().numpy() for support_image in
                              support_images]
        support_embedding = np.mean(support_embeddings, axis=0)
        support_dataset.append((class_name, support_embedding))

# 遍历 query 目录下的所有图像
query_dir = 'D:/PyTorch/OSL-pt/dataset_total/query'
query_image_path = 'D:/PyTorch/OSL-pt/dataset_total/query/Engineering/Engineering3.jpg' # 修改为你想要查询的图像路径
query_image = Image.open(query_image_path).convert('RGB')
query_tensor = transform(query_image).unsqueeze(0)
query_tensor = query_tensor.to('cuda')

# 提取特征向量
query_embedding = model.forward_once(query_tensor).detach().cpu().numpy()

similarities = np.array(similarities)
# print(similarities)
labels = np.argmax(similarities)
similarity = np.round(similarities[labels], decimals=8)

similarities = np.array(similarities)
max_similarity = np.max(similarities)
print("cosine_similarity：", similarities)
print(f"The maximum cosine similarity value is {max_similarity}")

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

for support_label, support_embeddings in support_dataset:
    for support_embedding in support_embeddings:
        similarity = np.dot(query_embedding, support_embedding.T) / (
            np.linalg.norm(support_embedding) * np.linalg.norm(query_embedding))
        similarity = similarity.item()
        ax.scatter(support_embedding[0], support_embedding[1], label=support_label)
        similarity_str = f'{support_label} ({similarity*100:.5f}%)'
        ax.annotate(similarity_str, (support_embedding[0], support_embedding[1]))


# 根据 query_embedding 的形状选择正确的索引
if query_embedding.shape[1] == 1:
    ax.scatter(query_embedding[0], similarity_score, color='red', label='Query')
else:
    ax.scatter(query_embedding[0][0], query_embedding[0][1], color='red', label='Query')

ax.annotate(f'Query', (query_embedding[0][0], query_embedding[0][1]))

ax.legend()
ax.set_title('Query and Support Similarity')

plt.show()

