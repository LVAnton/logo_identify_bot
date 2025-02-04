import cv2
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
from PIL import Image

import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

from transformers import ViTFeatureExtractor, ViTModel
from sklearn.metrics.pairwise import cosine_distances

#Для демонстрации результатов, мы загрузим несколько групп изображений.
#Сохраним их в массивы, в которых все элементы, кроме последнего принадлежат одной компании.
#Про последний элемент мы должны сказать, является ли он элементом этой компании


# Функция, которая стандартизует размер изображений, делая их 224 на 224.
# Если изображение не квадратных пропорций, то оно изменит его размер, чтобы оно вписывалось в 224 на 224.
# И дорисует по бокам пиксели черного цвета, чтобы дать модели на вход именно квадраь 224 на 224
def resize_with_padding(image, target_size):
    h, w, _ = image.shape
    target_h, target_w = target_size
    scale_h = target_h / h
    scale_w = target_w / w
    scale = min(scale_h, scale_w)

    resized_h = int(h * scale)
    resized_w = int(w * scale)
    resized_image = cv2.resize(image, (resized_w, resized_h))

    pad_top = (target_h - resized_h) // 2
    pad_bottom = target_h - resized_h - pad_top
    pad_left = (target_w - resized_w) // 2
    pad_right = target_w - resized_w - pad_left

    padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT,
                                      value=[0, 0, 0])

    return padded_image

#Обработка изображений. Перевод в RGB и в тензор, нормализация.
transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# Функция получения эмбендингов изображения используя для этого предобуенный resNet без последнего слоя
# (последний слой отвечает за классификацию изображений, поэтому его можно удалить)

def get_embeddings(image_tensors):
    embedding_model = models.resnet50(pretrained=True)
    embedding_model = nn.Sequential(*list(embedding_model.children())[:-1])

    embedding_model.eval()
    embeddings = []
    with torch.no_grad():
        for tensor in image_tensors:
            tensor = tensor.unsqueeze(0)
            embedding = embedding_model(tensor)
            embeddings.append(embedding)

    return embeddings

#Функция для подсета косинусныого расстояния между двумя эмбендингами
from numpy.linalg import norm
def cosine_distance(embedding1, embedding2):
    embedding1 = embedding1.detach().cpu().numpy().astype(float)
    embedding2 = embedding2.detach().cpu().numpy().astype(float)
    embedding1 = embedding1.flatten()
    embedding2 = embedding2.flatten()

    cosine_similarity = np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
    cosine_distance = 1 - cosine_similarity

    return cosine_distance


# Функция для проверки, относится ли последний логотип из массива к логотипам компании.
def decide(embeddings, arr):
    summ = 0
    maxx = 0
    for i in range(len(arr) - 1):
        for j in range(i, len(arr) - 1):
            length = cosine_distance(embeddings[i], embeddings[j])
            summ += length
            if length > maxx:
                maxx = length
    count = ((len(arr) - 1) * (len(arr) - 2)) // 2
    average = summ / count

    summ2 = 0
    for i in range(len(arr) - 1):
        summ2 += cosine_distance(embeddings[i], embeddings[len(arr) - 1])
    average2 = summ2 / (len(arr) - 1)

    if average2 < average + 0.7 * (maxx - average):
        return True
    else:
        return False

'''
for i in range(len(test_arr1)):
    test_arr1[i] = resize_with_padding(test_arr1[i], (224, 224))
show_images(test_arr1)
for i in range(len(test_arr1)):
    test_arr1[i] = transform(test_arr1[i])
embeddings = get_embeddings(test_arr1)
decide(test_arr1)
'''