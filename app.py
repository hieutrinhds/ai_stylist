import os
import shutil
import numpy as np
import cv2
import pickle
import torch
import torch.nn as nn
from torchvision import models as torchmodel
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage.color import gray2rgb, rgba2rgb
import skimage.io
from PIL import Image
from annoy import AnnoyIndex
import re
import streamlit as st
import matplotlib.pyplot as plt


dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
CROPPED_FOLDER = 'static/cropped'
MEDIA_FOLDER = 'media/'

# Load model
# Object detection
net = cv2.dnn.readNet(STATIC_FOLDER + "/" + "yolov4_last.weights", STATIC_FOLDER + "/" + "yolov4.cfg")
font = cv2.FONT_HERSHEY_PLAIN
# Input class name
classes = ['top', 'bottom', 'dress']
# Get Output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Preprocess an image
def load_image(img_path):
    # image loading
    """
    Read image from path
    Input : Image path
    Output: Image in array, height, width, channels
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (416, 416))
    height, width, channel = img.shape
    return img, height, width, channel

def detecting_items(img_path):
    """
    Use Yolo to extract items in Image
    Input: Image path
    OUtput: dictionary key item_id, values: image in array format
    """

    img, height, width, channel = load_image(img_path)
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    items = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))


    # Find overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.2)
    detected_items = {}
    j = 0
    for i in range(len(boxes)):
        if i in indexes:
            j += 1
            x, y, w, h = boxes[i]
            confidence = confidences[i]
            crop_img = img[y:y + h, x:x + w]
            crop_img = cv2.resize(crop_img, None, fx=1.5, fy=1.5)
            im_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            # Save image
            cropped_image_path = os.path.join(CROPPED_FOLDER, 'item' + str(j) + ".jpg")
            # cv2.imwrite(cropped_image_path, crop_img)
            detected_items['item' + str(j)] = (im_rgb, cropped_image_path)
            img_cropped_save = Image.fromarray(im_rgb)
            img_cropped_save.save(cropped_image_path)
    if len(detected_items.keys()) == 0:
        img = cv2.imread(img_path)
        img = cv2.resize(img, None, fx=1.5, fy=1.5)    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropped_image_path = os.path.join(CROPPED_FOLDER, 'item1.jpg')
        shutil.copyfile(img_path, cropped_image_path)
        detected_items['item1'] = (img, cropped_image_path)

    return detected_items

def delete_cropped():
    for filename in os.listdir(CROPPED_FOLDER):
        file_path = os.path.join(CROPPED_FOLDER, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

# Feature extraction

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# load net to extract features
cnn_model = torchmodel.resnet50(pretrained=True)
# skip last layer (the classifier)
cnn_model = nn.Sequential(*list(cnn_model.children())[:-1])
cnn_model = cnn_model.to(device)
cnn_model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), normalize
])


def process_image(img_array):
    im = img_array
    if len(im.shape) == 2:
        im = gray2rgb(im)
    if im.shape[2] == 4:
        im = rgba2rgb(im)

    im = resize(im, (256, 256))
    im = img_as_ubyte(im)
    im = transform(im)
    im = im.unsqueeze_(0)
    im = im.to(device)
    out = cnn_model(im)
    out = out.squeeze()

    feats = out.detach().numpy()
    return feats

# Approximate Nearest Neighbor
annoy_model = AnnoyIndex(2048, 'angular')
annoy_model.load(STATIC_FOLDER + '/' + 'annoy_fit.ann')

df = pd.read_csv(STATIC_FOLDER + '/' + 'filepath.csv')


def recommended_item(idx):
    recom_item_list = []
    for i in range(1, 9):
        if i != df.iloc[idx]['id']:
            recom_item_list.append(MEDIA_FOLDER + str(df.iloc[idx]['set']) + '_'
                                   + str(i) + '.jpg')
    return recom_item_list


def recomend_set(indices):
    item_set = {}
    for i, idx in enumerate(indices):
        item_set['set' + str(i + 1)] = {'base': MEDIA_FOLDER + df.iloc[idx]['filename'], 'recomend_list': recommended_item(idx)}
    return item_set


def final_recomemnd(img_path):
    detected = detecting_items(img_path)
    final_result = {}
    for item in detected.keys():
        feats = process_image(detected[item][0])
        feat_test = feats
        indices= annoy_model.get_nns_by_vector(feat_test, 5, search_k=-1, include_distances=False)
        final_result[item] = recomend_set(indices)
    return final_result



st.title('Hey! What should I wear?')
image_path = st.sidebar.selectbox('Select an image', os.listdir('test_image'))
def read_image(image_path):
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        print(image_path)
        image = None
    return image

detected_items_dict = detecting_items('test_image/' + image_path)

detected_items = []
for l in detected_items_dict.values():
    detected_items.append(l[1])


st.subheader('Your image here')
st.image(read_image('test_image/' + image_path), use_column_width=True)

st.header(f'We found {len(detected_items)} item(s) in this image')

st.subheader('Select an item to get recommendation')
selection = {'button' + str(i): False for i in range(len(detected_items))}

for i, path in enumerate(detected_items):
    image = read_image(path)
    st.image(image, width=500)
    selection[f'button{i}'] = st.checkbox('Select',key=f'button{i}')



if any(selection.values()):
    selection_index = list(selection.values()).index(True)
    # st.text(selection_index)
    st.subheader('Here are some recommends for you. Pick your favorite ones :)')

    recommend_dict = final_recomemnd(detected_items[selection_index])['item1']
    recommend_items = [v['base'] for _,v in recommend_dict.items()]
    #based_items = [v['base'] for _,v in recommend_dict.items()]
    #for i, path in enumerate(based_items):
    #    print('=====', path)
    #    image = read_image('./static/' + path)
    #    st.image(image, width=500)

    for i, s in enumerate(recommend_items):
        st.subheader(f'item {i + 1}')
        fig, ax = plt.subplots(1, len(s))
        for j, item in enumerate(s):
            try:
                image = read_image('./static/' + item)
                ax[j].imshow(image)
                ax[j].axis('off')
            except: 
                ax[j].remove()
        st.pyplot(fig)
        



