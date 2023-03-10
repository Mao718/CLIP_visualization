#!/usr/bin/env python
# coding: utf-8

# In[40]:


import torch
import clip
from PIL import Image
from tqdm import trange
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from matplotlib import pyplot as plt
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision.transforms.functional as F
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go


# In[25]:


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device)
model = model.float()


# In[26]:


class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def custom_preprocess():
    return Compose([
        # SquarePad(),
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def gradcam_preprocess():
    return Compose([
        # SquarePad(),
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
    ])

# preprocess = custom_preprocess()
raw_preprocess = gradcam_preprocess()


# In[27]:


n = 1024
def image_feat(raw_image):
    image = preprocess(raw_image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)

    saliency_layer = "layer4"
    cam = GradCAM(model=model.visual, target_layers=getattr(model.visual, saliency_layer), use_cuda=True)

    grayscale_cam = []
    for i in trange(n):
        target = [ClassifierOutputSoftmaxTarget(i)]
        map = cam(input_tensor=image, targets=target)[0, :]
        grayscale_cam.append(map)
        # visualization = show_cam_on_image(image[0,:], grayscale_cam, use_rgb=True)
    return  grayscale_cam,image


# In[30]:


def visualise(caption, grayscale_cam,image):
    c = len(caption)
    
    text = clip.tokenize(caption).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

    fig = plt.figure(figsize=(16, 9), dpi=196)
    fig.clear()
    ax = fig.subplots(1, c+1)
    ax[0].imshow(raw_preprocess(image))

    # for i in range(n):
    #     ax[i+1].imshow(raw_preprocess(raw_image))
    #     ax[i+1].imshow(grayscale_cam[i], cmap='jet', alpha=0.5)
    #     ax[i+1].set(title=f"Embedding {i}")
    # fig.show()

    for i in range(c):
        map = np.zeros((224,224))
        feat = text_features[i, :].cpu().numpy()
        for j in range(n):
            map += feat[j]*grayscale_cam[j]
        x = map
        x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
        map = x_norm
        print(map.max(), map.min())
        ax[i+1].imshow(raw_preprocess(raw_image))
        ax[i+1].imshow(map, cmap='jet')   
        ax[i+1].set(title=f"{caption[i]} {probs[0, i]*100:.2f}%")


# In[192]:


def px_visualise(caption, grayscale_cam,image):
    c = len(caption)
    
    text = clip.tokenize(caption).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
    image_view = raw_preprocess(image).permute(0,2,3,1).cpu().numpy()[0]
    x = image_view 
    x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
    image_view = x_norm*255
    print(image_view.max() )
    fig = make_subplots(rows=1, cols=c+1,subplot_titles=['Ture Image']+caption)
    fig.add_trace( px.imshow(image_view ).data[0],  row=1, col=1)
    #ax = fig.subplots(1, c+1)
    #ax[0].imshow(raw_preprocess(raw_image))

    # for i in range(n):
    #     ax[i+1].imshow(raw_preprocess(raw_image))
    #     ax[i+1].imshow(grayscale_cam[i], cmap='jet', alpha=0.5)
    #     ax[i+1].set(title=f"Embedding {i}")
    # fig.show()

    all_maps = []
    feat = text_features.cpu().numpy()
    for i in range(c):
        map = np.zeros((224,224))
        for j in range(n):
            map += feat[i, j]*grayscale_cam[j]
        all_maps.append(map)
    all_maps = np.array(all_maps)

    x_norm = (all_maps-np.min(all_maps))/(np.max(all_maps)-np.min(all_maps))

    for x in range(c):

        map = x_norm[x]
        print(map.max(), map.min())
        #px.imshow(raw_preprocess(raw_image))
        #print(map)
        #overlap_fig = go.Figure([
        #go.Image(name='raccoon', z=np.array(raw_preprocess(raw_image)), opacity=1), # trace 0
        #go.Image(name='noise', z=map*255, opacity=1)  # trace 1
        #    ])
        #fig1 = go.Figure()
        #fig1.add_trace( px.imshow().data[0])
        #fig1.add_trace( px.imshow().data[0])
#         fig1 = go.Figure([
#             go.Image(name='raccoon', z=np.array(raw_preprocess(raw_image)), opacity=1), # trace 0
#             go.Heatmap(name='noise', z=map*255., opacity=0.5)  # trace 1
#         ])
        #fig1.show()
        
        fig.add_trace( go.Image(name='raccoon', z=image_view, opacity=1),  row=1, col=i+2)
        fig.add_trace( go.Heatmap(name='noise', z=map*255., opacity=0.7),  row=1, col=i+2)
        #fig.add_trace( ,  row=1, col=i+2)
        #ax[i+1].imshow(raw_preprocess(raw_image))
        #ax[i+1].imshow(map, cmap='jet')   
        #ax[i+1].set(title=f"{caption[i]} {probs[0, i]*100:.2f}%")
    fig.update_layout(height=600, width=1600)
    return fig,probs


# In[39]:


if __name__ == "__main__":
    raw_image = Image.open("s.jpg")
    feat ,image = image_feat(raw_image)
    caption = list(set(["bird" , "cat", ""]))
    px_visualise(caption, feat,image)


# In[ ]:




