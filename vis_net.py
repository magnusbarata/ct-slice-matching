import keras
from triplet_loss import *
from utils import *

import pydicom as dcm
import numpy as np
from numpy import matlib as mb
import cv2
import matplotlib.pyplot as plt

def getPixelData(fname):
    ds = dcm.dcmread(fname)
    pixel_array = ds.pixel_array.astype('float64')
    return np.expand_dims(pixel_array, -1)


def show_image_heatmap(image, heatmap):
    fig, ax = plt.subplots()
    heatmap_img = cv2.resize(heatmap, image.shape, cv2.INTER_LINEAR)
    heatmap -= np.min(heatmap)
    heatmap /= np.max(heatmap)

    ax.set_axis_off()
    ax.imshow(image)
    ax.imshow(heatmap_img, cmap='jet', alpha=0.4)
    return fig


def compute_spatial_similarity2(model, imgA, imgB, last_cnn_name='mixed10'):
    input = np.array([imgA, imgB])
    last_cnn = model.get_layer(last_cnn_name).output
    forward_func = keras.backend.function([model.input], [last_cnn])
    cv = forward_func([input])[0]

    conv1 = cv[0].reshape(-1,cv[0].shape[-1])
    conv2 = cv[1].reshape(-1,cv[1].shape[-1])
    pool1 = np.mean(conv1,axis=0)
    pool2 = np.mean(conv2,axis=0)
    out_sz = (int(np.sqrt(conv1.shape[0])),int(np.sqrt(conv1.shape[0])))
    conv1_normed = conv1 / np.linalg.norm(pool1) / conv1.shape[0]
    conv2_normed = conv2 / np.linalg.norm(pool2) / conv2.shape[0]
    im_similarity = np.zeros((conv1_normed.shape[0],conv1_normed.shape[0]))
    for px in range(conv1_normed.shape[0]):
        rep = mb.repmat(conv1_normed[px,:],conv1_normed.shape[0],1)
        im_similarity[px,:] = np.multiply(rep,conv2_normed).sum(axis=1)
    similarity1 = np.reshape(np.sum(im_similarity,axis=1),out_sz)
    similarity2 = np.reshape(np.sum(im_similarity,axis=0),out_sz)

    return similarity1, similarity2


def normalize(alpha):
    alpha = alpha.reshape(2, -1, alpha.shape[-1])
    beta = np.mean(alpha, axis=1)
    alpha1_normed = alpha[0] / np.linalg.norm(beta[0]) / alpha[0].shape[0]
    alpha2_normed = alpha[1] / np.linalg.norm(beta[1]) / alpha[1].shape[0]
    return alpha1_normed, alpha2_normed


def compute_spatial_similarity(model, imgA, imgB, last_cnn_name='mixed10'):
    input = np.array([imgA, imgB])
    last_cnn = model.get_layer(last_cnn_name).output
    forward_func = keras.backend.function([model.input], [last_cnn])
    last_cnn_out = forward_func([input])[0]
    out_sz = last_cnn_out.shape[1:-1]

    a1_normed, a2_normed = normalize(last_cnn_out)
    decomposed_sim = np.zeros((a1_normed.shape[0], a2_normed.shape[0]))
    for px in range(a1_normed.shape[0]):
        rep = mb.repmat(a1_normed[px,:], a1_normed.shape[0], 1)
        decomposed_sim[px,:] = np.multiply(rep, a2_normed).sum(axis=1)

    sim1 = np.reshape(np.sum(decomposed_sim, axis=1), out_sz)
    sim2 = np.reshape(np.sum(decomposed_sim, axis=0), out_sz)
    return sim1, sim2


params = Params('JSAI_EXP/strict3_EXP/train_params.json')
model_path = 'JSAI_EXP/strict3_EXP/model.h5'
model = keras.models.load_model(model_path, custom_objects={'tf': tf, 'mean_l2': mean_l2, 'triplet_loss': get_triplet_loss(params.margin, PAIRED_DISTANCES[params.metric], params.use_slice_dist)})
#emb = model.predict(np.array([imgA, imgB]), verbose=1)

img1_path = 'data0227/ZA-H10-25_000/00000038.DCM'
img2_path = 'data0227/ZA-H10-5_000/00000038.DCM'
figname = 'vis/{}_{}_'.format(img1_path.split('/')[-2], img2_path.split('/')[-2])
img1 = getPixelData(img1_path)
img2 = getPixelData(img2_path)
hm1, hm2 = compute_spatial_similarity(model, img1, img2)
fig1 = show_image_heatmap(np.squeeze(img1), hm1)
fig2 = show_image_heatmap(np.squeeze(img2), hm2)
fig1.savefig(figname + img1_path.split('/')[-1][6:-4], transparent=True)
fig2.savefig(figname + img2_path.split('/')[-1][6:-4], transparent=True)


#model_mod = keras.Model(model.input, model.layers[-2].output)
#cnn_out = model_mod.predict(input, verbose=1)
#cnn_out.shape
#emb_mod = np.mean(cnn_out, axis=(1,2))
#np.allclose(np.mean(cnn_out.reshape(cnn_out.shape[0],-1,cnn_out.shape[-1]), axis=1), emb_mod)

#fig, ax = plt.subplots()
#ax.imshow(load_image(img_path, preprocess=False))
#plt.imshow(np.squeeze(imgA))
