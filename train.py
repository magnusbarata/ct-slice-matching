import argparse
from datetime import datetime
from utils import *

import pandas as pd
from models import *
from triplet_generator import TripletGenerator
from triplet_loss import *

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
keras.backend.clear_session()

def plot_distance_matrix(dist, xlabel, ylabel):
    fig, ax = plt.subplots()
    im = ax.imshow(dist, cmap='hot')
    ax.set_ylabel(xlabel)
    ax.set_xlabel(ylabel)
    cbar = ax.figure.colorbar(im, ax=ax)

    return fig

def main(args):
    params = Params(args.settings)
    if args.data: create_dataset(args.data, params.data)

    if args.train:
        df = pd.read_csv(params.data)
        train_gen = TripletGenerator(df.Fpath, df.MaxIndex, batch_size=params.batch_size, class_range=params.class_range)

        if create_dir(args.train_dir):
            print('continue training') #TODO
            params = Params(args.train_dir + '/train_params.json')
            model = keras.models.load_model(args.train_dir + '/model.h5', custom_objects={'tf': tf, 'mean_l2': mean_l2, 'triplet_loss': get_triplet_loss(params.margin, PAIRED_DISTANCES[params.metric], params.use_slice_dist)})
        else:
            model = MODELS[params.model]((512, 512, 1))

            optimizer = keras.optimizers.Adam(lr=params.lr, decay=params.decay)
            if params.model == 'SiameseModel':
                train_gen.gen_siamese = True
                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            else:
                model.compile(loss=get_triplet_loss(params.margin, PAIRED_DISTANCES[params.metric], params.use_slice_dist),
                              optimizer=optimizer, metrics=[mean_l2])

            chkpoint = keras.callbacks.ModelCheckpoint(args.train_dir + '/model.h5', monitor='loss', save_weights_only=False, save_best_only=True, period=1)
            logger = keras.callbacks.CSVLogger(args.train_dir + '/loss.csv')
            stopper = keras.callbacks.EarlyStopping(monitor='loss', patience=20)
            model.fit_generator(train_gen, epochs=params.n_epochs, verbose=1, use_multiprocessing=True, workers=2, callbacks=[chkpoint, logger, stopper])
            params.save(args.train_dir + '/train_params.json')

    if args.eval:
        params = Params(args.train_dir + '/train_params.json')

        df = pd.read_csv(args.eval, usecols=['Fpath'])
        test_gen = TripletGenerator(df.Fpath, shuffle=False, batch_size=1)

        model = keras.models.load_model(args.train_dir + '/model.h5', custom_objects={'tf': tf, 'mean_l2': mean_l2, 'triplet_loss': get_triplet_loss(params.margin, PAIRED_DISTANCES[params.metric], params.use_slice_dist)})
        embeddings = model.predict_generator(test_gen, use_multiprocessing=True, verbose=1)

        print('dumping embeddings...', end='')
        data_embeddings = {}
        for i, f in enumerate(df.Fpath):
            case = '/'.join(f.split('/')[:-1])
            if case not in data_embeddings:
                data_embeddings[case] = np.empty((0, embeddings.shape[-1]))
            data_embeddings[case] = np.vstack((data_embeddings[case], embeddings[i]))
        np.save(args.train_dir + '/eval-emb.npy', data_embeddings)

        print('done!\nwriting similarity file...', end='')
        X = data_embeddings[params.caseX]
        Y = data_embeddings[params.caseY]
        if params.metric == 'euclidean': dist = euclidean_distances(X, Y)
        elif params.metric == 'cosine': dist = 1 - cosine_similarity(X, Y)
        resultX = np.argmin(dist, axis=1)
        XYname = '/{}_{}'.format(params.caseX.split('/')[-1], params.caseY.split('/')[-1])
        np.savetxt(args.train_dir + XYname + '[sim].csv', resultX+1, fmt='%u', delimiter=',')

        print('done!\nplotting distance matrix...', end='')
        fig = plot_distance_matrix(dist, params.caseY, params.caseX)
        fig.savefig(args.train_dir + XYname + '[dist].png')
        print('done!')

    if params.plot_model:
        model.summary()
        keras.utils.plot_model(model, args.train_dir + '/model.png', show_shapes=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir', nargs='?', default=datetime.today().strftime('%Y%m%d_EXP'))
    parser.add_argument('--settings', default='default_settings.json')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', nargs='?', const='new-exp/data.csv', default=False)
    parser.add_argument('--data')
    main(parser.parse_args())
