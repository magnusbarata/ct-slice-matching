import argparse
from datetime import datetime
from utils import *

import pandas as pd
from models import *
from triplet_generator import TripletGenerator
from triplet_loss import *
keras.backend.clear_session()

def mean_l2(y_true, y_pred):
    return tf.reduce_mean(tf.nn.l2_normalize(y_pred, axis=1)) # K.mean(K.l2_normalize(y_pred, axis=1))

def main(args):
    params = Params(args.settings)
    if args.data: create_dataset(args.data, params.data)

    df = pd.read_csv(params.data)
    train_gen = TripletGenerator(df.Fpath, df.MaxIndex, batch_size=params.batch_size, class_range=params.class_range)

    if create_dir(args.train_dir):
        print('continue training') #TODO
    else:
        model = MODELS[params.model]((512, 512, 1))
        if params.plot_model:
            keras.utils.plot_model(model, args.train_dir + '/model.png', show_shapes=True)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir', nargs='?', default=datetime.today().strftime('%Y%m%d_EXP'))
    parser.add_argument('--settings', default='default_settings.json')
    parser.add_argument('--data')
    main(parser.parse_args())

# df = pd.read_csv('dataset.csv')
# data_gen = TripletGenerator(df.Fpath, df.MaxIndex, batch_size=5)
#
# model = build_model((512,512,1))
# #model = build_normalized_model((512,512,1))
# model.compile(loss=get_triplet_loss(0.5, paired_chi_squared),
#               optimizer=keras.optimizers.Adam(lr=1e-5, decay=1e-4), metrics=[mean_l2])
# chkpoint = keras.callbacks.ModelCheckpoint('191212InceptionV3-chi.h5', monitor='loss', save_weights_only=False, save_best_only=True, period=1)
# logger = keras.callbacks.CSVLogger('loss-files/191212InceptionV3-chi_loss.csv')
# stopper = keras.callbacks.EarlyStopping(monitor='loss', patience=20)
# model.fit_generator(data_gen, epochs=300, verbose=1, use_multiprocessing=True, workers=2, callbacks=[chkpoint, logger, stopper])
# #model.save('191122InceptionV3-100.h5')
