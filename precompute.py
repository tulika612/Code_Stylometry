import cv2
import os
import glob
import pickle
from keras_vggface.vggface import VGGFace
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import time


def pickle_stuff(filename, stuff):
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    save_stuff.close()


def main():
    resnet50_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),
                                pooling='avg')  # pooling: None, avg or max
    resnet50_features.summary()
    def image2x(image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)  # or version=2
        return x

    def cal_mean_feature(image_folder):
        face_images = list(glob.iglob(os.path.join(image_folder, '*.*')))

        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]

        batch_size = 32
        face_images_chunks = chunks(face_images, batch_size)
        fvecs = None
        for face_images_chunk in face_images_chunks:
            images = np.concatenate([image2x(face_image) for face_image in face_images_chunk])
            batch_fvecs = resnet50_features.predict(images)
            if fvecs is None:
                fvecs = batch_fvecs
            else:
                fvecs = np.append(fvecs, batch_fvecs, axis=0)
        return np.array(fvecs).sum(axis=0) / len(fvecs)

    FACE_IMAGES_FOLDER = "datasets"
    #VIDEOS_FOLDER = "./data/videos"
    folders = list(glob.iglob(os.path.join(FACE_IMAGES_FOLDER, '*')))
    os.makedirs(FACE_IMAGES_FOLDER, exist_ok=True)
    names = [os.path.basename(folder) for folder in folders]
    

    precompute_features = []
    start = time.time()
    for i, folder in enumerate(folders):
        name = names[i]
        save_folder = os.path.join(FACE_IMAGES_FOLDER, name)
        mean_features = cal_mean_feature(image_folder=save_folder)
        print(f"crossed {name}")
        precompute_features.append({"name": name, "features": mean_features})
    end = time.time()
    pickle_stuff("data/precompute_features.pickle", precompute_features)
    print(f"time taken is {end-start}")

if __name__ == "__main__":
    main()

