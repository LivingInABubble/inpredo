from glob import glob

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm


def predict(model, file):
    img_width, img_height = 150, 150
    x = load_img(file, target_size=(img_width, img_height))
    x = np.expand_dims(img_to_array(x), axis=0)

    array = model.predict(x)
    result = array[0]
    if result[0] > result[1]:
        if result[0] > 0.9:
            # print("Predicted answer: Buy")
            answer = 'buy'
            # print(result)
            # print(array)
        else:
            # print("Predicted answer: Not confident")
            answer = 'n/a'
            # print(result)
    else:
        if result[1] > 0.9:
            # print("Predicted answer: Sell")
            answer = 'sell'
            # print(result)
        else:
            # print("Predicted answer: Not confident")
            answer = 'n/a'
            # print(result)

    return answer


def main():
    tb, ts, fb, fs, na = 0, 0, 0, 0, 0

    # os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
    # weights_path = '../src/models/weights'
    model = load_model('../src/models/model.h5')

    print("Label: buy")
    for filepath in tqdm(glob('../data/test/buy/*')):
        result = predict(model, filepath)
        if result == "buy":
            tb += 1
        elif result == 'n/a':
            # print('no action')
            na += 1
        else:
            fb += 1

    print("Label: sell")
    for filepath in tqdm(glob('../data/test/sell/*')):
        result = predict(model, filepath)
        if result == "sell":
            ts += 1
        elif result == 'n/a':
            # print('no action')
            na += 1
        else:
            fs += 1

    """
    Check metrics
    """
    print("True buy: ", tb)
    print("True sell: ", ts)
    print("False buy: ", fb)  # important
    print("False sell: ", fs)
    print("No action", na)

    precision = (tb + ts) / (tb + ts + fb + fs)
    recall = tb / (tb + fs)
    print("Precision: ", precision)
    print("Recall: ", recall)

    f_measure = (2 * recall * precision) / (recall + precision)
    print("F-measure: ", f_measure)


if __name__ == '__main__':
    main()
