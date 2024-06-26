import tensorflow as tf
import numpy as np
import librosa
SAVED_MODEL_PATH = "model.h5"
SAMPLES_TO_CONSIDER = 22050
class _Keyword_Spotting_Service:
    model = None
    _mapping = [
        "bird",
        "cat",
        "dog",
        "down",
        "eight",
        "five",
        "four",
        "happy",
        "house",
        "left"
    ]
    _instance = None

    def predict(self, file_path):

        # extract MFCCs
        MFCCs = self.preprocess(file_path)

        # convert 2d dimension to 4d
        # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # make prediction
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword

    def preprocess(self, file_path,num_mfcc=13, n_fft=2048, hop_length=512):
        # load signal file
        signal, sr = librosa.load(file_path)
        # drop audio files with less than pre-decided number of samples
        if len(signal) >= SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER]

        # extract the MFCCs
        MFCCs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)

        return MFCCs.T



def Keyword_Spotting_Service():

    # ensure an instance is created only the first time the factory function is called
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _Keyword_Spotting_Service._instance


if __name__ == "__main__":

    # create 2 instances of the keyword spotting service
    kss = Keyword_Spotting_Service()

    # make a prediction
    keyword = kss.predict("test/down.wav")

    print(f"Predicted keyword is : {keyword}")