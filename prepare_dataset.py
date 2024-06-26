import librosa
import os
import json

DATASET_PATH = "dataset"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050 # librosa use 22050 samples in 1s

#  argument of preprocess_dataset dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512
def preprocess_dataset(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512):

    # dictionary where we'll store mapping, labels, MFCCs and filenames
    data = {
    "mapping": [],
    "labels": [],
    "MFCCs": [],
    "files": []
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(DATASET_PATH)):
        if DATASET_PATH is not dirpath:
            category = dirpath.split("\\")[-1]
            data['mapping'] = category
            print(f"Processing :{category}")


            for f in filenames:
                # get the file path
                file_path = os.path.join(dirpath,f)

                #load signal file
                signal ,sr =librosa.load(file_path)
                # drop audio files with less than pre-decided number of samples
                if len(signal) >= SAMPLES_TO_CONSIDER:
                    # ensure consistency of the length of the signal
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    #extract the MFCCs
                    MFCCs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    # store data
                    data["labels"].append(i-1)
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["files"].append(file_path)

                    print(MFCCs)
                    print(f"{file_path} : {i-1}")

    # saving data in json file
        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=4)

if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH, JSON_PATH)
