import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import scipy.signal as sg
import wfdb

PATH = Path("dataset")
PATH1 = Path("data1")
sampling_rate = 257

# non-beat labels
invalid_labels = ['|', '~', '!', '+', '[', ']', '"', 'x']

# for correct R-peak location
tol = 0.05


def worker(record):
    # read ML II signal & r-peaks position and labels
#    if record == 114:
#        signal = wfdb.rdrecord((PATH / record).as_posix(), channels=[1]).p_signal[:, 0]
#    else:
#        signal = wfdb.rdrecord((PATH / record).as_posix(), channels=[0]).p_signal[:, 0]
        
    
#    signal = wfdb.rdrecord((PATH / record).as_posix(), channels=[1]).p_signal[:, 0]
        
    signal = wfdb.rdrecord((PATH / record).as_posix(), channels=[7]).p_signal[:, 0]
        

    annotation = wfdb.rdann((PATH / record).as_posix(), extension="atr")
    r_peaks, labels = annotation.sample, np.array(annotation.symbol)

    # filtering uses a 200-ms width median filter and 600-ms width median filter
    baseline = sg.medfilt(sg.medfilt(signal, int(0.2 * sampling_rate)), int(0.6 * sampling_rate)-1)
    filtered_signal = signal - baseline

    # remove non-beat labels
    indices = [i for i, label in enumerate(labels) if label not in invalid_labels]
    r_peaks, labels = r_peaks[indices], labels[indices]

    # align r-peaks
    newR = []
    for r_peak in r_peaks:
        r_left = np.maximum(r_peak - int(tol * sampling_rate), 0)
        r_right = np.minimum(r_peak + int(tol * sampling_rate), len(filtered_signal))
        newR.append(r_left + np.argmax(filtered_signal[r_left:r_right]))
    r_peaks = np.array(newR, dtype="int")

    # remove inter-patient variation
    normalized_signal = filtered_signal / np.mean(filtered_signal[r_peaks])

    # AAMI categories
    AAMI = {
        "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,  # N
        "A": 1, "a": 1, "S": 1, "J": 1,  # SVEB
        "V": 2, "E": 2,  # VEB
        "F": 3,  # F
        "/": 4, "f": 4, "Q": 4    , "n": 4 , "B": 4  # Q
        
    }
    categories = [AAMI[label] for label in labels]

    return {
        "record": record,
        "signal_2": normalized_signal, "r_peaks": r_peaks, "categories": categories
    }


if __name__ == "__main__":
    # for multi-processing
    cpus = 22 if joblib.cpu_count() > 22 else joblib.cpu_count() - 1

    
    
#    train_records = [
#        'I01','I02','I03','I04','I05','I06','I07','I08','I09','I10',
#        'I11','I12','I13','I14','I15','I16','I17','I18','I19','I20',
#        'I21','I22','I23','I24','I25','I26','I27','I28','I29','I30',
#        'I31','I32','I33','I34','I35','I36','I37','I38','I39','I40',
#        'I41','I42','I43','I44','I45','I46','I47','I48','I49','I50',
#        'I51','I52','I53','I54','I55','I56','I57','I58','I59','I60',
#        'I61','I62','I63','I64','I65','I66','I67','I68','I69','I70',
#        'I71','I73','I74','I75'
#    ]
    train_records = [
        'I01','I02','I03','I04','I05','I06','I07','I08','I09','I10',
        'I11','I12','I13','I14','I15','I16','I17','I18','I19','I20',
        'I21','I22','I23','I24','I25','I26','I27','I28','I29','I30',
        'I31','I32','I33','I34','I35','I36','I37'
    ]
#    train_records = [
#        'I72'
#    ]
    
    print("train processing...")
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        train_data = [result for result in executor.map(worker, train_records)]
#    train_data=worker('I18')

    test_records = [
        'I38','I39','I40',
        'I41','I42','I43','I44','I45','I46','I47','I48','I49','I50',
        'I51','I52','I53','I54','I55','I56','I57','I58','I59','I60',
        'I61','I62','I63','I64','I65','I66','I67','I68','I69','I70',
        'I71','I73','I74','I75'
    ]
    print("test processing...")
        
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        test_data = [result for result in executor.map(worker, test_records)]
#
    with open((PATH1 / "mitdb_2.pkl").as_posix(), "wb") as f:
        pickle.dump((train_data,test_data), f, protocol=4)
#    with open((PATH1 / "mitdb_2.pkl").as_posix(), "wb") as f:
#        pickle.dump((train_data), f, protocol=4)
#    
#
    print("ok!")
