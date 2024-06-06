import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import scipy.signal as sg
import wfdb

PATH = Path("dataset")
PATH1 = Path("data1")
sampling_rate = 128

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
        
    
    signal = wfdb.rdrecord((PATH / record).as_posix(), channels=[0]).p_signal[:, 0]
#    signal = wfdb.rdrecord((PATH / record).as_posix(), channels=[1]).p_signal[:, 0]
        
#    signal = wfdb.rdrecord((PATH / record).as_posix(), channels=[7]).p_signal[:, 0]
        

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
        "signal": normalized_signal, "r_peaks": r_peaks, "categories": categories
    }


if __name__ == "__main__":
    # for multi-processing
    cpus = 22 if joblib.cpu_count() > 22 else joblib.cpu_count() - 1

    
    
#    train_records = [
#        '800','801','802','803','804','805','806','807','808','809','810',
#        '811','812',
#        '820','821','822','823','824','825','826','827','828','829',
#        '840','841','842','843','844','845','846','847','848','849','850',
#        '851','852','853','854','855','856','857','858','859','860',
#        '861','862','863','864','865','866','867','868','869','870',
#        '871','872','873','874','875','876','877','878','879','880',
#        '881','882','883','884','885','886','887','888','889','890',
#        '891','892','893','894'
#
#    ]
    train_records = [
        '802','804','805','808','810','812','841','843','845','849',
        '866','871','873','876','877','886','890'
    ]
#    train_records = [
#        '800'
#    ]
    
    print("train processing...")
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        train_data = [result for result in executor.map(worker, train_records)]
#    train_data=worker('I18')

    test_records = [
        '800','803','806','807','809','811','840','842','844','846',
        '850','867','872','874','875','878','879'
    ]
    print("test processing...")
        
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        test_data = [result for result in executor.map(worker, test_records)]
#
    with open((PATH1 / "mitdb.pkl").as_posix(), "wb") as f:
        pickle.dump((train_data,test_data), f, protocol=4)
#    with open((PATH1 / "mitdb_2.pkl").as_posix(), "wb") as f:
#        pickle.dump((train_data), f, protocol=4)
#    
#
    print("ok!")
