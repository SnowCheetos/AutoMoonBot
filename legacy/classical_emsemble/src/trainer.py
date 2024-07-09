import json
import pika
import redis
import numpy as np

import multiprocessing as mp
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import *

class Trainer:
    def __init__(
            self, 
            num_periods=4
        ) -> None:
        self.procs = int(mp.cpu_count() * 0.8)
        self.nb = GaussianNB()
        self.rf = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            n_jobs=self.procs
        )
        self.kn = KNeighborsClassifier(
            n_neighbors=5,
            weights="distance",
            n_jobs=self.procs
        )
        self.periods = np.logspace(3, num_periods+2, num=num_periods, base=2).astype(int)
        self.col_idx = np.random.permutation(np.arange(5 * self.periods.shape[0]))
        self.confs = np.ones(3) / 3

        self.redis_conn = redis.Redis(decode_responses=True)

        self.pred_map = {
            0: "sell",
            1: "hold",
            2: "buy"
        }

    def transform(self, x):
        rsis = np.stack([calculate_rsi(x, p) for p in self.periods], axis=-1)
        smas = np.stack([calculate_sma(x, p) for p in self.periods], axis=-1)
        emas = np.stack([calculate_ema(x, p) for p in self.periods], axis=-1)
        blbs = np.stack([calculate_bollinger_bands(x, p) for p in self.periods], axis=-1)
        stos = np.stack([calculate_stochastic_oscillator(x, p) for p in self.periods], axis=-1)
        features = np.hstack([
            rsis, smas, emas, blbs, stos
        ])
        features = features[:, self.col_idx]
        return features

    def fit(self, x, chunk_size=72):
        y = label_data(x, chunk_size)
        x = self.transform(x)
        rows_without_nan = ~np.any(np.isnan(x), axis=1)
        indices = np.where(rows_without_nan)[0]
        x, y = x[indices], y[indices]
        x, y = shuffle(x, y)
        n = int(0.8 * x.shape[0])
        x_train, x_test = x[:n], x[n:]
        y_train, y_test = y[:n], y[n:]
        print(x_train)
        print("[INFO] Training models... \n")
        self.kn.fit(x_train, y_train)
        print("[INFO] K-Nearest-Neighbor trained. \n")
        self.nb.fit(x_train, y_train)
        print("[INFO] Naive-Bayes trained. \n")
        self.rf.fit(x_train, y_train)
        print("[INFO] Random-Forest trained. \n")
        self.assign_weights(x_test, y_test)
        print("[INFO] Weight assigned, training complete. \n")

    def predict(self, x):
        x = self.transform(x)
        rows_without_nan = ~np.any(np.isnan(x), axis=1)
        indices = np.where(rows_without_nan)[0]
        x = x[indices]
        preds = np.stack([
            self.kn.predict_proba(x),
            self.rf.predict_proba(x),
            self.nb.predict_proba(x)
        ], -1)
        preds = np.einsum("ijk,k->ij", preds, self.confs)
        return preds.argmax(1)[-1]

    def assign_weights(self, x, y):
        nb_p = self.nb.predict(x)
        kn_p = self.kn.predict(x)
        rf_p = self.rf.predict(x)

        accuracies = np.asarray([
            accuracy_score(nb_p, y),
            accuracy_score(kn_p, y),
            accuracy_score(rf_p, y)
        ])
        self.confs = accuracies / np.linalg.norm(accuracies)

    def get_x(self, num: int) -> np.ndarray:
        try:
            # Fetch the last 'num' values for each key using Redis pipeline
            pipe = self.redis_conn.pipeline()
            pipe.lrange("open", -num, -1)
            pipe.lrange("high", -num, -1)
            pipe.lrange("low", -num, -1)
            pipe.lrange("close", -num, -1)
            results = pipe.execute()

            # Convert the results to a NumPy array and ensure type is float
            data_array = np.asarray(results).astype(float)
            return data_array.T
        
        except Exception as e:
            print(f"Error in get_x method: {e}")
            return None

    def callback(self, channel, method_frame, header_frame, body):
        ops = json.loads(body)['ops']
        print(f"got ops {ops}")
        if ops == "pred":
            x = self.get_x(72)
            res = self.predict(x)
            self.redis_conn.set("prediction", self.pred_map[res])
            print("Prediction: ", self.pred_map[res])

        elif ops == "train":
            x = self.get_x(720)
            self.fit(x)

    def start(self):
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
            channel = connection.channel()
            print("Connection success")
            channel.queue_declare('pred_service')
            
            # Set up the consumer
            channel.basic_consume(queue='pred_service', on_message_callback=self.callback, auto_ack=True)
            
            # Start consuming messages
            channel.start_consuming()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            # Ensure the connection is closed
            # if connection:
            #     connection.close()
            print("Close connection")

    def run_forever(self):
        while True:
            self.start()

if __name__ == "__main__":
    s = Trainer()
    s.run_forever()