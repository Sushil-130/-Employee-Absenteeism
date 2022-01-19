from flask import Flask, url_for, request, render_template
import pickle
import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.losshistory = []

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        # weights initialization
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)

                loss = self.__loss(h, y)
                self.losshistory.append(loss)
                print("iteration: ", i, " loss value: ", loss)

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold


app = Flask(__name__)
scaler = pickle.load(open("scaler.scaler", "rb"))
classifier = pickle.load(open("classifier.model", "rb"))


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        default_value = 0
        reason = int(request.form.get("reason"))
        reason1 = default_value  # dummy
        reason2 = default_value  # dummy
        reason3 = default_value  # dummy
        reason4 = default_value  # dummy

        if reason == 1:
            reason1 = 1
        elif reason == 2:
            reason2 = 1
        elif reason == 3:
            reason3 = 1
        elif reason == 4:
            reason4 = 1

        monthvalue = request.form.get("monthvalue", default_value)
        day = request.form.get("day", default_value)
        transportation = request.form.get("transportation", default_value)
        distancetowork = request.form.get("distancetowork", default_value)
        age = request.form.get("age", default_value)
        dailywork = request.form.get("dailywork", default_value)
        bmi = request.form.get("bmi", default_value)
        education = request.form.get("education", default_value)  # dummy
        children = request.form.get("children", default_value)
        pet = request.form.get("pet", default_value)

        temp = [int(monthvalue), int(day), int(transportation), int(
            distancetowork), int(age), int(dailywork), int(bmi), int(children), int(pet)]
        print(temp)

        scaledvalues = scaler.transform([temp])

        print("Scaled values : ", scaledvalues[0])

        inputs = np.zeros(14)

        inputs[0] = reason1
        inputs[1] = reason2
        inputs[2] = reason3
        inputs[3] = reason4

        inputs[11] = education

        inputs[4] = scaledvalues[0][0]
        inputs[5] = scaledvalues[0][1]
        inputs[6] = scaledvalues[0][2]
        inputs[7] = scaledvalues[0][3]
        inputs[8] = scaledvalues[0][4]
        inputs[9] = scaledvalues[0][5]
        inputs[10] = scaledvalues[0][6]
        inputs[12] = scaledvalues[0][7]
        inputs[13] = scaledvalues[0][8]

        print("Processed Input : ", inputs)

        prediction = classifier.predict(inputs.reshape(1, -1))
        print(prediction)
        return render_template('result.html', result="Will be absent." if prediction else "Will be present.")
    elif request.method == "GET":
        return render_template('predict.html')


if __name__ == "__main__":
    app.run(debug=True)
