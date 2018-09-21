import numpy as np
import pandas as pd
from flask import Flask
from flask import render_template
from flask import request


def lin_reg_gd(X, y, N=50, step_size=0.000001):
    # estraggo numero di osservazioni e di feature
    m, n = X.shape
    # creo copia di X con aggiunta una colonna di 1
    # (hstack concatena matrici orizzontalmente)
    X1 = np.hstack([np.ones((m, 1)), X])
    # inizializzo vettore parametri a zero
    theta = np.zeros(n + 1)
    for it in range(N):
        # calcolo l'errore su ciascuna osservazione
        error = X1.dot(theta) - y
        # calcolo il gradiente
        grad = 2 / m * (X1.T.dot(error))
        # aggiorno il vettore dei parametri
        theta -= step_size * grad
    return theta


app = Flask(__name__)


@app.route("/")
def redirect():
    return render_template("default.html", result=0)


@app.route("/", methods=["POST"])
def calculate():
    vino = request.form["kind"]
    if(vino == 'white'):
        data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=";")
    else:
        data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")

    y = data['quality']
    X = data.drop(columns=['quality'])

    theta = lin_reg_gd(X, y)

    acid = request.form['acidity']
    acid = float(acid)
    v_ac = request.form['v_acidity']
    v_ac = float(v_ac)
    c_ac = request.form['c_acidity']
    c_ac = float(c_ac)
    sugar = request.form['sugar']
    sugar = float(sugar)
    chlorides = request.form['chlorides']
    chlorides = float(chlorides)
    sulfur = request.form['sulfur']
    sulfur = float(sulfur)
    t_sulfur = request.form['t_sulfur']
    t_sulfur = float(t_sulfur)
    density = request.form['density']
    density = float(density)
    ph = request.form['ph']
    ph = float(ph)
    sulph = request.form['sulph']
    sulph = float(sulph)
    alcohol = request.form['alcohol']
    alcohol = float(alcohol)

    sample = np.array([acid, v_ac, c_ac, sugar, chlorides, sulfur, t_sulfur, density, ph, sulph, alcohol])

    import sklearn
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = \
        train_test_split(X, y, test_size=0.3, random_state=44)
    from sklearn.linear_model import LinearRegression
    lrm = LinearRegression()
    lrm.fit(X_train, y_train)
    sample = sample.reshape(1, -1)
    pred_train = lrm.predict(sample)
  # name = theta[0] + theta[1:].dot(sample)

    return render_template("default.html", result=pred_train)
