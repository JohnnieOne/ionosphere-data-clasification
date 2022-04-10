# SBURATURA GEORGE-ADRIAN, 423B
# PROIECT ISIA - ionosphere

from sklearn import metrics, ensemble
import numpy as np
import pandas as pd

data = pd.read_csv('ionosphere.data')
data_backup = data  # salvam datele intr-o variabila de back-up in caz ca mai este nevoie de ele
etichete_l = data["eticheta"].values.tolist()
for i in range(len(etichete_l)):
    # se convertesc datele din etichete din good/bad in 1, respectiv 0 intrucat este mult mai simplu de lucrat
    if etichete_l[i] == 'g':
        etichete_l[i] = 1  # eticheta good primeste valoarea 1
    if etichete_l[i] == 'b':
        etichete_l[i] = 0  # eticheta good primeste valoarea 0
etichete = np.array(etichete_l)
etichete_train = etichete[88:]
etichete_test = etichete[:88]

data = data.drop(columns='eticheta')
date = data.to_numpy()

good_train = np.count_nonzero(etichete_train == 1)
bad_train = len(etichete_train) - good_train
proportie_good_train = (good_train / len(etichete_train)) * 100
proportie_bad_train = (bad_train / len(etichete_train)) * 100

good_test = np.count_nonzero(etichete_test == 1)
bad_test = len(etichete_test) - good_test
proportie_good_test = (good_test / len(etichete_test)) * 100
proportie_bad_test = (bad_test / len(etichete_test)) * 100

date_train = date[88:][:]
date_test = date[:88][:]

clf = ensemble.RandomForestClassifier(n_estimators=14, max_samples=0.85, max_features=0.5, random_state=15000)
clf.fit(date_train, etichete_train)
predictii = clf.predict(date_test)

acuratete = metrics.accuracy_score(y_true=etichete_test, y_pred=predictii)
acuratete = int(acuratete * 10000) / 100

print(date_train)
print()
print(date_test)
print(etichete_train)
print("good train = " + str(good_train), "bad train =" + str(bad_train))
print("Proportie good train = " + str(proportie_good_train), "Proportie bad train = " + str(proportie_bad_train))
print(etichete_test)
print("good test = " + str(good_test), "bad test = " + str(bad_test))
print("Proportie good test = " + str(proportie_good_test), "Proportie bad test = " + str(proportie_bad_test))
print("Acuratete este de " + str(acuratete) + "%")