from tensorflow.keras.models import load_model
later_model = load_model('my_model.h5')
# [[Feature1, Feature2]]
new_gem2 = [[998,1000]]
import pickle
scaler2 = pickle.load(open('scaler.sav', 'rb'))
new_gem2 = scaler2.transform(new_gem2)
later_model.predict(new_gem2)