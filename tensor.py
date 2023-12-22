import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

'''Load data from csv file'''

data = pd.read_csv('./Iris.csv')


'''Droping the unwanted field in data'''
data = data.drop('Id',axis=1)

'''Coverting label into number of output field for learning purpose'''
ln = LabelEncoder()

data['Species'] = ln.fit_transform(data['Species'])

'''Creating X and y values as input and output for model'''
X = data.drop('Species',axis=1)
y = data['Species']

'''Creating train and test data'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

'''Bring values to standard scale'''
st = StandardScaler()

X_train = st.fit_transform(X_train)
X_test = st.transform(X_test)

'''Training tensorflow model'''
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64,activation='relu',input_shape=(4,)),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(3,activation='softmax')]
)

model.compile(optimizer='adam',loss='SparseCategoricalCrossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=100,validation_steps=0.2)


'''Prediction class'''
class predication_code():
    def __init__(self,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm):
        self.SepalLengthCm = SepalLengthCm
        self.SepalWidthCm  = SepalWidthCm
        self.PetalLengthCm = PetalLengthCm
        self.PetalWidthCm = PetalWidthCm
    
    def predict(self):
        new_data = st.transform([[self.SepalLengthCm,self.SepalWidthCm,self.PetalLengthCm,self.PetalWidthCm]])
        predict = model.predict(new_data)
        predicted_class = tf.argmax(predict, axis=1).numpy()[0]
        if predicted_class == 0:
            return 'Iris-setosa'
        elif predicted_class == 1:
            return 'Iris-versicolor'
        elif predicted_class == 2:
            return 'Iris-virginica'

'''Input from users'''
SepalLengthCm_input = float(input('SepalLength: '))
SepalWidthCm_input = float(input('SepalWidthCm: '))
PetalLengthCm_input = float(input('PetalLengthCm: '))
PetalWidthCm_input = float(input('PetalWidthCm: '))       

'''Calling the function'''
pd = predication_code(SepalLengthCm_input,SepalWidthCm_input,PetalLengthCm_input,PetalWidthCm_input)

result = pd.predict()

print('Output : {}'.format(result))
