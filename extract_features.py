from keras.models import Model
from keras.models import load_model
from dataset_loaders import SimpleDatasetLoader
from imutils import paths
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from keras.datasets import mnist
from sklearn.metrics import classification_report
import keras
from sklearn.preprocessing import LabelEncoder

(x_train, y_train) = mnist.load_data()[0]
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
model = load_model('models/hdf5/mnist_lenet_official.hdf5')
# print(model.summary())
sdl = SimpleDatasetLoader()
image_paths = list(paths.list_images('./datasets/digits'))
x_test, y_test = sdl.load(image_paths)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)
le = LabelEncoder()
y_test = le.fit_transform(y_test)
# y_train = le.fit_transform(y_train)
# 取某一层的输出为输出新建为model，采用函数模型
flatten1_layer_model = Model(inputs=model.input,
                             outputs=model.get_layer('flatten_1').output)
# 以这个model的预测值作为输出
features = flatten1_layer_model.predict(x_train)
print("[INFO] tuning hyperparameters...")
params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=-1)
model.fit(features, y_train)
print("[INFO] best hyperparameters: {}".format(model.best_params_))
# evaluate the model
print("[INFO] evaluating...")
preds = model.predict(x_test)
print(classification_report(y_test, preds,
                            target_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']))
