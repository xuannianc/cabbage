from keras.models import Model
from keras.models import load_model
from dataset_loaders import SimpleDatasetLoader
from imutils import paths
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

model = load_model('models/hdf5/mnist_lenet_official.hdf5')
# print(model.summary())
sdl = SimpleDatasetLoader()
image_paths = list(paths.list_images('./datasets/digits'))
x_test, y_test = sdl.load(image_paths)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# 取某一层的输出为输出新建为model，采用函数模型
flatten1_layer_model = Model(inputs=model.input,
                             outputs=model.get_layer('flatten_1').output)
# 以这个model的预测值作为输出
features = flatten1_layer_model.predict(x_test)

print("[INFO] tuning hyperparameters...")
params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=4)
model.fit(features, y_test)
print("[INFO] best hyperparameters: {}".format(model.best_params_))
# evaluate the model
print("[INFO] evaluating...")
preds = model.predict(db["features"][i:])
print(classification_report(db["labels"][i:], preds,
                            target_names=db["label_names"]))
