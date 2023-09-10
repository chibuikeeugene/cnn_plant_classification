from sklearn.pipeline import Pipeline

from cnn_model_package.config import config
from cnn_model_package.processing import preprocessors as pp
from cnn_model_package import model


pipe = Pipeline([
                ('dataset', pp.CreateDataset(config.IMAGE_SIZE)),
                ('cnn_model', model.cnn_clf)])