from cnn_model_package import __version__ as _version
from cnn_model_package.predict import (make_single_prediction)


def test_make_prediction_on_sample(charlock_dir):
    # Given
    filename = '1.png'
    expected_classification = 'Charlock'

    # When
    results = make_single_prediction(image_directory=charlock_dir,
                                     image_name=filename)

    # Then
    assert results['predictions'] is not None
    assert results['readable_predictions'][0] == expected_classification
    assert results['version'] == _version
