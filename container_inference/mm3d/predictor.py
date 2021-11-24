# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

from __future__ import print_function

import os
import io
import json
import tempfile
import pickle

import flask

import torch

from mmdet3d.apis import init_model, inference_detector

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

"""
The model artifact must contain the configuration file and the model checkpoint.

The configuration file should contain absolute paths to /mmdetection3d if it points
to base files.

The prediction method accepts a point cloud byte stream, and returns a pickled version
of the response.
"""

class PredictService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
                
            config_file = os.path.join(model_path, "3dssd_4x4_a2d2-3d-car.py")
            checkpoint_file = os.path.join(model_path, "epoch_1.pth")
            print(f"Loading config file {config_file} from path {model_path}")

            cls.model = init_model(config_file, checkpoint_file, device=device)
            
        return cls.model

    @classmethod
    def predict(cls, input):
        
        clf = cls.get_model()
        f = io.BytesIO(input)
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(f.read())
        
        # get inference results
        res, data = inference_detector(clf, tfile.name)
        results = {}
        
        # change torch tensors to numpy arrays
        results['boxes_3d'] = res[0]['boxes_3d'].tensor.detach().cpu().numpy()
        results['scores_3d'] = res[0]['scores_3d'].tensor.detach().cpu().numpy()
        results['labels_3d'] = res[0]['labels_3d'].tensor.detach().cpu().numpy()
        mm_result = {'result': results, 'data': data}
        return mm_result

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = PredictService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():

    predictions = PredictService.predict(flask.request.data)

    result = pickle.dumps(predictions)

    return flask.Response(response=result, status=200, mimetype='application/octet-stream')
