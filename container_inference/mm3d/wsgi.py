import predictor as myapp

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# This is just a simple wrapper for gunicorn to find your app.
# If you want to change the algorithm file, simply change "predictor" above to the
# new file.

app = myapp.app
