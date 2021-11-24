## End-to-End 3D ML on SageMaker Workshop

This repository contains the code for running the End-to-End 3D ML on SageMaker workshop. The goal of this workshop is to give you hands on experience with building an end-to-end 3D machine learning pipeline on Amazon SageMaker. 

You will learn how to train and deploy a real-time 3D object detection model with Amazon SageMaker through:

1. Downloading and visualizing a point cloud dataset
2. Prepping data to be labeled with the SageMaker Ground Truth point cloud tool
3. Launching a distributed SageMaker training job with MMDetection3D
4. Evaluating your training job results and profiling your resource utilization
5. Deploying an asynchronous SageMaker endpoint
6. Calling the endpoint and visualizing 3D object predictions

MMDetection3D provides samples of several datasets, for the purpose of this workshop, we will not be using any of those samples and will instead use the A2D2 dataset.

To reduce processing time we are including pickle files containing paths and some annotation information that will be used in the dataloader. They can be generated using a2d2_converter.py and a2d2_database.py. For more details take a look at the README in the `a2d2` folder.




## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

