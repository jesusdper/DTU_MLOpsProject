# object_detection

Object Detection Pipeline for Visual Object Recognition


Members: Estel Clua i Sánchez, Andrea Matamoros Alonso, Javier Alonso Fernández, Juanjo Martinez Mañas, Jesús Díaz Pereira

With this project we aim to develop and deploy an object detection pipeline capable of identifying and classifying objects in images. By the end of this project, I aim to have a fully functional pipeline that can preprocess data, train a Deep Learning model, evaluate its performance, and make predictions on unseen data.

We aim to use the COCO dataset, a well-established benchmark dataset for object detection tasks. The COCO dataset is known for its large size and diversity, containing over 200,000 labeled images and more than 80 object categories. However, due to its large size (~13GB), we will initially work with a small portion of it to streamline the development process and reduce computational requirements. Alternatively, we are considering using a smaller dataset for fast prototyping, such as the PASCAL VOC 2012 dataset. The PASCAL VOC dataset is another popular benchmark in the field of object detection, containing around 11,000 images and 20 object categories. Using a smaller dataset will allow us to quickly iterate and test our pipeline before scaling up to the full COCO dataset.

The definitive framework of the project is still uncertain, as we are considering different options. One of the suggestions for the framework is YOLO (You Only Look Once), a state-of-the-art object detection system known for its speed and accuracy. Other possibility is using MMDetection, an open source object detection toolbox based on Pytorch. Despite what we end up choosing, we will start working with a simple implementation, like a pre-trained YOLO model for transfer learning, which will allow us to quickly prototype and validate our approach.


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
