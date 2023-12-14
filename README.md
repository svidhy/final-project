# final-project

# Dish Search Engine

## Overview

Welcome to the Dish Search Engine project! This application serves as a dish search engine where users can upload a picture, and a deep learning model will recommend the likely category of the dish. The model has been trained on a dataset containing 91 categories, spanning multiple cuisines.

## Features

- **Dish Identification:** Upload a picture of a dish, and the deep learning model will identify and recommend the likely category it belongs to.

## Files

- **Main.ipynb:** Main Jupyter notebook containing all the coding information.
- **deploy_model.py:** Streamlit file for deploying the model on a website.
- **Functions.py:** Collection of functions needed to deploy the model.
- **recipe_database.csv:** CSV file containing recipes from Edamam with nutrition information.
- **Prediction/:** Folder containing photos used to test the model.

## Source Data

- The project leverages data from the [Food 101 dataset](https://www.kaggle.com/datasets/kmader/food41/data) and [Indian Food Classification dataset](https://www.kaggle.com/datasets/theeyeschico/indian-food-classification).
- Special acknowledgment to the [Kaggle notebook](https://www.kaggle.com/code/theeyeschico/food-classification-using-tensorflow) providing detailed information on using TensorFlow.

## Getting Started

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/dish-search-engine.git
    cd dish-search-engine
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:

    ```bash
    streamlit run deploy_model.py
    ```

    Access the application at [http://localhost:8501](http://localhost:8501) in your web browser.

## Model Training

The deep learning model has been trained on a dataset with 91 dish categories. Refer to `Main.ipynb` for details on model training and evaluation.

## Contributing

Contributions are welcome! If you'd like to enhance or extend the functionality of the dish search engine, please submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
