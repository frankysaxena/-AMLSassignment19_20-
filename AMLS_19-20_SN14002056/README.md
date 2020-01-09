## Structure

A1, A2, B1, B2 are the corresponding folders where the models are stored for each task with results already saved as confusion matrices.
Models have been saved as distinct files which were the last best trained models when I ran through the dataset.

### Files

To run their respective tasks:
* A1/A1.py
* A2/A2.py
* B1/B1.py
* B2/B2.py

For data pre-processing and loading data for the classes:
* Datasets/DataPreprocessing.py

For ad-hoc testing and gathering graph data results for both Tasks A and B:
* TaskAmodel_testing_notebook.ipynb
* Taskbmodel_testing_notebook.ipynb

You can run the Jupyter notebooks as is to display results. These notebooks were essentially my 'working out' while I was going through the different models and tasks.

Main program execution file:
* main.py
    
Total execution time should be about 10-15 minutes.

## Usage

### Requirements

Please ensure you have these dependencies:

`pandas`
`numpy`
`scikit-learn`
`opencv-python`
`parfit`
`scikit-image`
`Pillow-6.2.2` (latest Pillow will be fine too)

These have all been listed in the `requirements.txt` file

### Run

This code has been tested on Python 3.7+. It should work fine on any Python3 flavour, however Python2 is not advised.

Make sure you have set Python 3 as default.
Navigate to your preferred directory

1. ` git clone << this repository >>`
2. `source venv bin activate`
3. `cd AMLS_19-20_SN14002056`
4. `pip install -r requirements.txt`
5. `vi  main.py` 
6. Edit the `path_to_dir` variable accordingly
7. `python main.py`

Or if you already have all the above dependencies installed, then just change the path directory  and run python main.py


