from model.prep import prep_input_data, prep_models, split_file
from model.utils import SMALL_MODEL, LARGER_MODEL

if __name__ == "__main__":
    # prep_models([SMALL_MODEL, LARGER_MODEL])
    prep_input_data()
    split_file()