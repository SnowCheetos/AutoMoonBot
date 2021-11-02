from models import *
from gen_transform import *

if __name__ == "__main__":
    asset = input("Asset: ")
    try:
        X, Y = load_data(asset, 'data/')
    except:
        gen_transform(asset, chunk_size = 320, start = 300, end = 100)
        X, Y = load_data(asset, 'data/')
    model_gen(asset, X, Y)