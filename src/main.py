from app_core import *

if __name__ == "__main__":

    app = APPCore()

    path = "/Users/michal/Pictures/2025_26_7_sulovske_skaly/"
    app.process_batch(path, 3)

    '''
    app.load_files("images_a/")
    app.process()
   
    app.save("result/")
    '''