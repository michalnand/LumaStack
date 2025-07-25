from app_core import *

if __name__ == "__main__":

    app = APPCore()

    app.load_files("images_a/")
    app.process()
   
    app.save("result/")