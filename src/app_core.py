from images_utils   import *
from images_loader  import *
from images_align   import *
from images_merge   import *



import cv2
import tifffile

import os


class APPCore:

    def __init__(self):
        pass


    def load_files(self, path):
        print("\n\n")
        print("loading files")
        self.path = path
        self.images = ImagesLoaderRaw(path)


        # rate images exposures and information quality
        self.ratings, self.dynamical_range = rate_images(self.images)
        self.best_image_idx           = numpy.argmax(self.ratings)
        
        print("\n\n")
        print("rating ", self.ratings)
        print("dynamical range ", self.dynamical_range, "dB")
        print("best image id", self.best_image_idx)

        result_file_name = ""
        for n in range(len(self.images)):
            file_nama_tmp = os.path.split(self.images.file_names[n])[-1]
            file_nama_tmp = os.path.splitext(file_nama_tmp)[0]

            result_file_name+= file_nama_tmp + "_"

        result_file_name+= "HDR"

        self.result_file_name = result_file_name

   
    def get_thumbnails(self): 
        self.thumbnails = []
        for n in range(len(self.images)):
            img = numpy.array(self.images[n], dtype=numpy.float32)
            img = self._normalise(img)
            
            h = img.shape[0]//8
            w = img.shape[1]//8
                        
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

            self.thumbnails.append(img)

        return self.thumbnails
    

    def get_result(self): 
        return self._normalise(self.result_rect)
        

    def process(self):
        print("\n\n")
        print("warping images")
        align = ImagesAlign()

        
        
        warped_images, scores, rect = align.process_all(self.images, self.best_image_idx, cv2.MOTION_HOMOGRAPHY, 2)

        print("\n\n")
        print("stacking images")    
        merger = ImagesMerge()

        self.result, self.result_rect, self.result_crop = merger.process(warped_images, self.images.exposure_time, self.images.iso, self.best_image_idx, rect)
        
        db_new = compute_dynamic_range_db(self.result)
        ddb = round(db_new - numpy.max(self.dynamical_range), 2) 
        print("result dims ", self.result.shape)
        print("result dynamical range ", db_new, "dB",  " improvement ", ddb , "dB")
        print("values range ", numpy.min(self.result), numpy.max(self.result))

    def export(self): 
        self.save(self.path + "/hdr_result/", False)
        pass
    
    def export_cropped(self): 
        self.save(self.path + "/hdr_result/", True)
        

    def save(self, path = "./", cropped = False):

        if cropped:
            result = self.result_crop
        else:
            result = self.result_rect

        matadata = self.images.metadata[self.best_image_idx]
        
        # Extract useful EXIF values (only a few shown here)
        exposure    = str(matadata.get('EXIF ExposureTime', '1/60'))
        fnumber     = str(matadata.get('EXIF FNumber', '2.8'))
        make        = str(matadata.get('Image Make', 'Unknown'))
        model       = str(matadata.get('Image Model', 'Unknown'))
        datetime    = str(matadata.get('Image DateTime', '2024:01:01 00:00:00'))

        focal_length = str(matadata.get('EXIF FocalLength', '50/1'))
        lens_make    = str(matadata.get('EXIF LensMake', 'Unknown'))
        lens_model   = str(matadata.get('EXIF LensModel', 'Unknown'))
    
        print("\n\n")
        print("saving")

        exposure_num, exposure_den = map(int, str(exposure).split('/'))
        f_num, f_den = map(int, str(fnumber).split('/'))
        
        extratags = [
            (33434, 5, 1, (exposure_num, exposure_den), False),          # ExposureTime
            (33437, 5, 1, (f_num, f_den), False),         # FNumber
            (271, 2, 5, make, False),            # Camera Make
            (272, 2, 10, model, False),         # Camera Model
            (306, 2, 19, datetime, False),  # DateTime
            (37386, 5, 1, (int(focal_length), int(1)), False),          # Focal Length
            (42035, 2, 5, lens_make, False),          # Lens Make
            (42036, 2, 16, lens_model, False)  # Lens Model
        ]
        

        thumbnail = ((result/numpy.max(result)) * 255).astype(numpy.uint8)
        thumbnail = thumbnail[::8, ::8]  


        if not os.path.exists(path):
            os.makedirs(path)
        
        if cropped:
            result_file_name = path + self.result_file_name + "_crop.tiff"
        else:
            result_file_name = path + self.result_file_name + ".tiff"


        with tifffile.TiffWriter(result_file_name) as tif:
            # Page 1: full image
            tif.write(
                result,
                photometric='rgb',
                dtype=numpy.float32,
                extratags=extratags
            )

            '''
            # Page 2: thumbnail
            tif.write(
                thumbnail,
                photometric='rgb',
                dtype=numpy.uint8,
                description='Thumbnail'
            )
            '''

        print("done")
            


    def _normalise(self, x):
        y = (x - numpy.min(x))/(numpy.max(x) - numpy.min(x))
        return 1.5*y