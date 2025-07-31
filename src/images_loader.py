import os
from PIL import Image
import numpy
import exifread
import rawpy


class ImagesLoader:
     
    def __init__(self, path, size = None):
        file_names = self._find_images(path)

        images = []
        exposure_time = []
        for f_name in file_names:
            
            im = Image.open(f_name) 
            if size is not None:
                im = im.resize(size)
            im = numpy.array(im, dtype=numpy.float32)/float(255.0)

            with open(f_name, 'rb') as f:
                tags = exifread.process_file(f, stop_tag='EXIF ExposureTime')
                ev = tags.get('EXIF ExposureTime')
                
            exposure_val = float(ev.values[0].num) / float(ev.values[0].den)

            print("loading ", f_name, im.shape, exposure_val)

            images.append(im)
            exposure_time.append(exposure_val)
   
        self.x = numpy.array(images, dtype=numpy.float32)
        self.exposure_time = exposure_time

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
         return self.x[idx]

    def _find_images(self, root_path):
            image_extensions = {'.jpg', '.JPG', '.png', '.PNG'}
            image_paths = []

            for dirpath, dirnames, filenames in os.walk(root_path):
                for filename in filenames:
                    _, ext = os.path.splitext(filename)
                    if ext in image_extensions:
                        full_path = os.path.join(dirpath, filename)
                        image_paths.append(full_path)

            return image_paths
    



class ImagesLoaderRaw:
     
    def __init__(self, file_names, size = None, start_idx = 0, end_idx = -1):

        if isinstance(file_names, str):
            self.file_names = self._find_images(file_names)
        else:
            self.file_names = list(file_names)
            self.file_names.sort()

        if end_idx != -1:
            self.file_names = self.file_names[start_idx:end_idx]

        images = []
        exposure_time = []
        iso = []
        metadata = []
        for f_name in self.file_names:
            
            raw = rawpy.imread(f_name)
            rgb = raw.postprocess(
                gamma=(1, 1),        # Linear output
                no_auto_bright=True,
                output_bps=16,       # Still returns float32 internally before clipping
                user_flip=0, 
                use_camera_wb=True 
            )

            im = rgb.astype(numpy.float32) / 65535.0  # Normalize to [0,1]

          

            with open(f_name, 'rb') as f:
                tags = exifread.process_file(f, stop_tag='EXIF ExposureTime')
                ev = tags.get('EXIF ExposureTime')
                iso_val = tags.get('EXIF ISOSpeedRatings')
                orientation = tags.get('Image Orientation', None)
                
            exposure_val = float(ev.values[0].num) / float(ev.values[0].den)

            iso_val = float(str(iso_val))

            print("orientation", orientation)

            if "90" in str(orientation):
                im = numpy.rot90(im, 1)  # Rotate 90 CW
                

            images.append(im)

            metadata.append(tags)

            print("loading :", f_name, "  dims :", im.shape,  "  exposure time :", ev,  " iso:", iso_val,  " min,max:", numpy.min(im), numpy.max(im))

            exposure_time.append(exposure_val)
            iso.append(iso_val)
   
        self.x = numpy.array(images, dtype=numpy.float32)
        
        self.exposure_time = numpy.array(exposure_time)
        self.iso           = numpy.array(iso)

        self.metadata = metadata

    def __len__(self):  
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]

    def _find_images(self, root_path):
        image_extensions = {'.ARW', '.arw'}
        image_paths = []

        for dirpath, dirnames, filenames in os.walk(root_path):
            for filename in filenames:
                _, ext = os.path.splitext(filename)
                if ext in image_extensions:
                    full_path = os.path.join(dirpath, filename)
                    image_paths.append(full_path)

        image_paths.sort()

        return image_paths