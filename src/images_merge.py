import cv2 
import numpy

from images_utils import *

class ImagesMerge:


    def process(self, images, exposure_time, iso, ref_idx, rect = None):
        result = self._merge_hdr(images, exposure_time, iso, ref_idx)

        result = numpy.clip(result, 0.0, 1000.0)

        result = self._normalise(result)

        # crop
        if rect is not None:
            top         = rect["top"]
            bottom      = rect["bottom"]
            left        = rect["left"]
            right       = rect["right"]

            result_rect = cv2.rectangle(numpy.array(result), (left, top), (right, bottom), (0, 1, 0), 2)
            result_crop = numpy.array(result[top:bottom, left:right, :])

            return result, result_rect, result_crop

        return result



    def _normalize_exposures(self, images, exposures, iso, best_id):
        
        # Avoid division by zero
        ref_exposure = exposures[best_id]
        ref_iso      = iso[best_id]
       
        scale = (ref_exposure / exposures) * (iso / ref_iso)

        # Reshape to broadcast over image dimensions
        scale = scale[:, None, None, None]

        # Apply scaling
        normalized_images = images * scale

        return normalized_images.astype(numpy.float32)




    def _merge_hdr(self, images, exposure_time, iso, ref_idx, kernel_size = 11):
        """
        Merge a stack of exposure-normalized images into an HDR image by selecting
        the best-exposed pixel at each location using per-pixel weights.

        Parameters:
            images (np.ndarray): Shape (N, H, W, 3), float32, values in [0, +inf)

        Returns:
            np.ndarray: HDR image of shape (H, W, 3), float32
        """
        # Compute simple triangular weights favoring mid-tone values
        #images_gs   = images.mean(axis=-1)
        #weights     = 1.0 - 2.0 * numpy.abs(images_gs - 0.5)  # Shape: (N, H, W, 3)
        #weights     = numpy.clip(weights, 0.0, 1.0)

        weights = []

        for n in range(len(images)):
            w = rate_image(images[n], exposure_bias=0.5, exp_power=2.0, sharp_weight=1.0, color_weight=1.0)
            #w = rate_image(images[n], exposure_bias=0.5, exp_power=0.1, sharp_weight=1.0, color_weight=1.0)
            weights.append(w)
            

        weights = numpy.array(weights)

        
        for n in range(len(images)):
            weights[n] = cv2.GaussianBlur(weights[n], (kernel_size, kernel_size), 3)


        weights     = numpy.clip(weights, 1e-8, 1.0)
        weights     = weights/numpy.expand_dims(weights.sum(axis=0), 0)
        
        images_norm = self._normalize_exposures(images, exposure_time, iso, ref_idx)

        result = float(0.0)
        for n in range(len(images)):
            result+= images_norm[n]*numpy.expand_dims(weights[n], 2)
            
        '''
        for n in range(len(images)):
            w = weights[n]
            cv2.imwrite("w_"+str(n)+".jpg", numpy.array(255*w, dtype=numpy.uint8))
        '''
         
        return result
    

    def _normalise(self, x):
        min = numpy.min(x)
        max = numpy.max(x)

        y = (x - min)/(max - min)

        return y