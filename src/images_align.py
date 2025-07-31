import numpy
import cv2


class ImagesAlign:


    def process_all(self, images, ref_idx, warp_mode=cv2.MOTION_AFFINE, scale = 2, iterations=128):

        warped_images = []
        w_matrices    = []
        scores        = []

        img_base = images[ref_idx]

        for n in range(len(images)):
            print("warping image ", n)
            # Load two grayscale images

            if n != ref_idx:
                img_dest    = images[n]

                warped_img, w, score = self.process(img_base, img_dest, scale)
                                
                warped_images.append(warped_img)
                w_matrices.append(w)
                scores.append(score)

            else:
                warped_images.append(images[n])
                #w_matrices.append(w)
                scores.append(0)

        warped_images   = numpy.array(warped_images)
        w_matrices      = numpy.array(w_matrices)
        scores          = numpy.array(scores)

      
        rect = self._compute_rect(w_matrices, img_base.shape)

        return warped_images, scores, rect
    

    def process(self, ref_img, target_img, scale = 2):        
        h, w, _ = ref_img.shape

        # preprocess images
        # grayscale, uint8, clahe equalisation, resize

        ref_img_gray    = self._preprocess_image(ref_img, scale)
        target_img_gray = self._preprocess_image(target_img, scale)

        
        # coarse alignment based on keypoints, SIFT detector, BF keypoints matcher
        w_mat = self._align_coarse(ref_img_gray, target_img_gray)

        # fine alingment based on ECC
        w_mat = self._align_ecc(ref_img_gray, target_img_gray, w_mat, iterations=64)

        # scale matrix
        S = numpy.array([[scale, 0, 0],
                        [0, scale, 0],
                        [0, 0, 1]], dtype=numpy.float32)

        # upscale the matrix
        warp_matrix = S @ w_mat @ numpy.linalg.inv(S)
        
        
        # warping
        align_coarse = cv2.warpPerspective(target_img, warp_matrix, (w, h), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)

        # non homogenous warping using optical flow
        ref_img_gray    = self._preprocess_image(ref_img, 1)
        target_img_gray = self._preprocess_image(align_coarse, 1)

        
        flow = cv2.calcOpticalFlowFarneback(
            ref_img_gray,
            target_img_gray,
            None,              # Output flow
            pyr_scale=0.5,     # Pyramid scale
            levels=3,          # Number of pyramid layers
            winsize=15,        # Averaging window size
            iterations=32,     # Iterations at each pyramid level
            poly_n=5,          # Size of pixel neighborhood
            poly_sigma=1.2,    # Standard deviation for Gaussian
            flags=0
        )
        
        h, w = flow.shape[:2]
        map_x, map_y = numpy.meshgrid(numpy.arange(w), numpy.arange(h))
        map_x = map_x + flow[..., 0]
        map_y = map_y + flow[..., 1]
        img_result = cv2.remap(numpy.array(align_coarse), map_x.astype(numpy.float32), map_y.astype(numpy.float32), interpolation=cv2.INTER_LINEAR)
                        


     
        score = self._score(ref_img, img_result)

        warp_matrix = numpy.eye(3)

        return img_result, warp_matrix, score
    


    def _preprocess_image(self, x, scale):
        h, w, _ = x.shape
        h_small, w_small = h // scale, w // scale

        x   = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        x   = numpy.clip(255*x, 0, 255).astype(numpy.uint8)

        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        x       = clahe.apply(x)

        x   = cv2.resize(x, (w_small, h_small), interpolation=cv2.INTER_AREA)

        return x


    def _align_coarse(self, ref_img, target_img):
    
        # Init warp matrix (low-res)
        warp_matrix = numpy.eye(3, 3, dtype=numpy.float32)
        
        # 1, Coarse matching using keypoints, fast method
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(ref_img, None)
        kp2, des2 = sift.detectAndCompute(target_img, None)

        # Match and estimate affine
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        if len(matches) >= 30:   
            pts1 = numpy.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts2 = numpy.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            matrix, _ = cv2.estimateAffinePartial2D(pts2, pts1)
            if matrix is not None:
                warp_matrix[:2] = matrix
                print("matching", len(matches))

        return warp_matrix
    

    def _align_ecc(self, ref_img, target_img, warp_initial, iterations=128):

        ref_img     = numpy.array(ref_img/255.0, dtype=numpy.float32)
        target_img = numpy.array(target_img/255.0, dtype=numpy.float32)

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 1e-7)

        try:
            cc, warp_matrix = cv2.findTransformECC(
                ref_img, target_img, numpy.array(warp_initial, dtype=numpy.float32),
                motionType=cv2.MOTION_HOMOGRAPHY,
                criteria=criteria
            )
        except cv2.error as e:
            print("ECC failed:", e)
        
        return warp_matrix
    


    def _score(self, img_a, img_b, scale=8):
        h, w, _ = img_a.shape
        h_small, w_small = h // scale, w // scale

        
        img_a_norm = img_a.mean(axis=-1)
        img_a_norm = cv2.resize(img_a_norm, (w_small, h_small), interpolation=cv2.INTER_AREA)
        img_a_norm = (img_a_norm - img_a_norm.mean())/(img_a_norm.std() + 1e-6)
        img_a_norm = cv2.Laplacian(img_a_norm, cv2.CV_32F)

        img_b_norm = img_b.mean(axis=-1)
        img_b_norm = cv2.resize(img_b_norm, (w_small, h_small), interpolation=cv2.INTER_AREA)
        img_b_norm = (img_b_norm - img_b_norm.mean())/(img_b_norm.std() + 1e-6)
        img_b_norm = cv2.Laplacian(img_b_norm, cv2.CV_32F)

        score = ((img_a_norm - img_b_norm)**2).mean()

        return score
    
    def _get_edges(self, img):
        img = (img - img.mean())/(img.std() + 1e-6)
        img = cv2.Laplacian(img, cv2.CV_32F)

        return img

    def _compute_mask(self, images, ref_id):
        result = []
        for n in range(len(images)):     
            mask = numpy.logical_and(images[ref_id].mean(axis=-1) > 0.00001, images[n].mean(axis=-1) > 0.00001)
            mask = numpy.array(mask, dtype=numpy.float32)
            mask = cv2.GaussianBlur(mask, (11, 11), 3)

            result.append(mask)
            
        result = numpy.array(result)
        return result
    
   


    def _compute_rect(self, homographies, image_shape):
        """
        Transforms the image corners using a list of homography matrices and returns the most inner rectangle
        (intersection of all transformed images).
        
        Args:
            homographies (List[np.ndarray]): List of 3x3 homography matrices (NumPy arrays).
            image_shape (Tuple[int, int]): (height, width) of the original image in pixels.

        Returns:
            Tuple[float, float, float, float]: (min_x, min_y, max_x, max_y) of the most inner rectangle.
        """
        h, w, _ = image_shape
        h = h-1
        w = w-1
        # Define corners of the original image (in homogeneous coordinates)
        corners = numpy.array([
            [0,   0, 1],
            [w,   0, 1],
            [w,   h, 1],
            [0,   h, 1]
        ]).T  # shape: (3, 4)

        all_transformed_corners = []

        for H in homographies:
            transformed = numpy.linalg.inv(H) @ corners # shape: (3, 4)
            transformed /= transformed[2]  # normalize by the third (homogeneous) coordinate
            all_transformed_corners.append(transformed[:2])  # take only x, y

        # Now stack all transformed corners per image: (N_images, 2, 4)
        transformed_corners = numpy.stack(all_transformed_corners)  # shape: (N, 2, 4)


        right_a = numpy.min(transformed_corners[:, 0, 1])
        right_b = numpy.min(transformed_corners[:, 0, 2])
        right   = min(right_a, right_b)

        left_a  = numpy.max(transformed_corners[:, 0, 0])
        left_b  = numpy.max(transformed_corners[:, 0, 3])
        left    = max(left_a, left_b)

        top_a   = numpy.max(transformed_corners[:, 1, 0])
        top_b   = numpy.max(transformed_corners[:, 1, 1])
        top     = max(top_a, top_b)

        bottom_a   = numpy.min(transformed_corners[:, 1, 2])
        bottom_b   = numpy.min(transformed_corners[:, 1, 3])
        bottom     = min(bottom_a, bottom_b)

        rect = {}   
        
        rect["left"]    = int(numpy.clip(left, 0, w))
        rect["right"]   = int(numpy.clip(right, 0, w))

        rect["top"]     = int(numpy.clip(top, 0, h))
        rect["bottom"]  = int(numpy.clip(bottom, 0, h))

        return rect

        # Find axis-aligned bounding box for each image
        min_x = transformed_corners[:, 0, :].min(axis=1)
        max_x = transformed_corners[:, 0, :].max(axis=1)
        min_y = transformed_corners[:, 1, :].min(axis=1)
        max_y = transformed_corners[:, 1, :].max(axis=1)

        # Intersection of all bounding boxes
        inner_min_x = numpy.max(min_x)  
        inner_max_x = numpy.min(max_x)
        inner_min_y = numpy.max(min_y)
        inner_max_y = numpy.min(max_y)

        if inner_min_x >= inner_max_x or inner_min_y >= inner_max_y:
            raise ValueError("No common inner rectangle found; transformed images do not overlap.")

        return inner_min_x, inner_min_y, inner_max_x, inner_max_y
    

    def _normalise(self, x):
       
        max_v = numpy.max(x)
        min_v = numpy.min(x)

        y = (x - min_v)/(max_v - min_v)

        y = numpy.clip(y, 0, 1).astype(numpy.float32)

        print("original ", numpy.min(x), numpy.max(x))
        print("normalised ", numpy.min(y), numpy.max(y))
        print("\n")

        return y