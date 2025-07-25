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

                warped_img, w, score = self.process(img_base, img_dest,  warp_mode, scale, iterations)
                                
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
    

    def process(self, ref_img, target_img, warp_mode=cv2.MOTION_AFFINE, scale = 2, iterations=128):
        img_result, w_mat = self._align_ecc(ref_img, target_img, warp_mode, scale, iterations)

        score = self._score(ref_img, img_result)

        return img_result, w_mat, score


    

    def _align_ecc(self, ref_img, target_img, warp_mode=cv2.MOTION_AFFINE, scale = 1, iterations=128):
        """
        Aligns target_img to ref_img using ECC with downsampling for speed.

        :param ref_img: Reference image (H, W, 3)
        :param target_img: Target image (H, W, 3)
        :param warp_mode: cv2.MOTION_HOMOGRAPHY or cv2.MOTION_AFFINE
        :param scale: Downscale factor (int > 1)
        :return: aligned image, warp matrix (full resolution)
        """
        gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype(numpy.float32)
        gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY).astype(numpy.float32)

        h, w = gray_ref.shape
        h_small, w_small = h // scale, w // scale

        ref_small = cv2.resize(gray_ref, (w_small, h_small), interpolation=cv2.INTER_AREA)
        target_small = cv2.resize(gray_target, (w_small, h_small), interpolation=cv2.INTER_AREA)

        # Init warp matrix (low-res)
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix_small = numpy.eye(3, 3, dtype=numpy.float32)
        else:
            warp_matrix_small = numpy.eye(2, 3, dtype=numpy.float32)

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 1e-7)

        try:
            cc, warp_matrix_small = cv2.findTransformECC(
                ref_small, target_small, warp_matrix_small,
                motionType=warp_mode,
                criteria=criteria
            )
        except cv2.error as e:
            print("ECC failed:", e)
            return None, None

        # Rescale warp matrix to original resolution
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            S = numpy.array([[scale, 0, 0],
                        [0, scale, 0],
                        [0, 0, 1]], dtype=numpy.float32)

            # Correctly upscale the matrix
            #warp_matrix = numpy.linalg.inv(S) @ warp_matrix_small @ S
            warp_matrix = S @ warp_matrix_small @ numpy.linalg.inv(S)


        else:  # MOTION_AFFINE
            warp_matrix = warp_matrix_small.copy()
            warp_matrix[0, 2] *= scale
            warp_matrix[1, 2] *= scale

        # Warp full-resolution image
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            aligned = cv2.warpPerspective(target_img, warp_matrix, (w, h),
                                        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        else:
            aligned = cv2.warpAffine(target_img, warp_matrix, (w, h),
                                    flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)

        return aligned, warp_matrix


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