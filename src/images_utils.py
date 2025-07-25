import cv2
import numpy

def compute_dynamic_range_db(image, filter_size = None):
    # Convert to grayscale (luminance approximation)
    gray = numpy.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    if filter_size is not None:
        gray = cv2.GaussianBlur(gray, (filter_size, filter_size), 0)
    
    # Avoid zeros to prevent log(0)
    non_zero = gray[gray > 0]
    if non_zero.size == 0:
        return 0.0  # No dynamic range in a black image
    
    max_val = numpy.max(non_zero)
    min_val = numpy.min(non_zero)
    
    dynamic_range_db = 20 * numpy.log10(max_val / min_val)
    return round(dynamic_range_db, 2)


def rate_image(img, exposure_bias=0.5, exp_power=2.0, sharp_weight=1.0, color_weight=1.0):

    # 1. Exposure weight (favor mid-gray)
    luminance = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    exposure_w = 1.0 - numpy.abs(luminance - exposure_bias) ** exp_power  # higher near mid
    exposure_w = numpy.clip(exposure_w, 0.0, 1.0)

    # 2. Sharpness weight (using Laplacian)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    sharpness_w = cv2.GaussianBlur(numpy.abs(laplacian), (5, 5), 0)
    sharpness_w = sharpness_w / (sharpness_w.max() + 1e-6)

    # 3. Colorfulness weight (Hasler-Süsstrunk method)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    colorfulness = numpy.sqrt(rg**2 + yb**2)
    colorfulness_w = cv2.GaussianBlur(colorfulness, (5, 5), 0)
    colorfulness_w = colorfulness_w / (colorfulness_w.max() + 1e-6)


    # 4, exposition range
    underexposed = luminance < 0.05
    overexposed  = luminance > 0.95

    exposition = 1.0 - numpy.logical_or(underexposed, overexposed)

    # Combine weights
    result_w = (
        exposure_w * (1.0 + exposition) *
        (1.0 + sharp_weight * sharpness_w) * 
        (1.0 + color_weight * colorfulness_w)
    )

    result_w = numpy.array(result_w, dtype=numpy.float32)

    return result_w



def rate_images(images):
    ratings = []
    dynamical_range = []
    for n in range(len(images)):
        db = compute_dynamic_range_db(images[n], 11)
        dynamical_range.append(db)
    
        rating = rate_image(images[n]).min()
        ratings.append(rating)

    return numpy.array(ratings), numpy.array(dynamical_range)