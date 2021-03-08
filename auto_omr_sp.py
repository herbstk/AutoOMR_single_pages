#!/usr/bin/env python3
# single page paper questionnaire registration pipeline
# 2021 Konrad Herbst, <k.herbst@zmbh.uni-heidelberg.de>

# much of the actual image analysis logic is based on code from the
# RescueOMR package by Yuri D'Elia (EURAC, Institute of Genetic Medicine)
# https://github.com/EuracBiomedicalResearch/RescueOMR

# TODO tidy up and distribute dependencies
import argparse
import sys
import re
import warnings
import logging
import os
import shutil
from time import sleep
import signal
from datetime import datetime

# TODO import only the necessary functions?
import skimage.feature
import skimage.measure
import skimage.transform
import skimage.util
import skimage.io
from skimage.filters import gaussian
import numpy as np

# from pyzbar.pyzbar import decode, ZBarSymbol

# TODO remove dependency on PIL
from PIL import Image, ImageDraw, ImageOps
import scipy as sp
import scipy.ndimage
import lxml.etree
from io import BytesIO
import base64

import matplotlib.pyplot as plt

from joblib import Parallel, delayed, hash

# primitive interruption handling
def signal_handler(signal, frame):
    global interrupted
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)

### SETTINGS ###
WORKERS = 1

# outdir set-up
OUTDIR_DIRS = {
    'processing' : 'processing',
    'failed'     : 'processed_failed',
    'done'       : 'processed',
    'done_crops' : 'processed/crops',
    'raw'        : 'processed_raw_files'
}

# templatedir set-up
# TEMPLATE_REGISTRATION = 'scan_form_pg1_template.png'
TEMPLATE_FORM = ('scan_form_pg1.svg')

# template specific settings
GAUSSIAN_SIGMA  = .7
EDGE_REGIONS = [(300, 300, 0, 0),        # upper left corner (w, h, x, y)
                (300, 300, 0, 3200),     # lower left corner
                (300, 300, 2200, 0),     # upper right corner
                (400, 400, 2160, 3150)]  # bottom right corner

PAGEMARK_CIRCLE_MIN = 45
PAGEMARK_CIRCLE_MAX = 55
PAGEMARK_CIRCLE_EMPTY_FRAC = 0.1

# tuned for 300 dpi grayscale text
CORNER_SIGMA    = 3
CORNER_MIN_DIST = 3

CANNY_THR_LOW  = .01
CANNY_THR_HIGH = .3

FEATURE_CROP_WIN = 47   # must be odd
FEATURE_SSIM_WIN = 23   # roughly CROP_WIN/2 (odd)
FEATURE_SSIM_K   = 4.
FEATURE_SSIM_THR = 0.999

FEATURE_MIN_SMP  = 20

RANSAC_TRIALS  = 1200 # original: 12000
RANSAC_MIN_THR = 0.70

RES_MAX_ROT   = 0.08  # (rad) ~5deg
RES_MAX_SHEAR = 0.01  # 0.1%
RES_MAX_SCALE = 0.1   # 1%

# tuned for 300 dpi grayscale text
BLACK_LEVEL = 0.90
OVRF_THR    = 0.12
FILL_THR    = 0.02
VOID_THR    = 0.004

# H/V line rejection
CLEAN_LEN = 47  # window length (must be odd)
CLEAN_W   = 3   # line width-1 (even)
CLEAN_THR = 0.9 # rejection threshold
CLEAN_CROP = .05 # additional border to include in cropped regions to clear (as fraction of crop width/height)

# SVG specific for embedded images
PREFIX = 'data:image/png;base64,'
ATTR   = '{http://www.w3.org/1999/xlink}href'

# create a path if not already there
def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
        return True
    else:
        return False

def plot_with_marks(image, coords):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)
    # ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',
            # linestyle='None', markersize=6)
    ax.plot(coords[:, 1], coords[:, 0], '+r', markersize=15)
    # ax.axis((0, 310, 200, 0))
    plt.show()

def load_image(path):
    try:
        image = skimage.io.imread(path, as_gray=True)
    except (OSError, ValueError) as err:
        sleep(3) # maybe image is still written?
        try:
            image = skimage.io.imread(path, as_gray=True)
        except (OSError, ValueError) as err:
            # print('Image "{} could not be loaded...'.format(path))
            return None
    image = skimage.util.img_as_float(image)
    return image


def save_image(image, path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        image = skimage.util.img_as_ubyte(image)
    if os.path.exists(path):
        path = os.path.splitext(path)
        path = '{}-1{}'.format(path[0], path[1])
    skimage.io.imsave(path, image)

## REGISTRATION
def detect_features(image, reg = None):
    if reg is None:
        data = skimage.feature.corner_harris(image, sigma=CORNER_SIGMA)
        peaks = skimage.feature.corner_peaks(data, min_distance=CORNER_MIN_DIST, threshold_rel = 0.1)
    else:
        image = crop_region(image, reg)
        data = skimage.feature.corner_harris(image, sigma=CORNER_SIGMA)
        peaks = skimage.feature.corner_peaks(data, min_distance=CORNER_MIN_DIST, threshold_rel = 0.1)
        peaks = peaks + [reg[3], reg[2]]
    return peaks

def detect_edge_features(image, regions):
    features = []
    for reg in regions:
        peaks = detect_features(image, reg)
        features.append(peaks)
        # np.vstack((features, peaks))
    # return features
    return np.vstack((features))

def _crop_axis(dim, pos, width):
    hw = width // 2
    x1 = min(dim, max(0, pos - hw))
    p1 = abs(pos - hw) - x1
    x2 = min(dim, max(0, pos + hw))
    p2 = abs(pos + hw) - x2
    return x1, x2, p1, p2

def crop_padded(im, pos, width):
    assert(width % 2)
    y1, y2, py1, py2 = _crop_axis(im.shape[0], pos[0], width)
    x1, x2, px1, px2 = _crop_axis(im.shape[1], pos[1], width)
    roi = im[y1:y2, x1:x2]
    pad = [[py1, py2], [px1, px2]]
    return np.pad(roi, pad, 'reflect')

def crop_region(im, reg):
    w, h, x, y = reg
    return im[y:y+h, x:x+w]

def extract_features(image, points, win):
    return [crop_padded(image, point, win) for point in points]

def parse_region(text):
    grp = re.match(r'^(\d+)x(\d+)\+(\d+)\+(\d+)$', text)
    if grp is None:
        return None
    return (int(grp.group(1)), int(grp.group(2)),
            int(grp.group(3)), int(grp.group(4)))


class ImageData():
    def __init__(self, image, points, feats):
        self.image = image
        self.points = points
        self.feats = feats


def analyze_image(image):
    # points = detect_features(image)
    points = detect_edge_features(image, EDGE_REGIONS)
    feats = extract_features(image, points, FEATURE_CROP_WIN)
    return ImageData(image, points, feats)


class ConstrainedAffineTransform(skimage.transform.AffineTransform):
    def __init__(self, *args, **kwargs):
        return super(ConstrainedAffineTransform, self).__init__(*args, *kwargs)

    def estimate(self, src, dst):
        ret = super(ConstrainedAffineTransform, self).estimate(src, dst)
        if ret is False:
            return False
        if abs(self.rotation) > RES_MAX_ROT:
            return False
        if abs(self.shear) > RES_MAX_SHEAR:
            return False
        if abs(1 - self.scale[0]) > RES_MAX_SCALE or \
           abs(1 - self.scale[1]) > RES_MAX_SCALE:
            return False
        return True

def extract_template(templ, image):
    log = []
    # check arguments
    rs_min_smp = int(len(templ.feats) * RANSAC_MIN_THR)
    abs_min_smp = max(FEATURE_MIN_SMP, rs_min_smp)
    if len(templ.feats) < abs_min_smp:
        log.append('insufficient features in template ({}, min={})'.format(
            len(templ.feats), abs_min_smp))
        return (None, log)
    if len(image.feats) < abs_min_smp:
        log.append('insufficient features in image ({}, min={})'.format(
            len(image.feats), abs_min_smp))
        return (None, log)

    # match points
    matches = []
    for pt1, ft1 in zip(templ.points, templ.feats):
        for pt2, ft2 in zip(image.points, image.feats):
            ret = skimage.metrics.structural_similarity(ft1, ft2, win_size=FEATURE_SSIM_WIN,
                                                         K1=FEATURE_SSIM_K, K2=FEATURE_SSIM_K)
            if ret < FEATURE_SSIM_THR:
                continue
            matches.append([pt1, pt2, ret])
    if len(matches) < abs_min_smp:
        log.append('insufficient matching features ({}, min={})'.format(
            len(matches), abs_min_smp))
        return (None, log)

    # find a matching transform
    matches = np.array(matches, dtype = object)
    m1 = np.array(list(matches[:,0]))
    m2 = np.array(list(matches[:,1]))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        model, inliers = skimage.measure.ransac((m1, m2), ConstrainedAffineTransform,
                                                min_samples=3, residual_threshold=3,
                                                max_trials=RANSAC_TRIALS)
    inliers = matches[inliers]
    log.append('model inliers ({} of {}, min={})'.format(len(inliers), len(matches), rs_min_smp))
    if len(inliers) < rs_min_smp:
        log.append('insufficient model inliers ({} of {}, min={})'.format(
            len(inliers), len(matches), rs_min_smp))
        return (None, log)

    # extract the sub-image
    tr = skimage.transform.AffineTransform(translation=[model.translation[1],
                                                        model.translation[0]],
                                           rotation=-model.rotation,
                                           shear=-model.shear,
                                           scale=[model.scale[1],
                                                  model.scale[0]])
    ret = skimage.transform.warp(image.image, tr, output_shape=templ.image.shape, order=3)
    return (ret, log)

def extract_page(image, reg):
    image = crop_region(image, reg)
    # detect edges
    edges = skimage.feature.canny(image, sigma=3, low_threshold=CANNY_THR_LOW, high_threshold=CANNY_THR_HIGH)

    # Detect radii
    hough_radii = np.arange(PAGEMARK_CIRCLE_MIN, PAGEMARK_CIRCLE_MAX, 2)
    hough_res = skimage.transform.hough_circle(edges, hough_radii)

    # Select the most prominent circle
    accums, cx, cy, radii = skimage.transform.hough_circle_peaks(hough_res, hough_radii,
                                                                 total_num_peaks=1)

    # calculate page number
    x = skimage.draw.disk((cy[0], cx[0]), radii[0], shape=image.shape)
    frac = np.sum(image[x] < BLACK_LEVEL) / len(image[x])
    page = round((frac - PAGEMARK_CIRCLE_EMPTY_FRAC) *  4)
    # print('frac = {}; page = {}'.format(frac, page))
    return page

def extract_barcode(image, reg = None):
    if reg is not None:
        image = crop_region(image, reg)
    barcode = decode(image  * 255, [ZBarSymbol.CODE128])
    return barcode



## OMR ################

# calculate filter for roi cleaning (from checkbox outline)
## filter is a cross of size s and -1 as values with a cross of +1 in the center
s = CLEAN_LEN
w = CLEAN_W
k = -np.ones(shape=(s, s))
k[:,s//2-w+1:s//2+w] = 1
k[s//2-w+1:s//2+w,:] = 1

# cleans only image parts which are present in rois
def clean_image(image, rois = None):
    if rois is None:
        tmp = sp.ndimage.convolve(image/255, k) / np.sum(k)
        ret = image.copy()
        ret[tmp > CLEAN_THR] = 255
    else:
        ret = np.full(shape = image.shape, fill_value = 255, dtype = np.uint8)
        for roi in rois:
            # be more generous with cropping (to remove also H/V lines at the crop border)
            w, h, x, y = roi
            ws = round(w * CLEAN_CROP)
            hs = round(h * CLEAN_CROP)
            roi = (w+ws, h+hs, x-ws//2, y-hs//2)
            img = crop_region(image, roi)
            tmp = sp.ndimage.convolve(img/255, k) / np.sum(k)
            nimg = img.copy()
            nimg[tmp > CLEAN_THR] = 255
            w, h, x, y = roi
            ret[y:y+h, x:x+w] = nimg
    return ret

def fill(image):
    return (image < BLACK_LEVEL * 255).sum() / (image.shape[0] * image.shape[1])

def _svg_translate(tag, tx=0, ty=0):
    if tag is None:
        return tx, ty
    trn = tag.get('transform')
    if trn is not None:
        grp = re.match(r'^translate\(([-\d.]+),([-\d.]+)\)$', trn)
        if grp is None:
            logging.error('SVG node contains unsupported transformations!')
            sys.exit(1)
        tx += float(grp.group(1))
        ty += float(grp.group(2))
    return _svg_translate(tag.getparent(), tx, ty)

def load_svg(path):
    data = lxml.etree.parse(path).getroot()
    shape = ( int(float(data.get('height'))), int(float(data.get('width'))) )
    # extract (first) embedded image
    href = data.find('.//{*}image')
    if href is not None:
        href = href.get(ATTR)
        if href.startswith(PREFIX):
            image = load_image(BytesIO(base64.b64decode(href[len(PREFIX):]))) * 255
            image = skimage.transform.resize(image, shape, anti_aliasing = True)
    else:
        # default to white background
        image = np.ones(shape = shape)  ## np.full(shape, 255) // np.zeros(shape = shape)
    rects = []
    for tag in data.iterfind('.//{*}rect'):
        tx, ty = _svg_translate(tag)
        i = tag.get('id')
        x = int((float(tag.get('x')) + tx))
        y = int((float(tag.get('y')) + ty))
        w = int(float(tag.get('width')))
        h = int(float(tag.get('height')))
        roi = crop_region(image, (w, h, x, y))
        roi[ roi > BLACK_LEVEL * 255 ] = 255
        b = fill(clean_image(roi)) # with removed H/V lines
        # b = fill(roi)
        rects.append((i, x, y, w, h, b))
    return (image, rects)

def scan_marks(image, marks):
    res = []
    for i, x, y, w, h, b in marks:
        roi = crop_region(image, (w, h, x, y))
        scr = fill(roi) #- b
        if scr > OVRF_THR:
            v = 2
        elif scr > FILL_THR:
            v = 1
        elif scr < VOID_THR:
            v = 0
        else:
            v = -1
        res.append((i, v, scr, b))
    return res


def debug_marks(path, image, clean, marks, res):
    buf = Image.new('RGB', image.shape[::-1])
    buf.paste(Image.fromarray(image, 'L'))
    draw = ImageDraw.Draw(buf, 'RGBA')
    for mark, row in zip(marks, res):
        i, x, y, w, h, b = mark
        v = row[1]
        if v == 1:
            c = (255, 0, 0, 127)
        elif v == 0:
            c = (0, 255, 0, 127)
        elif v == 2:
            c = (0, 0, 0, 64)
        else:
            c = (255, 127, 0, 127)
        draw.rectangle((x, y, x+w, y+h), c)
    bw = clean.copy()
    thr = bw < BLACK_LEVEL * 255
    bw[thr] = 255
    bw[~thr] = 0
    buf.paste((0, 127, 255),
              (0, 0, image.shape[1], image.shape[0]),
              Image.fromarray(bw, 'L'))
    if os.path.exists(path):
        path = os.path.splitext(path)
        path = '{}-1{}'.format(path[0], path[1])
    buf.save(path)

def omr(scan, subdir, outdirs, templ, marks, debug):
    log = ['Log for image "{}"'.format(scan)]
    sleep(3) ## wait, in case file is still written
    ### move scans to processing folder
    scan_file = shutil.move(os.path.join(subdir, scan), outdirs['processing'])
    ### register scan
    image = load_image(scan_file)
    if image is None:
        log.append('Unable to load image.')
        return log
    image = skimage.filters.gaussian(image, sigma = GAUSSIAN_SIGMA)

    # analyze image
    image = analyze_image(image)
    log.append('Found {} features in image ({} features in template)'.format(len(image.feats), len(templ.feats)))

    # extract matching template from image
    match, matchlog = extract_template(templ, image)
    log.extend(matchlog)
    if match is None:
        log.append('Registration failed. Rotating image 180° and try once more.')
        image = skimage.transform.rotate(image.image,  180)
        # analyze image
        image = analyze_image(image)
        log.append('Found {} features in image ({} features in template)'.format(len(image.feats), len(templ.feats)))
        # extract matching template from image
        match, matchlog = extract_template(templ, image)
        log.extend(matchlog)
        if match is None:
            log.append('Registration failed. Moving raw file to {}'.format(outdirs['failed']))
            scan_file = shutil.move(scan_file, outdirs['failed'])
            if debug:
                with open('{}-log.txt'.format(os.path.splitext(scan_file)[0]), 'w') as f:
                    f.write('\n'.join(log))
            return log

    # # extract barcode
    # barcode = extract_barcode(match, EDGE_REGIONS[4])
    # if len(barcode) != 1:
    #     scan_file = shutil.move(scan_file, outdirs['failed'])
    #     log.append('Found {} barcodes in image'.format(len(barcode)))
    #     if debug:
    #         with open('{}-log.txt'.format(os.path.splitext(scan_file)[0]), 'w') as f:
    #             f.write('\n'.join(log))
    #     return log
    # id = barcode[0].data.decode('ASCII')
    # log.append('image has ID {}'.format(id))
    # # extract page number
    # page = extract_page(match, EDGE_REGIONS[3])
    # log.append('image belongs to page {}'.format(page))

    # create a unique identifier for page
    ## the modification time should be unique, as this is the time when the scan was saved
    ctime = datetime.fromtimestamp(os.stat(scan_file).st_mtime)
    id = ctime.strftime('%Y%m%d-%H%M%S') ## resulting format 'YearMonthDay-HourMinuteSecond'
    hash6 = hash(image.image)[0:6] ## add first 6 chr of MD5-hashed image data to further diversify
    id = '{}-{}'.format(id, hash6)

    # use a simple OMR algorithm to classify marked regions (checkboxes etc.)
    ## image needs to be scaled 0...255 (simpleomr script requirement)
    img = skimage.util.img_as_ubyte(match)
    rois = [(w, h, x, y) for i, x, y, w, h, b in marks]
    img[ img > BLACK_LEVEL * 255 ] = 255
    clean = clean_image(img, rois) # with removed H/V lines
    res = scan_marks(clean, marks)

    ## TODO write filled TB into separate directory
    for i in range(len(res)):
        if res[i][0].find('TB') == 0 and res[i][2] > 0:
            tb_dir = os.path.join(outdirs['done_crops'], res[i][0])
            mkdir(tb_dir)
            r = [marks[i][j] for j in (3,4,1,2)]
            tb = crop_region(img, r)
            save_image(tb, os.path.join(tb_dir, '{}_{}.png'.format(id, res[i][0])))

    # write out results
    # output registered image
    save_image(match, os.path.join(outdirs['done'], '{}.png'.format(id)))

    if debug:
        debug_marks(os.path.join(outdirs['done'], '{}_debug.png'.format(id)), img, clean, marks, res) # with removed H/V lines

    if debug:
        with open(os.path.join(outdirs['done'], '{}.tsv'.format(id)), 'w') as f:
            for i, v, scr, b in res:
                f.write('{}\t{}\t{}\t{}\n'.format(i, v, scr, b))
        with open(os.path.join(outdirs['done'], '{}-log.txt'.format(id)), 'w') as f:
            f.write('\n'.join(log))
    else:
        with open(os.path.join(outdirs['done'], '{}.tsv'.format(id)), 'w') as f:
            for i, v, _, __ in res:
                f.write('{}\t{}\n'.format(i, v))

    shutil.move(scan_file, outdirs['raw'])
    return log

interrupted = False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('templatedir', help = 'template directory')
    ap.add_argument('outdir', help = 'output directory')
    ap.add_argument('-d', dest='debug', action='store_true', help='Debug mode')
    ap.add_argument('-v', dest='verbose', action='count', default=0, help='Increase verbosity')
    args = ap.parse_args()

    levels = (logging.WARNING, logging.INFO, logging.DEBUG)
    logging.basicConfig(level=levels[min(len(levels)-1, args.verbose)])

    # prepare output directory
    ## if output_dir already exists throw a warning and stop
    if not mkdir(args.outdir) and len(os.listdir(args.outdir)) > 0:
    #     for dir in OUTDIR_DIRS.values():
    #         mkdir(os.path.join(args.outdir, dir))
    # else:
        logging.info('The outdir already exists and is not empty. Please clear manually or provide a different outdir.')
        return 1

    # load and process templates
    if not os.path.isdir(args.templatedir):
        logging.info('The templatedir doesn\'t exist.')
        return 1
    else:
        # f = os.path.join(args.templatedir, TEMPLATE_REGISTRATION)
        # templ = load_image(f)
        f = os.path.join(args.templatedir, TEMPLATE_FORM)
        templ, marks = load_svg(f)
        templ = templ / 255
        logging.info('Loaded templatefile "{}" for registration and OMR marks.'.format(f))
        templ = skimage.filters.gaussian(templ, sigma = GAUSSIAN_SIGMA)
        templ = analyze_image(templ)
        logging.info('found {} features in template'.format(len(templ.feats)))

        logging.info('found {} marks in template {}'.format(len(marks), f))

    # loop for automatic processing
    while True:
        ## acquire scans to process from subdirectories of current directory (if none found we might not be in the right directory??)
        scans_todo = dict()
        with os.scandir('./') as it:
            for entry in it:
                if not entry.name.startswith('.') and entry.is_dir():
                    scans_todo[entry.name] = [x.name for x in os.scandir(entry.name) if x.name.endswith('.pnm')]   # filter for pnm files in current working directory
        if len(scans_todo) == 0:
            logging.info('So far no subdirectories exist in current working directory. Idle for 3 sec...')
            sleep(3)
            continue
        else:
            logging.info('Found {} files in {} subdirectories of the current working directory.'.format(len([s for l in scans_todo.values() for s in l]),
                                                                                                    len(scans_todo)))
        ### loop over found files
        for subdir, scans in scans_todo.items():
            mkdir(os.path.join(args.outdir, subdir))
            outdirs = dict()
            for k, v in OUTDIR_DIRS.items():
                outdirs[k] = os.path.join(args.outdir, subdir, v)
                mkdir(outdirs[k])

            if WORKERS == 1:
                # non-paralleled processing
                logs = [omr(scan, subdir, outdirs, templ, marks, args.debug) for scan in scans]
            else:
                Parallel(n_jobs = WORKERS)(delayed(omr)(scan, subdir, outdirs, templ, marks, args.debug) for scan in scans)

        ### run Rscript to aggregate data?
        ### write out logs??
        ## if no scans found sleep for a sec.
        if interrupted:
            print("Requested pipeline shutdown.")
            break
        logging.info('Done with this batch of files. Shutdown pipeline with Ctrl+C. Idle for 3 sec...')
        sleep(3)

if __name__ == '__main__':
    sys.exit(main())
