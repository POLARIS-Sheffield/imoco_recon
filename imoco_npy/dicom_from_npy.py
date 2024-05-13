#!/usr/bin/env python3
"""
Conversion of numpy array output of imoco/mocolor recon to DICOM series - adapted from imoco_py/dicom_creation.py
https://github.com/PulmonaryMRI/imoco_recon/blob/master/imoco_py/dicom_creation.py
Command line inputs needed: raw data (numpy array) directory; original DICOM directory
Modified by: Neil J Stewart njstewart-eju (2023/08)
Original author: Fei Tan (ftan1)
"""
import numpy as np
import pydicom as pyd
import datetime
import os
import glob
import sys
import argparse
import logging
import re

if __name__=='__main__':
    # parse arguments
    parser = argparse.ArgumentParser(
         description='Convert numpy array to DICOM')
    
    parser.add_argument('input_file', type=str, help='path to numpy file')
    parser.add_argument('orig_dicom_dir', type=str, help='path to DICOM directory')
    args = parser.parse_args()

    input_file = args.input_file
    orig_dicom_dir = args.orig_dicom_dir
    logging.basicConfig(level=logging.INFO)

    # set directories
    im = np.load(input_file)
    new_dicom_dir = input_file[:-4] + '_dcm'

    # load original dicom
    # exam number, series number
    dcm_ls = glob.glob(orig_dicom_dir + '/*.dcm')
    # Sort by slice number
    dcm_ls = sorted(dcm_ls,key=lambda s: int(re.findall(r'\d+',s)[-1]))
    series_mimic_dir = dcm_ls[-1]
    ds = pyd.dcmread(series_mimic_dir)
    # parse exam number, series number
    dcm_file = os.path.basename(series_mimic_dir)
    exam_number, series_mimic, _ = dcm_file.replace('Exam', '').replace(
        'Series', '_').replace('Image', '_').replace('.dcm', '').split('_')
    exam_number = int(exam_number)

    im_shape = np.shape(im)
    ds.Columns, ds.Rows = im_shape[-2], im_shape[-1]

    # Edit spatial resolution according to recon to 255 instead of 256
    spatial_resolution = ds.PixelSpacing[0] * (len(dcm_ls)/ds.Columns)

    # Update SliceLocation information
    series_mimic_slices = np.double(ds.Columns)  # assume recon is isotropic
#    ds.SpacingBetweenSlices = spatial_resolution
#    ds.SliceThickness = spatial_resolution
    ds.ReconstructionDiameter = spatial_resolution * im_shape[-1]
    # Update pixel spacing according to 255 vs 256 pixels
    ds.PixelSpacing = spatial_resolution

    SliceLocation_original = ds.SliceLocation
    ImagePositionPatient_original = ds.ImagePositionPatient 
    # Shift slice location by one pixel to make IPP match
    ImagePositionPatient_original[-1] = ImagePositionPatient_original[-1] - spatial_resolution
    
    logging.info(f'IPP original: {ImagePositionPatient_original}')

    im = np.transpose(im, axes=[0, 2, 1])
    # Comment this line to flip images up-down
    im = np.flip(im,0)

    try:
        os.mkdir(new_dicom_dir)
    except OSError as error:
        pass

    ds.SeriesDescription = '3D UTE - imoco' 
    # adding 10, this should ensure no overlap with other series numbers for Philips numbering which uses series number (in order acquired) *100
    series_write = int(series_mimic) + 10

    # modified time
    dt = datetime.datetime.now()

    im = np.abs(im) / np.amax(np.abs(im)) * 4095 # 65535 
    im = im.astype(np.uint16)

    # Window and level for the image
    ds.WindowCenter = int(np.amax(im) / 2)
    ds.WindowWidth = int(np.amax(im))

    # dicom series UID
    ds.SeriesInstanceUID = pyd.uid.generate_uid()

    # not currently accounting for oblique slices...
    for z in range(ds.Rows):
        ds.InstanceNumber = z + 1
        ds.SeriesNumber = series_write
        ds.SOPInstanceUID = pyd.uid.generate_uid()
        # SOPInstanceUID should == MediaStorageSOPInstanceUID
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        Filename = '{:s}/E{:d}S{:d}I{:d}.DCM'.format(
            new_dicom_dir, exam_number, series_write, z + 1)
        ds.SliceLocation = SliceLocation_original + \
            (im_shape[-1] / 2 - (z + 1)) * spatial_resolution
        ds.ImagePositionPatient = pyd.multival.MultiValue(float, [float(ImagePositionPatient_original[0]), float(
            ImagePositionPatient_original[1]), ImagePositionPatient_original[2] + (z + 1) * spatial_resolution])
        b = im[z, :, :].astype('<u2')
        ds.PixelData = b.T.tobytes()
        #ds.is_little_endian = False
        ds.save_as(Filename)
