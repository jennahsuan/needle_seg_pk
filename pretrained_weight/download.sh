#!/bin/bash

# 
DROPBOX_LINK="https://www.dropbox.com/scl/fi/g6asf8clbics15k183me6/memm2f_cls_rB384_pixu_RNA_C2sam2_BLLF20MA2_tciou78_2BqattGP_v6imgLCR_k2_update.pth?rlkey=6mjmtjoqy7yrjp4li12cph7lu&st=307xrrpz&dl=0"
OUTPUT_FILE="memm2f_cls_rB384_pixu_RNA_C2sam2_BLLF20MA2_tciou78_2BqattGP_v6imgLCR_k2_update.pth"

# Convert to direct download link (change ?dl=0 or ?dl=1 to ?raw=1 or ?dl=1)
DIRECT_LINK="${DROPBOX_LINK/\?dl=0/?dl=1}"

# Use curl or wget to download
echo "Downloading file from Dropbox..."
curl -L "$DIRECT_LINK" -o "pretrained_weight/$OUTPUT_FILE"

echo "Download complete: $OUTPUT_FILE"

#
DROPBOX_LINK="https://www.dropbox.com/scl/fi/13v0zf1d4h5ys3mlgwmdt/convnext_mask2former_cls_rB384_pixup_foc20aux1_v6imgLCR_k2.pth?rlkey=zt2n3pc7rys7n8cm1eeusxo86&st=i8305sxo&dl=0"
OUTPUT_FILE="convnext_mask2former_cls_rB384_pixup_foc20aux1_v6imgLCR_k2.pth"

# Convert to direct download link (change ?dl=0 or ?dl=1 to ?raw=1 or ?dl=1)
DIRECT_LINK="${DROPBOX_LINK/\?dl=0/?dl=1}"

# Use curl or wget to download
echo "Downloading file from Dropbox..."
curl -L "$DIRECT_LINK" -o "pretrained_weight/$OUTPUT_FILE"

echo "Download complete: $OUTPUT_FILE"