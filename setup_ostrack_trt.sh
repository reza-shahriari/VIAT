#!/bin/bash

# Exit on error
set -e

PRECISION="${1:-fp16}"

echo "================================================================="
echo "       OSTrack TensorRT Engine Exporter & Converter              "
echo "================================================================="
echo "Target Precision: ${PRECISION}"
echo "System Python: $(which python3 || echo /usr/bin/python3)"
echo "TensorRT Compiler: $(which trtexec || echo /usr/bin/trtexec)"
echo "-----------------------------------------------------------------"

# Step 1: Export ONNX model using system python
echo "[Step 1/2] Exporting OSTrack model to ONNX..."
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin /usr/bin/python3 export_ostrack_onnx.py

ONNX_FILE="checkpoints/ostrack.onnx"
ENGINE_FILE="checkpoints/ostrack.engine"

if [ ! -f "${ONNX_FILE}" ]; then
    echo "Error: ONNX file ${ONNX_FILE} was not created!"
    exit 1
fi

# Step 2: Convert ONNX to TensorRT Engine
echo "[Step 2/2] Compiling ONNX to TensorRT Engine (${ENGINE_FILE})..."

TRTEXEC_BIN="/usr/bin/trtexec"
if [ ! -f "${TRTEXEC_BIN}" ]; then
    TRTEXEC_BIN="trtexec"
fi

# Build trtexec command
TRT_CMD="${TRTEXEC_BIN} --onnx=${ONNX_FILE} --saveEngine=${ENGINE_FILE}"

# Append precision flags if supported
if [[ "${PRECISION}" == "fp16" ]]; then
    # TensorRT 10/11 precision check
    if ${TRTEXEC_BIN} --help | grep -q "\--precision"; then
        TRT_CMD="${TRT_CMD} --precision=fp16"
    elif ${TRTEXEC_BIN} --help | grep -q "\--fp16"; then
        TRT_CMD="${TRT_CMD} --fp16"
    fi
elif [[ "${PRECISION}" == "bf16" ]]; then
    if ${TRTEXEC_BIN} --help | grep -q "\--precision"; then
        TRT_CMD="${TRT_CMD} --precision=bf16"
    fi
fi

echo "Running command: ${TRT_CMD}"
eval "${TRT_CMD}"

echo "================================================================="
echo " Success! TensorRT Engine created at: ${ENGINE_FILE}"
echo " File Size: $(du -h "${ENGINE_FILE}" | cut -f1)"
echo "================================================================="
