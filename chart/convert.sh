#!/bin/bash
set -eux
tensorflowjs_converter --input_format keras model_k.h5 converted_k
tensorflowjs_converter --input_format keras model_k_sigma.h5 converted_k_sigma
tensorflowjs_converter --input_format keras model_w.h5 converted_w
tensorflowjs_converter --input_format keras model_kk.h5 converted_kk
tensorflowjs_converter --input_format keras model_cc.h5 converted_cc
tensorflowjs_converter --input_format keras model_ck.h5 converted_ck
tensorflowjs_converter --input_format keras model_o.h5 converted_o
tensorflowjs_converter --input_format keras model_ow.h5 converted_ow
tensorflowjs_converter --input_format keras model_perc_a.h5 converted_perc_a
tensorflowjs_converter --input_format keras model_perc_d.h5 converted_perc_d
