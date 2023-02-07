"""Now try running all these results for BC04 settings.

Run with:
    python analysis/results_bc04.py
    python analysis/results_bc04.py --show_raw_perf
The first shows _normalized_ performance, the second shows _raw_ performance.

Note that these are not the settings used to report in the original CoRL 2022 submission.
For the CNN v2 I'm using and reporting:

Actor(
  (encoder): PixelEncoder(
    (convs): ModuleList(
      (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2))
      (1): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
      (2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
      (3): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1))
    )
    (fc): Linear(in_features=29584, out_features=100, bias=True)
    (ln): LayerNorm((100,), eps=1e-05, elementwise_affine=True)
  )
  (trunk): Sequential(
    (0): Linear(in_features=100, out_features=256, bias=True)
    (1): ReLU()
    (2): Linear(in_features=256, out_features=256, bias=True)
    (3): ReLU()
    (4): Linear(in_features=256, out_features=6, bias=True)
  )
)

This is not what's in the repo (only difference is the repo has 32 conv filters, but
uses 50 feature dim). I think it makes it less suspicious to use 100 instead of 50 for
the feature dim, as 50 is REALLY compressing it. (Edit: wait I think I'm now using 50
by default, which is what CURL used. I see minor performance differences in any case.)
Anyway I think we should just say it's based on the CURL code?
"""
import os
from os.path import join
import ast
import json
import argparse
import numpy as np
np.set_printoptions(linewidth=180, suppress=True, precision=4, edgeitems=12)

# ------------------------------------------------------------------------- #
# Matplotlib stuff, adjust settings as needed.
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
titlesize = 33
xsize = 31
ysize = 31
ticksize = 29
legendsize = 18
er_alpha = 0.25
lw = 4
smooth_w = 5
# ------------------------------------------------------------------------- #

# Directories
SAVE_PATH = './data/plots/'
DATA_PATH = '/data/seita/softagent_mm/'

# Use this in case we use normalized performance.
DEMO_PERF = dict(
    PourWater=0.906,
    PourWater_6DoF=0.815,
    MMOneSphere=0.632,
    MMOneSphere_6DoF=1.000,
    MMOneSphereTransOnly=0.832,  # translation only variant has higher success rates
    MMMultiSphere=0.623,
)

# ================================================================================================ #
# ================================================================================================ #
# All the files have to be on my machine in `DATA_PATH`!!

MM_ONE_SPHERE_TRANS_ONLY = dict(
    NAIVE_CLASS_VECTOR_MSE = [
        'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_ee_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_21_10_34_07_0001',
        'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_ee_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_21_10_34_07_0002',
        'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_ee_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_21_10_34_07_0003',
        'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_ee_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_21_10_34_07_0004',
        'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_ee_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_21_10_34_07_0005',
    ],
    SEGM_PN2_AVG_LAYER = [  # PN++ with averaging layer
        'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_avg_ee_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_22_16_11_37_0001',
        'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_avg_ee_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_22_16_11_37_0002',
        'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_avg_ee_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_22_16_11_37_0003',
        'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_avg_ee_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_22_16_11_37_0004',
        'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_avg_ee_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_22_16_11_37_0005',
    ],
    FLOW3D_SVD_PW_CONSIST_0_1=[  # with translation only the SVD can only introduce noise right?
        'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_21_12_18_13_0001',
        'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_21_13_31_23_0001',
        'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_21_14_44_49_0001',
        'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_21_14_44_49_0002',
        'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_21_14_44_49_0003',
    ],
)

# ======================================================================================= #
# =================================== 6DoF SCOOPING ===================================== #
# ======================================================================================= #

MM_ONE_SPHERE_6DOF = dict(
# ---------------------------------- OUR METHOD ----------------------------------- #

# All of these use Eddie's axes conversion fix from 08/16.
FLOW3D_SVD_PW_CONSIST_0_0=[
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_09_50_18_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_09_50_18_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_09_50_18_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_09_50_18_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_09_50_18_0005',
],
FLOW3D_SVD_PW_CONSIST_0_1=[
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_00_09_29_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_00_09_29_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_00_09_29_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_00_09_29_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_00_09_29_0005',
],
FLOW3D_SVD_PW_CONSIST_0_5=[
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_09_54_38_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_09_54_38_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_09_54_38_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_09_54_38_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_09_54_38_0005',
],
FLOW3D_SVD_PW_CONSIST_1_0=[
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_18_10_54_05_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_18_10_54_05_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_18_10_54_05_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_18_10_54_05_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_18_10_54_05_0005',
],

# --------------------------------------- ABLATIONS ----------------------------------- #

FLOW3D_SVD_PW_CONSIST_NOSKIP=[
    # Used Eddis' axes conversion from 08/16.
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_NOSKIP_scalePCL_noScaleTarg_2022_08_17_11_13_15_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_NOSKIP_scalePCL_noScaleTarg_2022_08_17_11_13_15_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_NOSKIP_scalePCL_noScaleTarg_2022_08_17_11_13_15_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_NOSKIP_scalePCL_noScaleTarg_2022_08_17_11_13_15_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_NOSKIP_scalePCL_noScaleTarg_2022_08_17_11_13_15_0005',
],
FLOW3D_PW_BEFORE_SVD_NOCONSISTENCY=[
    # Used Eddis' axes conversion from 08/16.
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_PW_bef_SVD_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_16_14_33_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_PW_bef_SVD_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_16_14_33_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_PW_bef_SVD_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_16_14_33_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_PW_bef_SVD_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_16_14_33_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_PW_bef_SVD_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_16_14_33_0005',
],
FLOW3D_MSE_AFTER_SVD_CONSISTENCY=[
    # Wait did this do axes conversion? Yes, since Eddie implemented that on 08/16 so this must have used it.
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_15_56_36_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_15_56_36_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_15_56_36_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_15_56_36_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_15_56_36_0005',
],

# --------------------------------------- BASELINES ------------------------------------- #

DIRECT_VECTOR_MSE = [
    ## Wait, these must have done the un-necessary SVD corrections at runtime.
    # These had a max of 0.792, avg of 0.572.
    #'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_00_16_41_0001',
    #'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_00_16_41_0002',
    #'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_00_16_41_0003',
    #'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_00_16_41_0004',
    #'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_00_16_41_0005',
    # I think we want these (they are a _slight_ improvement):
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_20_22_36_03_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_20_22_36_03_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_20_22_36_03_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_20_22_36_03_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_20_22_36_03_0005',
],
DIRECT_VECTOR_PW = [
    ## This did the axes conversion at runtime, which I think we want now?
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_classif_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_15_32_38_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_classif_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_15_32_38_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_classif_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_15_32_38_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_classif_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_15_32_38_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_classif_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_15_32_38_0005',
    # This does NOT do the axes conversion at runtime, let's just see if this helps. Running 08/24.
    # Never mind, I didn't finish these, these results are AWFUL.
    #'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_classif_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_24_10_40_27_0001',
    #'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_classif_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_24_10_40_27_0002',
    #'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_classif_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_24_10_40_27_0003',
    #'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_classif_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_24_10_40_27_0004',
    #'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_classif_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_24_10_40_27_0005',
],
DENSE_TRANSFORMATION_MSE = [
    ## Wait, these must have done the un-necessary SVD corrections at runtime.
    # These had a max of 0.792, avg of 0.590.
    #'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_MSE_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_11_00_46_0001',
    #'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_MSE_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_11_00_46_0002',
    #'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_MSE_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_11_00_46_0003',
    #'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_MSE_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_11_00_46_0004',
    #'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_MSE_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_11_00_46_0005',
    # I think we want these (_slight_ improvement):
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_MSE_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_20_22_30_37_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_MSE_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_20_22_30_37_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_MSE_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_20_22_30_37_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_MSE_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_20_22_30_37_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_MSE_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_20_22_30_37_0005',
],
DENSE_TRANSFORMATION_PW = [
    # This did the axes conversion at runtime, which I think we want now? But this was worse than the alternative of not doing it?
    # Max of 0.288, avg of 0.122.
    #'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_15_39_28_0001',
    #'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_15_39_28_0002',
    #'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_15_39_28_0003',
    #'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_15_39_28_0004',
    #'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_17_15_39_28_0005',
    # I think we want these, which are actually giving slightly higher results (0.360, 0.158) though still bad:
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_21_10_08_43_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_21_10_08_43_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_21_10_08_43_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_21_10_08_43_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_21_10_08_43_0005',
],
RGB_DIRECT_VECTOR = [  # actually v1 but I want to keep the name the same, run locally so file names don't have same stem
    ## Wait, these must have done the un-necessary SVD corrections at runtime.
    ## 0.808, 0.628 for two evaluation metrics.
    #'BC04_MMOneSphere_v02_ntrain_0025_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_17_11_39_27_0001',
    #'BC04_MMOneSphere_v02_ntrain_0025_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_17_11_09_21_0001',
    #'BC04_MMOneSphere_v02_ntrain_0025_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_17_11_07_53_0001',
    #'BC04_MMOneSphere_v02_ntrain_0025_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_17_10_37_45_0001',
    #'BC04_MMOneSphere_v02_ntrain_0025_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_17_10_36_50_0001',
    # I think we want these.
    'BC04_MMOneSphere_v02_ntrain_0025_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_19_56_15_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_19_56_15_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_19_56_15_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_19_56_15_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_19_56_15_0005',
],
RGB_DIRECT_VECTOR_AUGM = [  # actually v1 but I want to keep the name the same, run locally so file names don't have same stem
    ## Wait, these must have done the un-necessary SVD corrections at runtime.
    ## 0.952, 0.698 for two evaluation metrics.
    #'BC04_MMOneSphere_v02_ntrain_0025_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_17_12_44_17_0001',
    #'BC04_MMOneSphere_v02_ntrain_0025_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_17_12_43_17_0001',
    #'BC04_MMOneSphere_v02_ntrain_0025_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_17_12_14_07_0001',
    #'BC04_MMOneSphere_v02_ntrain_0025_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_17_12_12_44_0001',
    #'BC04_MMOneSphere_v02_ntrain_0025_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_17_11_40_46_0001',
    # I think we want these.
    'BC04_MMOneSphere_v02_ntrain_0025_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_20_06_47_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_20_06_47_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_20_06_47_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_20_06_47_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_20_06_47_0005',
],
RGBD_DIRECT_VECTOR = [
    ## Wait, these must have done the un-necessary SVD corrections at runtime.
    # 0.872 +/- 0.03 and 0.737 +/- 0.03 for the two evaluation metrics.
    #'BC04_MMOneSphere_v02_ntrain_0025_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_18_13_24_47_0001',
    #'BC04_MMOneSphere_v02_ntrain_0025_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_18_13_24_47_0002',
    #'BC04_MMOneSphere_v02_ntrain_0025_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_18_13_24_47_0003',
    #'BC04_MMOneSphere_v02_ntrain_0025_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_18_13_24_47_0004',
    #'BC04_MMOneSphere_v02_ntrain_0025_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_18_13_24_47_0005',
    # I think we want these. Results are slightly better.
    'BC04_MMOneSphere_v02_ntrain_0025_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_20_26_00_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_20_26_00_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_20_26_00_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_20_26_00_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_20_26_00_0005',
],
RGBD_DIRECT_VECTOR_AUGM = [
    ## Wait, these must have done the un-necessary SVD corrections at runtime.
    # 0.888 +/- 0.06 and 0.699 +/- 0.03 for the two evaluation metrics.
    #'BC04_MMOneSphere_v02_ntrain_0025_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_18_13_32_23_0001',
    #'BC04_MMOneSphere_v02_ntrain_0025_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_18_13_32_23_0002',
    #'BC04_MMOneSphere_v02_ntrain_0025_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_18_13_32_23_0003',
    #'BC04_MMOneSphere_v02_ntrain_0025_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_18_13_32_23_0004',
    #'BC04_MMOneSphere_v02_ntrain_0025_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_18_13_32_23_0005',
    # I think we want these. Results are slightly better.
    'BC04_MMOneSphere_v02_ntrain_0025_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_22_33_08_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_22_33_08_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_22_33_08_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_22_33_08_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_22_33_08_0005',
],
D_SEGM_DIRECT_VECTOR=[
    # Segmented depth --> Image CNN --> Vector (no SVD at runtime)
    # Running these locally.
    'BC04_MMOneSphere_v02_ntrain_0025_depth_segm_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_24_19_51_28_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_depth_segm_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_24_19_51_41_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_depth_segm_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_24_21_38_25_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_depth_segm_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_24_21_39_32_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_depth_segm_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_24_23_30_44_0001',
],
RGB_SEGM_DIRECT_VECTOR=[
    # Ran 08/25 on cluster. Use `toolflownet` branch.
    'BC04_MMOneSphere_v02_ntrain_0025_rgb_segm_masks_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_16_52_40_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_rgb_segm_masks_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_16_52_40_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_rgb_segm_masks_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_16_52_40_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_rgb_segm_masks_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_16_52_40_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_rgb_segm_masks_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_16_52_40_0005',
],
RGBD_SEGM_DIRECT_VECTOR=[
    # Ran 08/25 on cluster. Use `toolflownet` branch.
    'BC04_MMOneSphere_v02_ntrain_0025_rgbd_segm_masks_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_16_50_51_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_rgbd_segm_masks_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_16_50_51_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_rgbd_segm_masks_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_16_50_51_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_rgbd_segm_masks_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_16_50_51_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_rgbd_segm_masks_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_16_50_51_0005',
],
D_DIRECT_VECTOR=[
    # Cluster, closer to rebuttal deadline.
    'BC04_MMOneSphere_v02_ntrain_0025_depth_img_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_23_36_57_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_depth_img_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_23_36_57_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_depth_img_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_23_36_57_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_depth_img_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_23_36_57_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_depth_img_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_23_36_57_0005',
],
STATE_GT_DIRECT_VECTOR=[
    ## This has state info. For 4D and 6D scooping we use (in order): 3d ball center, 3d ladle tip, 4d ladle quaternion.
    ## The quaternion is computed analytically instead of us storing the data (though we have regenerated such data for later).
    ## This one got: 0.496 $\pm$ 0.14
    #'BC04_MMOneSphere_v02_ntrain_0025_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_00_59_52_0001',
    #'BC04_MMOneSphere_v02_ntrain_0025_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_00_59_52_0002',
    #'BC04_MMOneSphere_v02_ntrain_0025_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_00_59_52_0003',
    #'BC04_MMOneSphere_v02_ntrain_0025_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_00_59_52_0004',
    #'BC04_MMOneSphere_v02_ntrain_0025_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_00_59_52_0005',
    # This has NEW BC data where I now actually save the ladle quaternion in the obs tuple, so we do not have to
    # recompute such data after the fact. I think we should use this. Edit: well it got worse performance
    # for some reason LOL. 0.336 +/- 0.06, no idea why. Also no idea why this is not as good as the learned
    # state predictor, and the MSE losses for this seem to be going down (for eval as well).
    'BC04_MMOneSphere_v02_ntrain_0025_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_12_48_41_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_12_48_41_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_12_48_41_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_12_48_41_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_12_48_41_0005',
],
STATE_LEARNED_DIRECT_VECTOR = [
    # Literally last-minute before rebuttal. Uses the ACTUAL ball + ladle tip + ladle quaternion, not the tool reducer one.
    # I have NO IDEA why this is better than using ground truth state.
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_state_predictor_then_mlp_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_27_16_11_17_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_state_predictor_then_mlp_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_27_16_11_17_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_state_predictor_then_mlp_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_27_16_11_17_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_state_predictor_then_mlp_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_27_16_11_17_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_state_predictor_then_mlp_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_27_16_11_17_0005',
],

# --------------------------------------- NEW ROT REPRESENTATIONS ------------------------------------- #
# Also includes repeats of our earlier axis-angle results just in case
PCL_DIRECT_VECTOR_MSE_NEWAPI_AXANG=[
    # Ran 5x on cluster 09/27. One used a bad GPU so had to regen.
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_eepose_convert_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_27_09_27_20_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_eepose_convert_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_27_09_27_20_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_eepose_convert_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_27_09_27_20_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_eepose_convert_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_27_09_27_20_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_eepose_convert_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_27_10_47_00_0001',
],
PCL_DIRECT_VECTOR_MSE_INTRIN_AXANG=[
    # 5x on cluster.
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_eepose_convert_intrinsic_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_09_44_07_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_eepose_convert_intrinsic_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_09_44_07_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_eepose_convert_intrinsic_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_09_44_07_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_eepose_convert_intrinsic_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_09_44_07_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_eepose_convert_intrinsic_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_09_44_07_0005',
],
PCL_DENSE_TRANSF_MSE_EXTRIN_AXANG=[
    # Ran 5x on cluster.
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_10_29_48_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_10_29_48_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_10_29_48_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_10_29_48_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_10_29_48_0005',
],
PCL_DENSE_TRANSF_MSE_INTRIN_AXANG=[
    # Ran 5x on cluster.
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_intrinsic_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_10_19_25_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_intrinsic_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_10_19_25_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_intrinsic_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_10_19_25_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_intrinsic_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_10_19_25_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_intrinsic_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_10_19_25_0005',
],
PCL_DIRECT_VECTOR_MSE_FRO_4D_ROTS=[
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_rpmg_eepose_convert_rotation_4D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_29_13_04_30_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_rpmg_eepose_convert_rotation_4D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_29_13_04_30_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_rpmg_eepose_convert_rotation_4D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_29_13_04_30_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_rpmg_eepose_convert_rotation_4D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_29_13_04_30_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_rpmg_eepose_convert_rotation_4D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_29_13_04_30_0005',
],
PCL_DIRECT_VECTOR_MSE_FRO_6D_ROTS=[
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_rpmg_eepose_convert_rotation_6D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_28_19_47_15_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_rpmg_eepose_convert_rotation_6D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_28_19_47_15_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_rpmg_eepose_convert_rotation_6D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_28_19_47_15_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_rpmg_eepose_convert_rotation_6D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_28_19_47_15_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_rpmg_eepose_convert_rotation_6D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_28_19_47_15_0005',
],
PCL_DIRECT_VECTOR_MSE_FRO_9D_ROTS=[
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_rpmg_eepose_convert_rotation_9D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_11_03_09_28_24_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_rpmg_eepose_convert_rotation_9D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_11_03_09_28_24_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_rpmg_eepose_convert_rotation_9D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_11_03_09_28_24_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_rpmg_eepose_convert_rotation_9D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_11_03_09_28_24_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_rpmg_eepose_convert_rotation_9D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_11_03_09_28_24_0005',
],
PCL_DIRECT_VECTOR_MSE_FRO_10D_ROTS=[
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_rpmg_eepose_convert_rotation_10D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_30_20_54_08_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_rpmg_eepose_convert_rotation_10D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_30_20_54_08_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_rpmg_eepose_convert_rotation_10D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_30_20_54_08_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_rpmg_eepose_convert_rotation_10D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_30_20_54_08_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_rpmg_eepose_convert_rotation_10D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_30_20_54_08_0005',
],

# --------------------------------------------- segless --------------------------------------------- #
SEGLESS_NAIVE=[
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_15_33_20_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_15_33_20_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_15_33_20_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_15_33_20_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_15_33_20_0005',
],
SEGLESS_WEIGHT_CONSIST=[
    # Use L1 for segmentation normalization.
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_15_17_22_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_15_17_22_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_15_17_22_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_15_17_22_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_15_17_22_0005',
],
SEGLESS_WEIGHT_CONSIST_SOFTMAX=[
    # Use softmax for segmentation normalization, temperature 0.1.
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_16_09_46_0001',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_16_09_46_0002',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_16_09_46_0003',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_16_09_46_0004',
    'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_16_09_46_0005',
],
)

# ============================================================================================== #
# =================================== 4DoF SCOOPING ============================================ #
# ============================================================================================== #

MM_ONE_SPHERE = dict(
# ---------------------------------- OUR METHOD ----------------------------------- #
FLOW3D_SVD_PW_CONSIST_0_0=[
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_17_18_23_01_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_17_18_23_01_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_17_18_23_01_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_17_18_23_01_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_17_18_23_01_0005',
],
FLOW3D_SVD_PW_CONSIST_0_1=[
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_11_00_57_07_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_11_00_57_07_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_11_00_57_07_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_11_00_57_07_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_11_00_57_07_0005',
],
FLOW3D_SVD_PW_CONSIST_0_5=[
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_17_18_38_44_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_17_18_38_44_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_17_18_38_44_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_17_18_38_44_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_17_18_38_44_0005',
],
FLOW3D_SVD_PW_CONSIST_1_0=[
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_18_11_04_58_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_18_11_04_58_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_18_11_04_58_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_18_11_04_58_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_18_11_04_58_0005',
],

# --------------------------------------- ABLATIONS ----------------------------------- #
FLOW3D_SVD_PW_CONSIST_NOSKIP=[
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_NOSKIP_scalePCL_noScaleTarg_2022_06_18_16_55_02_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_NOSKIP_scalePCL_noScaleTarg_2022_06_18_16_55_02_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_NOSKIP_scalePCL_noScaleTarg_2022_06_18_16_55_02_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_NOSKIP_scalePCL_noScaleTarg_2022_06_18_16_55_02_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_NOSKIP_scalePCL_noScaleTarg_2022_06_18_16_55_02_0005',
],
FLOW3D_PW_BEFORE_SVD_NOCONSISTENCY=[
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_PW_bef_SVD_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_20_11_21_36_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_PW_bef_SVD_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_20_11_21_36_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_PW_bef_SVD_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_20_11_21_36_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_PW_bef_SVD_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_20_11_21_36_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_PW_bef_SVD_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_20_11_21_36_0005',
],
FLOW3D_MSE_AFTER_SVD_CONSISTENCY=[
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_21_07_51_43_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_21_07_51_43_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_21_07_51_43_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_21_07_51_43_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_21_07_51_43_0005',
],

# ----------------------------- SCALING (or lack thereof) --------------------------------- #
FLOW3D_SVD_PW_CONSIST_0_1_NOSCALE=[
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_rawPCL_noScaleTarg_2022_06_18_13_45_31_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_rawPCL_noScaleTarg_2022_06_18_13_45_31_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_rawPCL_noScaleTarg_2022_06_18_13_45_31_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_rawPCL_noScaleTarg_2022_06_18_13_45_31_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_rawPCL_noScaleTarg_2022_06_18_13_45_31_0005',
],
DIRECT_VECTOR_MSE_NOSCALE=[
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_4DoF_ar_8_hor_100_rawPCL_noScaleTarg_2022_06_19_08_21_26_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_4DoF_ar_8_hor_100_rawPCL_noScaleTarg_2022_06_19_08_21_26_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_4DoF_ar_8_hor_100_rawPCL_noScaleTarg_2022_06_19_08_21_26_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_4DoF_ar_8_hor_100_rawPCL_noScaleTarg_2022_06_19_08_21_26_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_4DoF_ar_8_hor_100_rawPCL_noScaleTarg_2022_06_19_08_21_26_0005',
],
DIRECT_VECTOR_MSE_MULT_ROT_250=[ # small experiment where we mult rot by 250x and downscale by 250x at inference, instead of lambda_rot=100
    # Update: actually I think THIS should be the main result we now use. Because this seems more principled and got 0.620 max perf instead of 0.456.
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_09_21_17_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_09_21_17_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_09_21_17_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_09_21_17_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_09_21_17_0005',
],
RGB_DIRECT_VECTOR_NOSCALE=[
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_noScaleTarg_2022_06_19_11_17_05_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_noScaleTarg_2022_06_19_11_17_05_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_noScaleTarg_2022_06_19_11_17_05_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_noScaleTarg_2022_06_19_11_17_05_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_noScaleTarg_2022_06_19_11_17_05_0005',
],

# ---------------------------------------- NOISE/REDUCETOOL ----------------------------------------- #
FLOW3D_GAUSSNOISE_005=[
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_GaussNoise_0.005_scalePCL_noScaleTarg_2022_06_20_09_21_54_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_GaussNoise_0.005_scalePCL_noScaleTarg_2022_06_20_09_21_54_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_GaussNoise_0.005_scalePCL_noScaleTarg_2022_06_20_09_21_54_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_GaussNoise_0.005_scalePCL_noScaleTarg_2022_06_20_09_21_54_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_GaussNoise_0.005_scalePCL_noScaleTarg_2022_06_20_09_21_54_0005',
],
FLOW3D_GAUSSNOISE_010=[
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_GaussNoise_0.01_scalePCL_noScaleTarg_2022_06_20_11_56_45_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_GaussNoise_0.01_scalePCL_noScaleTarg_2022_06_20_11_56_45_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_GaussNoise_0.01_scalePCL_noScaleTarg_2022_06_20_11_56_45_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_GaussNoise_0.01_scalePCL_noScaleTarg_2022_06_20_11_56_45_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_GaussNoise_0.01_scalePCL_noScaleTarg_2022_06_20_11_56_45_0005',
],
FLOW3D_REDUCETOOL_010=[  # Eddie ran these
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_15_15_30_43_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_15_15_33_08_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_15_18_34_08_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_15_19_22_11_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_15_19_39_34_0001',
],

# --------------------------------------- BASELINES ------------------------------------- #
DIRECT_VECTOR_MSE=[
    ## This keeps the rotation at a small scale but adds a weight by 100. I think this is less principled. And got 0.456 +/- 0.10 vs the other way.
    #'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_11_01_41_34_0001',
    #'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_11_01_41_34_0002',
    #'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_11_01_41_34_0003',
    #'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_11_01_41_34_0004',
    #'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_11_01_41_34_0005',
    # This multiples rotation by 250 for scaling, and downscales by 250x at inference, and does NOT do lambda_rot=100 but uses 1.
    # Update: actually I think THIS should be the main result we now use. Because this seems more principled and got 0.620 +/- 0.07.,
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_09_21_17_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_09_21_17_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_09_21_17_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_09_21_17_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_17_09_21_17_0005',
],
DIRECT_VECTOR_PW=[
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_classif_6D_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_19_16_33_37_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_classif_6D_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_19_16_33_37_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_classif_6D_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_19_16_33_37_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_classif_6D_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_19_16_33_37_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_classif_6D_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_19_16_33_37_0005',
],
DENSE_TRANSFORMATION_MSE=[  # first 2 runs were separate earlier
    ## This one is giving 0.278 $\pm$ 0.09.
    #'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_12_02_18_11_0001',
    #'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_12_04_13_20_0001',
    #'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_19_21_44_31_0001',
    #'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_19_21_44_31_0002',
    #'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_19_21_44_31_0003',
    # I ran this later and instead of lambda_rot=100, I set it to 1 and scale rotation. This helps this baseline to get 0.582 $\pm$ 0.08.
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_25_16_32_41_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_25_16_32_41_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_25_16_32_41_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_25_16_32_41_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_25_16_32_41_0005',
],
DENSE_TRANSFORMATION_PW=[
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_19_16_41_34_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_19_16_41_34_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_19_16_41_34_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_19_16_41_34_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_19_16_41_34_0005',
],
RGB_DIRECT_VECTOR_v1=[
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_06_18_19_02_24_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_06_18_19_02_24_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_06_18_19_02_24_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_06_18_19_02_24_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_06_18_19_02_24_0005',
],
RGB_DIRECT_VECTOR_v1_AUGM=[
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_06_18_17_36_02_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_06_18_17_36_02_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_06_18_17_36_02_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_06_18_17_36_02_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_06_18_17_36_02_0005',
],
RGB_DIRECT_VECTOR=[  # ran on marvin, similar as v1 but with 1/2 as many filters, 2x more encoder dim.
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_06_20_16_54_43_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_06_20_18_05_31_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_06_20_18_05_52_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_06_20_18_57_02_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_06_20_18_57_17_0001',
],
RGB_DIRECT_VECTOR_AUGM=[  # ran on marvin, similar as v1 but with 1/2 as many filters, 2x more encoder dim.
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_06_20_15_42_49_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_06_20_15_43_08_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_06_20_16_17_00_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_06_20_16_17_08_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgb_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_06_20_16_53_50_0001',
],
RGBD_DIRECT_VECTOR = [ # (ran Aug 18) oh this might be v1 but whatever, performance was almost the same
    ## Wait, these must have done the un-necessary SVD corrections at runtime.
    ## 0.443 +/- 0.06 and 0.272 +/- 0.02 for the two statistics.
    #'BC04_MMOneSphere_v01_ntrain_0100_cam_rgbd_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_18_14_37_49_0001',
    #'BC04_MMOneSphere_v01_ntrain_0100_cam_rgbd_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_18_14_37_49_0002',
    #'BC04_MMOneSphere_v01_ntrain_0100_cam_rgbd_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_18_14_37_49_0003',
    #'BC04_MMOneSphere_v01_ntrain_0100_cam_rgbd_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_18_14_37_49_0004',
    #'BC04_MMOneSphere_v01_ntrain_0100_cam_rgbd_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_18_14_37_49_0005',
    ## I think we want these.
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgbd_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_21_10_54_21_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgbd_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_21_10_54_21_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgbd_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_21_10_54_21_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgbd_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_21_10_54_21_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgbd_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_21_10_54_21_0005',
],
RGBD_DIRECT_VECTOR_AUGM = [ # (ran Aug 18) oh this might be v1 but whatever, performance was almost the same
    ## Wait, these must have done the un-necessary SVD corrections at runtime.
    ## 1.038 +/- 0.09 and 0.613 +/- 0.08 for the two statistics.
    #'BC04_MMOneSphere_v01_ntrain_0100_cam_rgbd_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_18_14_39_53_0001',
    #'BC04_MMOneSphere_v01_ntrain_0100_cam_rgbd_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_18_14_39_53_0002',
    #'BC04_MMOneSphere_v01_ntrain_0100_cam_rgbd_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_18_14_39_53_0003',
    #'BC04_MMOneSphere_v01_ntrain_0100_cam_rgbd_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_18_14_39_53_0004',
    #'BC04_MMOneSphere_v01_ntrain_0100_cam_rgbd_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_18_14_39_53_0005',
    ## I think we want these. Interestingly for augmentation it's actually worse (?) compared to the one with the
    # un-necessary SVD correction?
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgbd_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_21_10_57_24_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgbd_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_21_10_57_24_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgbd_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_21_10_57_24_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgbd_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_21_10_57_24_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_cam_rgbd_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_21_10_57_24_0005',
],
D_SEGM_DIRECT_VECTOR=[
    ## Segmented depth --> Image CNN --> Vector
    # Running these on cluster 5x 08/24 evening. Does not use lambda_rot=100 but just 1, and scales rotation.
    'BC04_MMOneSphere_v01_ntrain_0100_depth_segm_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_24_21_20_12_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_depth_segm_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_24_21_20_12_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_depth_segm_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_24_21_20_12_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_depth_segm_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_24_21_20_12_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_depth_segm_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_24_21_20_12_0005',
],
RGB_SEGM_DIRECT_VECTOR=[
    # Ran 08/25 on cluster. Use `toolflownet` branch. Scales rotation, lambda_rot=1.
    'BC04_MMOneSphere_v01_ntrain_0100_rgb_segm_masks_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_25_16_41_22_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_rgb_segm_masks_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_25_16_41_22_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_rgb_segm_masks_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_25_16_41_22_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_rgb_segm_masks_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_25_16_41_22_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_rgb_segm_masks_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_25_16_41_22_0005',
],
RGBD_SEGM_DIRECT_VECTOR=[
    # Ran 08/25 on cluster. Use `toolflownet` branch. Scales rotation, lambda_rot=1.
    'BC04_MMOneSphere_v01_ntrain_0100_rgbd_segm_masks_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_25_16_44_51_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_rgbd_segm_masks_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_25_16_44_51_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_rgbd_segm_masks_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_25_16_44_51_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_rgbd_segm_masks_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_25_16_44_51_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_rgbd_segm_masks_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_25_16_44_51_0005',
],
D_DIRECT_VECTOR=[
    # Ran on cluster, 08/26, I ran 08/25 but maybe I pushed with MM v01?? I re-did it 08/26:
    'BC04_MMOneSphere_v01_ntrain_0100_depth_img_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_26_01_44_28_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_depth_img_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_26_01_44_28_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_depth_img_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_26_01_44_28_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_depth_img_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_26_01_44_28_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_depth_img_pixel_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_26_01_44_28_0005',
],
STATE_GT_DIRECT_VECTOR=[
    ## This has state info. For 4D and 6D scooping we use (in order): 3d ball center, 3d ladle tip, 4d ladle quaternion.
    ## The quaternion is computed analytically instead of us storing the data (though we have regenerated such data for later).
    ## This one gets 0.456 +/- 0.05 normalized performance.
    #'BC04_MMOneSphere_v01_ntrain_0100_state_mlp_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_27_01_09_59_0001',
    #'BC04_MMOneSphere_v01_ntrain_0100_state_mlp_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_27_01_09_59_0002',
    #'BC04_MMOneSphere_v01_ntrain_0100_state_mlp_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_27_01_09_59_0003',
    #'BC04_MMOneSphere_v01_ntrain_0100_state_mlp_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_27_01_09_59_0004',
    #'BC04_MMOneSphere_v01_ntrain_0100_state_mlp_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_27_01_09_59_0005',
    ## This has NEW BC data where I now actually save the ladle quaternion in the obs tuple, so we do not have to
    ## recompute such data after the fact. I think we should use this. Eeek this one got 1.152 +/- 0.04, virtually tied w/TFN!
    'BC04_MMOneSphere_v01_ntrain_0100_state_mlp_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_27_12_55_55_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_state_mlp_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_27_12_55_55_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_state_mlp_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_27_12_55_55_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_state_mlp_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_27_12_55_55_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_state_mlp_eepose_4DoF_ar_8_hor_100_scaleTarg_2022_08_27_12_55_55_0005',
],
STATE_LEARNED_DIRECT_VECTOR=[
    # Uses the data with the actual ladle instead of the tool reducer. This _should_ be lower than the GT version.
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_27_16_02_01_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_27_16_02_01_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_27_16_02_01_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_27_16_02_01_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_27_16_02_01_0005',
],

# --------------------------------------- NUM_DEMOS ------------------------------------- #
FLOW3D_SVD_PW_CONSIST_010_DEMOS=[
    'BC04_MMOneSphere_v01_ntrain_0010_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_23_01_08_52_0001',
    'BC04_MMOneSphere_v01_ntrain_0010_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_23_01_08_52_0002',
    'BC04_MMOneSphere_v01_ntrain_0010_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_23_01_08_52_0003',
    'BC04_MMOneSphere_v01_ntrain_0010_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_23_01_08_52_0004',
    'BC04_MMOneSphere_v01_ntrain_0010_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_23_01_08_52_0005',
],
FLOW3D_SVD_PW_CONSIST_050_DEMOS=[
    'BC04_MMOneSphere_v01_ntrain_0050_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_22_15_02_07_0001',
    'BC04_MMOneSphere_v01_ntrain_0050_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_22_15_02_07_0002',
    'BC04_MMOneSphere_v01_ntrain_0050_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_22_15_02_07_0003',
    'BC04_MMOneSphere_v01_ntrain_0050_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_22_15_02_07_0004',
    'BC04_MMOneSphere_v01_ntrain_0050_PCL_PNet2_svd_pointwise_ee2flow_4DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_22_15_02_07_0005',
],

# ======================================== new rot representations ========================================== #
# Also includes repeats of our earlier axis-angle results just in case
PCL_DIRECT_VECTOR_MSE_NEWAPI_AXANG=[
    # Cluster 5X, meant to reproduce earlier results.
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_convert_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_28_10_47_14_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_convert_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_28_10_47_14_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_convert_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_28_10_47_14_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_convert_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_28_10_47_14_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_convert_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_28_10_47_14_0005',
],
PCL_DIRECT_VECTOR_MSE_INTRIN_AXANG=[
    # 5x on cluster.
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_convert_intrinsic_axis_angle_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_25_15_17_00_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_convert_intrinsic_axis_angle_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_25_15_17_00_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_convert_intrinsic_axis_angle_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_25_15_17_00_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_convert_intrinsic_axis_angle_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_25_15_17_00_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_eepose_convert_intrinsic_axis_angle_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_25_15_17_00_0005',
],
PCL_DENSE_TRANSF_MSE_EXTRIN_AXANG=[
    # Ran 5x on cluster.
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_axis_angle_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_15_20_37_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_axis_angle_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_15_20_37_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_axis_angle_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_15_20_37_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_axis_angle_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_15_20_37_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_axis_angle_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_15_20_37_0005',
],
PCL_DENSE_TRANSF_MSE_INTRIN_AXANG=[
    # Ran 5x on cluster.
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_intrinsic_axis_angle_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_15_28_15_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_intrinsic_axis_angle_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_15_28_15_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_intrinsic_axis_angle_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_15_28_15_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_intrinsic_axis_angle_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_15_28_15_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_intrinsic_axis_angle_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_15_28_15_0005',
],
PCL_DIRECT_VECTOR_MSE_FRO_4D_ROTS=[
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_4D_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_29_13_12_24_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_4D_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_29_13_12_24_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_4D_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_29_13_12_24_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_4D_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_29_13_12_24_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_4D_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_29_13_12_24_0005',
],
PCL_DIRECT_VECTOR_MSE_FRO_6D_ROTS=[
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_6D_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_29_09_47_22_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_6D_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_29_09_47_22_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_6D_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_29_09_47_22_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_6D_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_29_09_47_22_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_6D_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_29_09_47_22_0005',
],
PCL_DIRECT_VECTOR_MSE_FRO_9D_ROTS=[
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_9D_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_11_02_10_47_11_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_9D_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_11_02_10_47_11_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_9D_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_11_02_10_47_11_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_9D_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_11_02_10_47_11_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_9D_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_11_02_10_47_11_0005',
],
PCL_DIRECT_VECTOR_MSE_FRO_10D_ROTS=[
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_10D_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_02_12_46_45_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_10D_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_02_12_46_45_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_10D_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_02_12_46_45_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_10D_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_02_12_46_45_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_10D_4DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_02_12_46_45_0005',
],

# --------------------------------------------- segless --------------------------------------------- #
SEGLESS_NAIVE=[
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_4DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_12_08_48_55_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_4DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_12_08_48_55_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_4DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_12_08_48_55_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_4DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_12_08_48_55_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_4DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_12_08_48_55_0005',
],
SEGLESS_WEIGHT_CONSIST=[
    # Use L1 for segmentation normalization.
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_4DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_12_09_32_54_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_4DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_12_09_32_54_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_4DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_12_09_32_54_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_4DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_12_09_32_54_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_4DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_12_09_32_54_0005',
],
SEGLESS_WEIGHT_CONSIST_SOFTMAX=[
    # Use softmax for segmentation normalization, temperature 0.1.
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_4DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_12_14_07_57_0001',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_4DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_12_14_07_57_0002',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_4DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_12_14_07_57_0003',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_4DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_12_14_07_57_0004',
    'BC04_MMOneSphere_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_4DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_12_14_07_57_0005',
],
)

# ====================================================================================== #
# =================================== 6DoF POURING ===================================== #
# ====================================================================================== #

POUR_WATER_6DOF = dict(
# ================================================= our method ============================================ #
FLOW3D_SVD_PW_CONSIST_0_0=[  # Ran this 08/22 on cluster, has axes conversion
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_22_16_31_38_0001',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_22_16_31_38_0002',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_22_16_31_38_0003',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_22_16_31_38_0004',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_22_16_31_38_0005',
],
FLOW3D_SVD_PW_CONSIST_0_1=[  # Ran this 08/21 on cluster
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_21_23_21_52_0001',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_21_23_21_52_0002',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_21_23_21_52_0003',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_21_23_21_52_0004',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_21_23_21_52_0005',
],
FLOW3D_SVD_PW_CONSIST_0_5=[  # Ran this 08/21 on cluster
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_21_22_58_55_0001',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_21_22_58_55_0002',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_21_22_58_55_0003',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_21_22_58_55_0004',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_21_22_58_55_0005',
],
FLOW3D_SVD_PW_CONSIST_1_0=[  # Ran this 08/24 on cluster
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_24_09_54_04_0001',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_24_09_54_04_0002',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_24_09_54_04_0003',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_24_09_54_04_0004',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_24_09_54_04_0005',
],

# ================================================= ablations ============================================ #

FLOW3D_SVD_PW_CONSIST_NOSKIP=[
    # ran locally, this _should_ get 0.000 due to no rotations.
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_NOSKIP_scalePCL_noScaleTarg_2022_08_22_20_45_47_0001',
],
FLOW3D_PW_BEFORE_SVD_NOCONSISTENCY=[
    # Ran 08/22 on cluster, does axes conversion.
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_PW_bef_SVD_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_22_16_17_05_0001',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_PW_bef_SVD_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_22_16_17_05_0002',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_PW_bef_SVD_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_22_16_17_05_0003',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_PW_bef_SVD_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_22_16_17_05_0004',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_PW_bef_SVD_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_22_16_17_05_0005',
],
FLOW3D_MSE_AFTER_SVD_CONSISTENCY=[
    # Ran 08/22 on cluster. WAIT this one did not do axes conversion but also we need Eddie's newest fix.
    # Though amazingly this one actually looks good, unlike prior runs that used this ablation??
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_22_16_44_54_0001',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_22_16_44_54_0002',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_22_16_44_54_0003',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_22_16_44_54_0004',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_22_16_44_54_0005',
],

# ================================================= baselines ============================================ #
# NOTE: these use -0.004, -0.004, -0.004, -0.015, -0.015, -0.015] for scaling bounds to improve training.
# This means for example the translation values are multiplied by 250 for getting values closer to (-1,1) range for prediction.
DIRECT_VECTOR_MSE=[  # Ran this 08/21 on cluster
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_21_23_38_01_0001',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_21_23_38_01_0002',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_21_23_38_01_0003',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_21_23_38_01_0004',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_21_23_38_01_0005',
],
DIRECT_VECTOR_PW=[
    # Ran this on the cluster 08/24, WITH the axes conversion at test time (since the PW or PM loss is over the global axes-angle).
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_classif_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_24_10_12_00_0001',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_classif_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_24_10_12_00_0002',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_classif_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_24_10_12_00_0003',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_classif_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_24_10_12_00_0004',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_classif_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_24_10_12_00_0005',
],
DENSE_TRANSFORMATION_MSE=[ # Ran this 08/22 on cluster. 1 run failed so I re-ran it.
    # This does NOT do the axes conversion at test time.
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_22_11_02_29_0001',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_22_11_02_29_0002',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_22_11_02_29_0003',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_22_11_02_29_0004',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_22_12_02_33_0001',
],
DENSE_TRANSFORMATION_PW=[
    # Ran this on the cluster 08/24, WITH the axes conversion at test time (since the PW or PM loss is over the global axes-angle).
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_24_10_06_05_0001',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_24_10_06_05_0002',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_24_10_06_05_0003',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_24_10_06_05_0004',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_08_24_10_06_05_0005',
],
RGB_DIRECT_VECTOR=[  # Run before Aug 21 change in point cloud but this uses RGB and should not matter.
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_15_03_15_0001',
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_15_03_15_0002',
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_15_03_15_0003',
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_15_03_15_0004',
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_15_03_15_0005',
],
RGB_DIRECT_VECTOR_AUGM=[  # Run before Aug 21 change in point cloud but this uses RGB and should not matter.
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_15_19_45_0001',
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_15_19_45_0002',
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_15_19_45_0003',
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_15_19_45_0004',
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgb_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_15_19_45_0005',
],
RGBD_DIRECT_VECTOR=[  # Run before Aug 21 change in point cloud, which means depth has info about water.
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_14_10_13_0001',
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_14_10_13_0002',
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_14_10_13_0003',
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_14_10_13_0004',
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_14_10_13_0005',
],
RGBD_DIRECT_VECTOR_AUGM=[  # Run before Aug 21 change in point cloud, which means depth has info about water.
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_14_04_10_0001',
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_14_04_10_0002',
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_14_04_10_0003',
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_14_04_10_0004',
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgbd_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_20_14_04_10_0005',
],
D_SEGM_DIRECT_VECTOR=[
    ## Segmented depth --> Image CNN --> Vector (no SVD at runtime)
    # Ran on cluster, 08/25. I might wnat to double check JUST IN CASE ... I thought the results would be better here.
    'BC04_PourWater6D_v01_ntrain_0100_depth_segm_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_02_57_40_0001',
    'BC04_PourWater6D_v01_ntrain_0100_depth_segm_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_02_57_40_0002',
    'BC04_PourWater6D_v01_ntrain_0100_depth_segm_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_02_57_40_0003',
    'BC04_PourWater6D_v01_ntrain_0100_depth_segm_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_02_57_40_0004',
    'BC04_PourWater6D_v01_ntrain_0100_depth_segm_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_02_57_40_0005',
],
RGB_SEGM_DIRECT_VECTOR=[
    # Ran 08/25 on cluster. Must be on SoftGym `toolflownet-pouring-depth-segm` branch.
    'BC04_PourWater6D_v01_ntrain_0100_rgb_segm_masks_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_14_40_26_0001',
    'BC04_PourWater6D_v01_ntrain_0100_rgb_segm_masks_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_14_40_26_0002',
    'BC04_PourWater6D_v01_ntrain_0100_rgb_segm_masks_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_14_40_26_0003',
    'BC04_PourWater6D_v01_ntrain_0100_rgb_segm_masks_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_14_40_26_0004',
    'BC04_PourWater6D_v01_ntrain_0100_rgb_segm_masks_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_14_40_26_0005',
],
RGBD_SEGM_DIRECT_VECTOR=[
    # Ran 08/25 on cluster. Must be on SoftGym `toolflownet-pouring-depth-segm` branch.
    'BC04_PourWater6D_v01_ntrain_0100_rgbd_segm_masks_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_14_32_12_0001',
    'BC04_PourWater6D_v01_ntrain_0100_rgbd_segm_masks_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_14_32_12_0002',
    'BC04_PourWater6D_v01_ntrain_0100_rgbd_segm_masks_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_14_32_12_0003',
    'BC04_PourWater6D_v01_ntrain_0100_rgbd_segm_masks_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_14_32_12_0004',
    'BC04_PourWater6D_v01_ntrain_0100_rgbd_segm_masks_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_14_32_12_0005',
],
D_DIRECT_VECTOR=[
    # We might want to do this again but with the water in the depth, this doesn't have water in it:
    #'BC04_PourWater6D_v01_ntrain_0100_depth_img_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_23_52_55_0001',
    #'BC04_PourWater6D_v01_ntrain_0100_depth_img_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_23_52_55_0002',
    #'BC04_PourWater6D_v01_ntrain_0100_depth_img_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_23_52_55_0003',
    #'BC04_PourWater6D_v01_ntrain_0100_depth_img_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_23_52_55_0004',
    #'BC04_PourWater6D_v01_ntrain_0100_depth_img_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_25_23_52_55_0005',
    # Actually this has water in depth (not on a branch, I just used Yufei's earlier data, and made SoftGym output the same thing)
    # So I think _this_ is a much fairer comparison. And it's very bad. :)
    'BC04_PourWater6D_v01_ntrain_0100_depth_img_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_26_01_32_12_0001',
    'BC04_PourWater6D_v01_ntrain_0100_depth_img_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_26_01_32_12_0002',
    'BC04_PourWater6D_v01_ntrain_0100_depth_img_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_26_01_32_12_0003',
    'BC04_PourWater6D_v01_ntrain_0100_depth_img_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_26_01_32_12_0004',
    'BC04_PourWater6D_v01_ntrain_0100_depth_img_pixel_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_26_01_32_12_0005',
],
STATE_GT_DIRECT_VECTOR=[
    ## This has state info. For this, I used 72-D state, 35 for tool, 35 for target, 2 for water in either cup.
    ## 35 for each cup is due to five 3D positions (15) and five 4D quaternions (20).
    ## We regenerated the data for this experiment. FYI, performance of this is good.
    #'BC04_PourWater6D_v01_ntrain_0100_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_00_19_36_0001',
    #'BC04_PourWater6D_v01_ntrain_0100_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_00_19_36_0002',
    #'BC04_PourWater6D_v01_ntrain_0100_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_00_19_36_0003',
    #'BC04_PourWater6D_v01_ntrain_0100_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_00_19_36_0004',
    #'BC04_PourWater6D_v01_ntrain_0100_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_00_19_36_0005',
    ## 32-D state (no quaternions for the boxes). 15D for each of the two boxes, then 2 for water frac in both.
    #'BC04_PourWater6D_v01_ntrain_0100_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_02_12_56_0001',
    #'BC04_PourWater6D_v01_ntrain_0100_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_02_12_56_0002',
    #'BC04_PourWater6D_v01_ntrain_0100_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_02_12_56_0003',
    #'BC04_PourWater6D_v01_ntrain_0100_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_02_12_56_0004',
    #'BC04_PourWater6D_v01_ntrain_0100_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_02_12_56_0005',
    # Actually this seems better, 22D (same comments as 3D pouring).
    'BC04_PourWater6D_v01_ntrain_0100_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_22_59_44_0001',
    'BC04_PourWater6D_v01_ntrain_0100_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_22_59_44_0002',
    'BC04_PourWater6D_v01_ntrain_0100_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_22_59_44_0003',
    'BC04_PourWater6D_v01_ntrain_0100_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_22_59_44_0004',
    'BC04_PourWater6D_v01_ntrain_0100_state_mlp_eepose_6DoF_ar_8_hor_100_scaleTarg_2022_08_27_22_59_44_0005',
],
STATE_LEARNED_DIRECT_VECTOR=[
    ## TODO this took a while so I ran 2x local runs just to be safe, the first two here. There are 4 others.
    # Fortunately a lot worse than the ground truth version.
    #'BC04_PourWater6D_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_27_16_27_39_0001',
    #'BC04_PourWater6D_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_27_16_28_30_0001',
    #'BC04_PourWater6D_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_27_16_25_12_0001'
    #'BC04_PourWater6D_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_27_16_25_12_0002'
    #'BC04_PourWater6D_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_27_16_25_12_0003'
    #'BC04_PourWater6D_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_27_16_25_12_0004'
    # I think we want to use this, as this has the 22-D state dimension instead of 72-D (the latter doesn't make sense).
    'BC04_PourWater6D_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_29_20_17_26_0001',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_29_20_17_26_0002',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_29_20_17_26_0003',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_29_20_17_26_0004',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_29_20_17_26_0005',
],

# ======================================== new rot representations ========================================== #
# Also includes repeats of our earlier axis-angle results just in case
PCL_DIRECT_VECTOR_MSE_NEWAPI_AXANG=[
    # Ran on cluster 09/26, this is meant to reproduce earlier results and to make sure results are consistent.
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_eepose_convert_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_26_14_20_45_0001',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_eepose_convert_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_26_14_20_45_0002',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_eepose_convert_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_26_14_20_45_0003',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_eepose_convert_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_26_14_20_45_0004',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_eepose_convert_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_26_14_20_45_0005',
],
PCL_DIRECT_VECTOR_MSE_INTRIN_AXANG=[
    # 5x on cluster.
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_eepose_convert_intrinsic_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_25_19_57_07_0001',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_eepose_convert_intrinsic_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_25_19_57_07_0002',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_eepose_convert_intrinsic_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_25_19_57_07_0003',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_eepose_convert_intrinsic_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_25_19_57_07_0004',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_eepose_convert_intrinsic_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_25_19_57_07_0005',
],
PCL_DENSE_TRANSF_MSE_EXTRIN_AXANG=[
    # Ran 5x on cluster.
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_27_09_26_50_0001',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_27_09_26_50_0002',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_27_09_26_50_0003',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_27_09_26_50_0004',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_27_09_26_50_0005',
],
PCL_DENSE_TRANSF_MSE_INTRIN_AXANG=[
    # Ran 5x on cluster.
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_intrinsic_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_27_09_32_55_0001',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_intrinsic_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_27_09_32_55_0002',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_intrinsic_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_27_09_32_55_0003',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_intrinsic_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_27_09_32_55_0004',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_intrinsic_axis_angle_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_27_09_32_55_0005',
],
RGB_DIRECT_VECTOR_MSE_NEWAPI_AXANG=[
    # Ran locally on takeshi. Had to do some quick hacks to get this working.
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgb_pixel_eepose_convert_6DoF_ar_8_hor_100_scaleTarg_2022_09_26_15_34_14_0001',
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgb_pixel_eepose_convert_6DoF_ar_8_hor_100_scaleTarg_2022_09_26_15_34_47_0001',
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgb_pixel_eepose_convert_6DoF_ar_8_hor_100_scaleTarg_2022_09_26_16_48_11_0001',
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgb_pixel_eepose_convert_6DoF_ar_8_hor_100_scaleTarg_2022_09_26_16_48_31_0001',
    'BC04_PourWater6D_v01_ntrain_0100_cam_rgb_pixel_eepose_convert_6DoF_ar_8_hor_100_scaleTarg_2022_09_27_09_19_22_0001',
],
PCL_DIRECT_VECTOR_MSE_FRO_4D_ROTS=[
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_4D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_29_16_55_03_0001',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_4D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_29_16_55_03_0002',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_4D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_29_16_55_03_0003',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_4D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_29_16_55_03_0004',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_4D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_29_16_55_03_0005',
],
PCL_DIRECT_VECTOR_MSE_FRO_6D_ROTS=[
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_6D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_28_19_55_20_0001',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_6D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_28_19_55_20_0002',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_6D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_28_19_55_20_0003',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_6D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_28_19_55_20_0004',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_6D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_28_19_55_20_0005',
],
PCL_DIRECT_VECTOR_MSE_FRO_9D_ROTS=[
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_9D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_11_02_10_54_07_0001',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_9D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_11_02_10_54_07_0002',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_9D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_11_02_10_54_07_0003',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_9D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_11_02_10_54_07_0004',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_9D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_11_02_10_54_07_0005',
],
PCL_DIRECT_VECTOR_MSE_FRO_10D_ROTS=[
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_10D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_30_10_38_19_0001',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_10D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_30_10_38_19_0002',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_10D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_30_10_38_19_0003',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_10D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_30_10_38_19_0004',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_10D_6DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_30_10_38_19_0005',
],

# --------------------------------------------- segless --------------------------------------------- #
SEGLESS_NAIVE=[
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_21_23_28_0001',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_21_23_28_0002',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_21_23_28_0003',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_21_23_28_0004',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_21_23_28_0005',
],
SEGLESS_WEIGHT_CONSIST=[
    # Use L1 for segmentation normalization.
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_21_12_16_0001',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_21_12_16_0002',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_21_12_16_0003',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_21_12_16_0004',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_10_21_12_16_0005',
],
SEGLESS_WEIGHT_CONSIST_SOFTMAX=[
    # Use softmax for segmentation normalization, temperature 0.1.
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_11_08_56_59_0001',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_11_08_56_59_0002',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_11_08_56_59_0003',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_11_08_56_59_0004',
    'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_6DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_11_08_56_59_0005',
],
)

# ============================================================================================== #
# ==================================== 3DoF POURING ============================================ #
# ============================================================================================== #

POUR_WATER = dict(
# ================================================= our method ============================================ #
FLOW3D_SVD_PW_CONSIST_0_0=[
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_17_18_10_21_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_17_18_10_21_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_17_18_10_21_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_17_18_10_21_0004',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_17_18_10_21_0005',
],
FLOW3D_SVD_PW_CONSIST_0_1=[  # note: this I ran locally so all 'seed 0001'.
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_10_10_15_34_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_11_00_47_23_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_11_00_48_01_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_11_01_20_38_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_11_09_54_26_0001',
],
FLOW3D_SVD_PW_CONSIST_0_5=[
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_17_18_45_50_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_17_18_45_50_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_17_18_45_50_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_17_18_45_50_0004',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_17_18_45_50_0005',
],
FLOW3D_SVD_PW_CONSIST_1_0=[
   'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_18_11_02_22_0001',
   'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_18_11_02_22_0002',
   'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_18_11_02_22_0003',
   'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_18_11_02_22_0004',
   'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_18_11_02_22_0005',
],

# ================================================= ablations ============================================ #

FLOW3D_SVD_PW_CONSIST_NOSKIP=[ # ran 1 local run just so we get the number in the table, should give 0% performance
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_NOSKIP_scalePCL_noScaleTarg_2022_08_24_10_19_06_0001',
],
FLOW3D_PW_BEFORE_SVD_NOCONSISTENCY=[
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_PW_bef_SVD_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_20_11_21_02_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_PW_bef_SVD_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_20_11_21_02_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_PW_bef_SVD_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_20_11_21_02_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_PW_bef_SVD_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_20_11_21_02_0004',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_PW_bef_SVD_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_20_11_21_02_0005',
],
FLOW3D_MSE_AFTER_SVD_CONSISTENCY=[
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_21_07_54_22_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_21_07_54_22_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_21_07_54_22_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_21_07_54_22_0004',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_21_07_54_22_0005',
],

#================================================== scaling (lack thereof) ============================================ #
FLOW3D_SVD_PW_CONSIST_0_1_NOSCALE=[
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_rawPCL_noScaleTarg_2022_06_18_13_51_16_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_rawPCL_noScaleTarg_2022_06_18_13_51_16_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_rawPCL_noScaleTarg_2022_06_18_13_51_16_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_rawPCL_noScaleTarg_2022_06_18_13_51_16_0004',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_rawPCL_noScaleTarg_2022_06_18_13_51_16_0005',
],
DIRECT_VECTOR_MSE_NOSCALE=[
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_eepose_3DoF_ar_8_hor_100_rawPCL_noScaleTarg_2022_06_19_08_16_28_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_eepose_3DoF_ar_8_hor_100_rawPCL_noScaleTarg_2022_06_19_08_16_28_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_eepose_3DoF_ar_8_hor_100_rawPCL_noScaleTarg_2022_06_19_08_16_28_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_eepose_3DoF_ar_8_hor_100_rawPCL_noScaleTarg_2022_06_19_08_16_28_0004',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_eepose_3DoF_ar_8_hor_100_rawPCL_noScaleTarg_2022_06_19_08_16_28_0005',
],
RGB_DIRECT_VECTOR_NOSCALE=[
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_noScaleTarg_2022_06_19_11_14_10_0001',
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_noScaleTarg_2022_06_19_11_14_10_0002',
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_noScaleTarg_2022_06_19_11_14_10_0003',
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_noScaleTarg_2022_06_19_11_14_10_0004',
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_noScaleTarg_2022_06_19_11_14_10_0005',
],

# ================================================= noise/reducetool ============================================ #
FLOW3D_GAUSSNOISE_005=[
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_GaussNoise_0.005_scalePCL_noScaleTarg_2022_06_20_09_43_27_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_GaussNoise_0.005_scalePCL_noScaleTarg_2022_06_20_09_43_27_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_GaussNoise_0.005_scalePCL_noScaleTarg_2022_06_20_09_43_27_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_GaussNoise_0.005_scalePCL_noScaleTarg_2022_06_20_09_43_27_0004',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_GaussNoise_0.005_scalePCL_noScaleTarg_2022_06_20_09_43_27_0005',
],
FLOW3D_GAUSSNOISE_010=[
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_GaussNoise_0.01_scalePCL_noScaleTarg_2022_06_20_13_45_44_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_GaussNoise_0.01_scalePCL_noScaleTarg_2022_06_20_13_45_44_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_GaussNoise_0.01_scalePCL_noScaleTarg_2022_06_20_13_45_44_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_GaussNoise_0.01_scalePCL_noScaleTarg_2022_06_20_13_45_44_0004',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_GaussNoise_0.01_scalePCL_noScaleTarg_2022_06_20_13_45_44_0005',
],
FLOW3D_REDUCETOOL_010=[  # some of these crashed due to SVD errors
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_reducetool_scalePCL_noScaleTarg_2022_06_21_09_11_50_0001',
    #'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_reducetool_scalePCL_noScaleTarg_2022_06_21_09_11_50_0002', # crashed, SVD
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_reducetool_scalePCL_noScaleTarg_2022_06_21_09_11_50_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_reducetool_scalePCL_noScaleTarg_2022_06_21_09_11_50_0004',
    #'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_reducetool_scalePCL_noScaleTarg_2022_06_21_09_11_50_0005', # crashed, SVD
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_reducetool_scalePCL_noScaleTarg_2022_06_21_22_09_26_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_reducetool_scalePCL_noScaleTarg_2022_06_21_22_09_26_0002',
    #'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_reducetool_scalePCL_noScaleTarg_2022_06_21_22_09_26_0003', # might be good but we just need 5
],

# ================================================= baselines ============================================ #
# note that all the MSE ones have 250X scaling here instead of 100X for translations, as we had them earlier for the 8+n page submission.
DIRECT_VECTOR_MSE=[
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_19_08_10_41_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_19_08_10_41_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_19_08_10_41_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_19_08_10_41_0004',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_19_08_10_41_0005',
],
DIRECT_VECTOR_PW=[
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_classif_6D_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_19_18_15_18_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_classif_6D_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_19_18_15_18_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_classif_6D_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_19_18_15_18_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_classif_6D_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_19_18_15_18_0004',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_classif_6D_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_19_18_15_18_0005',
],
DENSE_TRANSFORMATION_MSE=[ # run separately, not numbered 1-5 (EDIT: has to be re-run due to older scaling of 100X), this got 0.777 +/- 0.08
    #'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_13_03_09_40_0001',
    #'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_12_13_05_27_0001',
    #'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_12_12_45_17_0001',
    #'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_12_02_08_26_0001',
    #'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_12_02_07_58_0001',
    # These are the ones with 250X SCALING, USE THESE. Interestingly, drops to 0.547 +/- 0.05.
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_19_12_58_01_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_19_12_58_01_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_19_12_58_01_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_19_12_58_01_0004',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_06_19_12_58_01_0005',
],
DENSE_TRANSFORMATION_PW=[
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_19_18_13_26_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_19_18_13_26_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_19_18_13_26_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_19_18_13_26_0004',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_19_18_13_26_0005',
],
RGB_DIRECT_VECTOR_v1=[
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_06_19_08_01_25_0001',
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_06_19_08_01_25_0002',
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_06_19_08_01_25_0003',
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_06_19_08_01_25_0004',
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_06_19_08_01_25_0005',
],
RGB_DIRECT_VECTOR_v1_AUGM=[
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_06_19_08_04_35_0001',
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_06_19_08_04_35_0002',
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_06_19_08_04_35_0003',
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_06_19_08_04_35_0004',
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_06_19_08_04_35_0005',
],
RGB_DIRECT_VECTOR=[  # ran on marvin, similar as v1 but with 1/2 as many filters, 2x more encoder_feature_dim.
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_06_20_13_01_02_0001',
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_06_20_13_01_34_0001',
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_06_20_13_32_03_0001',
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_06_20_13_32_14_0001',
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_06_20_14_09_42_0001',
],
RGB_DIRECT_VECTOR_AUGM=[  # ran on marvin, similar as v1 but with 1/2 as many filters, 2x more encoder_feature_dim.
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_06_20_14_14_03_0001',
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_06_20_14_43_45_0001',
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_06_20_14_43_54_0001',
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_06_20_15_11_54_0001',
    'BC04_PourWater_v01_ntrain_0100_cam_rgb_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_06_20_15_12_03_0001',
],
RGBD_DIRECT_VECTOR=[  # (ran Aug 18) oh this might be v1 but whatever, performance was almost the same
    ## Wait, the first 5 here have inconsistent depth at runtime with water (but training data did not have).
    #'BC04_PourWater_v01_ntrain_0100_cam_rgbd_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_18_14_58_00_0001',
    #'BC04_PourWater_v01_ntrain_0100_cam_rgbd_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_18_14_58_00_0002',
    #'BC04_PourWater_v01_ntrain_0100_cam_rgbd_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_18_14_58_00_0003',
    #'BC04_PourWater_v01_ntrain_0100_cam_rgbd_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_18_14_58_00_0004',
    #'BC04_PourWater_v01_ntrain_0100_cam_rgbd_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_18_14_58_00_0005',
    'BC04_PourWater_v01_ntrain_0100_cam_rgbd_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_19_21_22_50_0001',
    'BC04_PourWater_v01_ntrain_0100_cam_rgbd_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_19_21_22_50_0002',
    'BC04_PourWater_v01_ntrain_0100_cam_rgbd_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_19_21_22_50_0003',
    'BC04_PourWater_v01_ntrain_0100_cam_rgbd_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_19_21_22_50_0004',
    'BC04_PourWater_v01_ntrain_0100_cam_rgbd_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_19_21_22_50_0005',
],
RGBD_DIRECT_VECTOR_AUGM=[  # (ran Aug 18) oh this might be v1 but whatever, performance was almost the same
    ## Wait, the first 5 here have inconsistent depth at runtime with water (but training data did not have).
    #'BC04_PourWater_v01_ntrain_0100_cam_rgbd_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_18_15_03_02_0001',
    #'BC04_PourWater_v01_ntrain_0100_cam_rgbd_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_18_15_03_02_0002',
    #'BC04_PourWater_v01_ntrain_0100_cam_rgbd_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_18_15_03_02_0003',
    #'BC04_PourWater_v01_ntrain_0100_cam_rgbd_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_18_15_03_02_0004',
    #'BC04_PourWater_v01_ntrain_0100_cam_rgbd_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_18_15_03_02_0005',
    'BC04_PourWater_v01_ntrain_0100_cam_rgbd_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_19_21_39_01_0001',
    'BC04_PourWater_v01_ntrain_0100_cam_rgbd_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_19_21_39_01_0002',
    'BC04_PourWater_v01_ntrain_0100_cam_rgbd_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_19_21_39_01_0003',
    'BC04_PourWater_v01_ntrain_0100_cam_rgbd_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_19_21_39_01_0004',
    'BC04_PourWater_v01_ntrain_0100_cam_rgbd_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_19_21_39_01_0005',
],
D_SEGM_DIRECT_VECTOR=[
    ## Segmented depth --> Image CNN --> Vector
    # Ran locally 08/25.... oops actually I ran 7 LOL, just don't use 2 of them.
    #'BC04_PourWater_v01_ntrain_0100_depth_segm_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_25_02_39_59_0001',
    #'BC04_PourWater_v01_ntrain_0100_depth_segm_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_25_02_41_45_0001',
    'BC04_PourWater_v01_ntrain_0100_depth_segm_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_25_03_09_09_0001',
    'BC04_PourWater_v01_ntrain_0100_depth_segm_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_25_03_09_09_0002',
    'BC04_PourWater_v01_ntrain_0100_depth_segm_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_25_03_09_09_0003',
    'BC04_PourWater_v01_ntrain_0100_depth_segm_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_25_03_28_15_0001',
    'BC04_PourWater_v01_ntrain_0100_depth_segm_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_25_04_02_52_0001',
],
RGB_SEGM_DIRECT_VECTOR=[
    # Ran 08/25 on cluster. Must be on SoftGym `toolflownet-pouring-depth-segm` branch.
    'BC04_PourWater_v01_ntrain_0100_rgb_segm_masks_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_25_14_38_40_0001',
    'BC04_PourWater_v01_ntrain_0100_rgb_segm_masks_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_25_14_38_40_0002',
    'BC04_PourWater_v01_ntrain_0100_rgb_segm_masks_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_25_14_38_40_0003',
    'BC04_PourWater_v01_ntrain_0100_rgb_segm_masks_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_25_14_38_40_0004',
    'BC04_PourWater_v01_ntrain_0100_rgb_segm_masks_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_25_14_38_40_0005',
],
RGBD_SEGM_DIRECT_VECTOR=[
    # Ran 08/25 on cluster. Must be on SoftGym `toolflownet-pouring-depth-segm` branch.
    'BC04_PourWater_v01_ntrain_0100_rgbd_segm_masks_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_25_14_30_07_0001',
    'BC04_PourWater_v01_ntrain_0100_rgbd_segm_masks_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_25_14_30_07_0002',
    'BC04_PourWater_v01_ntrain_0100_rgbd_segm_masks_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_25_14_30_07_0003',
    'BC04_PourWater_v01_ntrain_0100_rgbd_segm_masks_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_25_14_30_07_0004',
    'BC04_PourWater_v01_ntrain_0100_rgbd_segm_masks_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_25_14_30_07_0005',
],
D_DIRECT_VECTOR=[
    # We might want to do this again but with the water in the depth (TODO) but this is a very bad baseline anyway.
    'BC04_PourWater_v01_ntrain_0100_depth_img_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_25_23_50_57_0001',
    'BC04_PourWater_v01_ntrain_0100_depth_img_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_25_23_50_57_0002',
    'BC04_PourWater_v01_ntrain_0100_depth_img_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_25_23_50_57_0003',
    'BC04_PourWater_v01_ntrain_0100_depth_img_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_25_23_50_57_0004',
    'BC04_PourWater_v01_ntrain_0100_depth_img_pixel_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_25_23_50_57_0005',
],
STATE_GT_DIRECT_VECTOR=[
    ## This has state info. For this, I used 72-D state, 35 for tool, 35 for target, 2 for water in either cup.
    ## 35 for each cup is due to five 3D positions (15) and five 4D quaternions (20).
    ## We regenerated the data for this experiment. FYI, norm. performance of this is good:0.777 $\pm$ 0.05"
    #'BC04_PourWater_v01_ntrain_0100_state_mlp_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_26_23_46_33_0001',
    #'BC04_PourWater_v01_ntrain_0100_state_mlp_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_26_23_46_33_0002',
    #'BC04_PourWater_v01_ntrain_0100_state_mlp_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_26_23_46_33_0003',
    #'BC04_PourWater_v01_ntrain_0100_state_mlp_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_26_23_46_33_0004',
    #'BC04_PourWater_v01_ntrain_0100_state_mlp_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_26_23_46_33_0005',
    ### This has 32-D state, same as above but dumping quaternions.
    #'BC04_PourWater_v01_ntrain_0100_state_mlp_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_27_01_58_40_0001',
    #'BC04_PourWater_v01_ntrain_0100_state_mlp_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_27_01_58_40_0002',
    #'BC04_PourWater_v01_ntrain_0100_state_mlp_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_27_01_58_40_0003',
    #'BC04_PourWater_v01_ntrain_0100_state_mlp_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_27_01_58_40_0004',
    #'BC04_PourWater_v01_ntrain_0100_state_mlp_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_27_01_58_40_0005',
    # This has 22-D state, this feels much more natural. 10D for each box with (position, quaternion, dimensions height length width)
    # This gets norm. 0.768 $\pm$ 0.02 so it's similar to ToolFlowNet.
    'BC04_PourWater_v01_ntrain_0100_state_mlp_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_27_23_00_23_0001',
    'BC04_PourWater_v01_ntrain_0100_state_mlp_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_27_23_00_23_0002',
    'BC04_PourWater_v01_ntrain_0100_state_mlp_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_27_23_00_23_0003',
    'BC04_PourWater_v01_ntrain_0100_state_mlp_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_27_23_00_23_0004',
    'BC04_PourWater_v01_ntrain_0100_state_mlp_eepose_3DoF_ar_8_hor_100_scaleTarg_2022_08_27_23_00_23_0005',
],
STATE_LEARNED_DIRECT_VECTOR=[
    # TODO This might take a while for some reason, but at least results are a lot worse than with g.t. state. (72-D, don't use!)
    #'BC04_PourWater_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_27_16_21_09_0001',
    #'BC04_PourWater_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_27_16_21_09_0002',
    #'BC04_PourWater_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_27_16_21_09_0003',
    #'BC04_PourWater_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_27_16_21_09_0004',
    #'BC04_PourWater_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_27_16_21_09_0005',
    # I think we want to use this, as this has the 22-D state dimension instead of 72-D (the latter doesn't make sense).
    'BC04_PourWater_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_29_20_20_57_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_29_20_20_57_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_29_20_20_57_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_29_20_20_57_0004',
    'BC04_PourWater_v01_ntrain_0100_PCL_state_predictor_then_mlp_eepose_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_08_29_20_20_57_0005',
],

# --------------------------------------- NUM_DEMOS ------------------------------------- #
FLOW3D_SVD_PW_CONSIST_010_DEMOS=[
    'BC04_PourWater_v01_ntrain_0010_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_23_08_49_45_0001',
    'BC04_PourWater_v01_ntrain_0010_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_23_08_49_45_0002',
    'BC04_PourWater_v01_ntrain_0010_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_23_08_49_45_0003',
    'BC04_PourWater_v01_ntrain_0010_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_23_08_49_45_0004',
    'BC04_PourWater_v01_ntrain_0010_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_23_08_49_45_0005',
],
FLOW3D_SVD_PW_CONSIST_050_DEMOS=[
    'BC04_PourWater_v01_ntrain_0050_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_22_15_04_13_0001',
    'BC04_PourWater_v01_ntrain_0050_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_22_15_04_13_0002',
    'BC04_PourWater_v01_ntrain_0050_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_22_15_04_13_0003',
    'BC04_PourWater_v01_ntrain_0050_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_22_15_04_13_0004',
    'BC04_PourWater_v01_ntrain_0050_PCL_PNet2_svd_pointwise_ee2flow_3DoF_ar_8_hor_100_scalePCL_noScaleTarg_2022_06_22_15_04_13_0005',
],

# ======================================== new rot representations ========================================== #
# Also includes repeats of our earlier axis-angle results just in case
PCL_DIRECT_VECTOR_MSE_NEWAPI_AXANG=[
    # Cluster 5X, meant to reproduce earlier results.
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_eepose_convert_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_27_20_08_17_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_eepose_convert_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_27_20_08_17_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_eepose_convert_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_27_20_08_17_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_eepose_convert_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_27_20_08_17_0004',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_eepose_convert_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_27_20_08_17_0005',
],
PCL_DIRECT_VECTOR_MSE_INTRIN_AXANG=[
    # 5x on cluster.
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_eepose_convert_intrinsic_axis_angle_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_25_15_31_20_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_eepose_convert_intrinsic_axis_angle_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_25_15_31_20_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_eepose_convert_intrinsic_axis_angle_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_25_15_31_20_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_eepose_convert_intrinsic_axis_angle_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_25_15_31_20_0004',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_eepose_convert_intrinsic_axis_angle_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_25_15_31_20_0005',
],
PCL_DENSE_TRANSF_MSE_EXTRIN_AXANG=[
    # Ran 5x on cluster.
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_axis_angle_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_18_35_17_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_axis_angle_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_18_35_17_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_axis_angle_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_18_35_17_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_axis_angle_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_18_35_17_0004',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_axis_angle_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_18_35_17_0005',
],
PCL_DENSE_TRANSF_MSE_INTRIN_AXANG=[
    # Ran 5x on cluster.
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_intrinsic_axis_angle_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_18_27_41_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_intrinsic_axis_angle_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_18_27_41_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_intrinsic_axis_angle_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_18_27_41_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_intrinsic_axis_angle_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_18_27_41_0004',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_dense_tf_6D_MSE_eepose_convert_intrinsic_axis_angle_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_10_26_18_27_41_0005',
],
PCL_DIRECT_VECTOR_MSE_FRO_4D_ROTS=[
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_4D_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_30_09_31_56_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_4D_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_30_09_31_56_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_4D_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_30_09_31_56_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_4D_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_30_09_31_56_0004',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_4D_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_30_09_31_56_0005',
],
PCL_DIRECT_VECTOR_MSE_FRO_6D_ROTS=[
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_6D_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_28_20_02_44_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_6D_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_28_20_02_44_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_6D_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_28_20_02_44_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_6D_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_28_20_02_44_0004',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_6D_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_28_20_02_44_0005',
],
PCL_DIRECT_VECTOR_MSE_FRO_9D_ROTS=[
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_9D_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_11_02_11_17_29_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_9D_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_11_02_11_17_29_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_9D_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_11_02_11_17_29_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_9D_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_11_02_11_17_29_0004',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_9D_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_11_02_11_17_29_0005',
],
PCL_DIRECT_VECTOR_MSE_FRO_10D_ROTS=[
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_10D_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_30_10_18_43_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_10D_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_30_10_18_43_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_10D_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_30_10_18_43_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_10D_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_30_10_18_43_0004',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_rpmg_eepose_convert_rotation_10D_3DoF_ar_8_hor_100_rawPCL_scaleTarg_2022_09_30_10_18_43_0005',
],

# --------------------------------------------- segless --------------------------------------------- #
SEGLESS_NAIVE=[
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_3DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_11_15_52_36_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_3DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_11_15_52_36_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_3DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_11_15_52_36_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_3DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_11_15_52_36_0004',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_3DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_11_15_52_36_0005',
],
SEGLESS_WEIGHT_CONSIST=[
    # Use L1 for segmentation normalization. [note: has one failed run due to SVD...]
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_3DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_11_16_04_21_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_3DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_11_16_04_21_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_3DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_11_16_04_21_0003',
    #'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_3DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_11_16_04_21_0004', # failed due to SVD
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_3DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_11_16_04_21_0005',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_3DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_11_18_18_43_0001', #  make-up run
],
SEGLESS_WEIGHT_CONSIST_SOFTMAX=[
    # Use softmax for segmentation normalization, temperature 0.1.
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_3DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_11_21_31_10_0001',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_3DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_11_21_31_10_0002',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_3DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_11_21_31_10_0003',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_3DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_11_21_31_10_0004',
    'BC04_PourWater_v01_ntrain_0100_PCL_PNet2_svd_pointwise_segless_ee2flow_segless_3DoF_ar_8_hor_100_keep_0s_scalePCL_noScaleTarg_2022_10_11_21_31_10_0005',
],
)

# ================================================================================================ #
# ================================================================================================ #

def _debug_print(cfg):
    print('Config:')
    for key in cfg:
        print(f'cfg[{key}]:  {cfg[key]}')
    print()


def form_paths(paths):
    #assert len(paths) == 5, f'#paths: {len(paths)}\n{paths}'
    full_paths = []
    for tail in paths:
        divided_path = tail.split('_2022_')  # we did all this in 2022 :)
        head_dir = divided_path[0]
        full_path = join(DATA_PATH, head_dir, tail)
        full_paths.append(full_path)
    return full_paths


def check_single_dir(key, single_dir, suppress_warning=False):
    """Sanity checks.

    Mostly to reduce the likelihood that we made errors in copying and pasting directory
    names in the dictionaries used for computing statistics, etc.
    """
    assert os.path.exists(join(single_dir, 'eval.log')), single_dir
    assert os.path.exists(join(single_dir, 'progress.csv')), single_dir

    # This has all the models we saved. In BC04 we should have 21 models, going
    # from 0, 25, ..., 475, 500. BC03 is a different story.
    model_dir = join(single_dir, 'model')
    assert os.path.exists(model_dir)
    models = sorted([x for x in os.listdir(model_dir) if x[-4:]=='.tar'])
    if len(models) != 5 and (not suppress_warning):
        print(f'Warning! {model_dir}')
        print(f'  num models: {len(models)}')

    # Now check the variant.
    variant_dir = join(single_dir, 'variant.json')
    assert os.path.exists(variant_dir), variant_dir
    with open(variant_dir) as fh:
        variant = json.load(fh)

    # Can do further checks as needed based on `variant` and `single_dir`, etc.
    # Actually this is a bit misleading as we could have avoided the consistency
    # just by setting its lambda value to be 0, but we used this extra boolean.
    if 'NOCONSISTENCY' in key:
        assert not variant['use_consistency_loss'], variant
    if 'CONSISTENCY' in key and 'NOCONSISTENCY' not in key:
        assert variant['use_consistency_loss'], variant

    if 'NOSKIP' in key:
        assert variant['remove_skip_connections']
    else:
        assert not variant['remove_skip_connections']

    # Check observation mode.
    obs_mode = variant['env_kwargs_observation_mode']
    if 'RGBD_SEGM_' in key:
        assert obs_mode == 'rgbd_segm_masks', obs_mode
    elif 'RGB_SEGM_' in key:
        assert obs_mode == 'rgb_segm_masks', obs_mode
    elif 'D_SEGM_' in key:
        assert obs_mode == 'depth_segm', obs_mode
    elif 'STATE_LEARNED_' in key:
        assert obs_mode == 'point_cloud', obs_mode
    elif 'STATE_' in key:
        assert obs_mode == 'state', obs_mode

    # An important check since we don't have the lambda in directory names.
    if ('PW_CONSIST_' in key and 'NOSCALE' not in key and 'NOSKIP' not in key and 'DEMOS' not in key):
        lambda_c = f"{key.split('_')[-2]}.{key.split('_')[-1]}"  # '_X_Y' --> 'X.Y'
        assert lambda_c == str(variant['lambda_consistency']), \
            "{} vs {}".format(lambda_c, variant['lambda_consistency'])

    # That's annoying, I changed the key from `data_augm` to `data_augm_img`.
    assert ('data_augm' in variant) or ('data_augm_img' in variant), variant.keys()
    if '_AUGM' in key:
        if 'data_augm' in variant:
            #print(variant['data_augm'])
            assert variant['data_augm']
        if 'data_augm_img' in variant:
            #print(variant['data_augm_img'])
            assert variant['data_augm_img'] == 'random_crop'
    else:
        if 'data_augm' in variant:
            #print(variant['data_augm'])
            assert not variant['data_augm']
        if 'data_augm_img' in variant:
            #print(variant['data_augm_img'])
            assert variant['data_augm_img'] == 'None'

    # Check the new rotations we added in late September 2022.
    if 'rotation_representation' in variant:
        rot_rep = variant['rotation_representation']
        if '_4D_ROTS' in key:
            assert rot_rep == 'rotation_4D', rot_rep
        elif '_6D_ROTS' in key:
            assert rot_rep == 'rotation_6D', rot_rep
        elif '_9D_ROTS' in key:
            assert rot_rep == 'rotation_9D', rot_rep
        elif '_10D_ROTS' in key:
            assert rot_rep == 'rotation_10D', rot_rep

    # Check segless experiments.
    if 'SEGLESS_' in key:
        seg_norm = variant['segmask_normalization']
        weight_c = variant['weight_consistency_loss']
        pred_seg = variant['predict_segmask']

        # segmentation normalization
        if '_WEIGHT_CONSIST_SOFTMAX' in key:
            assert seg_norm == 'softmax', f'{seg_norm}, {key}'
        elif '_WEIGHT_CONSIST' in key:
            assert seg_norm == 'l1', f'{seg_norm}, {key}'
        else:
            assert seg_norm is None, f'{seg_norm}, {key}'

        # weighting consistency using same weights from predicted seg mask
        if '_WEIGHT_CONSIST' in key:
            assert weight_c, f'{weight_c}, {key}'
        else:
            assert not weight_c, f'{weight_c}, {key}'

        # if we predict a seg mask weight at all
        if '_NAIVE' in key:
            assert not pred_seg, f'{pred_seg}, {key}'
        else:
            assert pred_seg, f'{pred_seg}, {key}'


def check_dir_get_performance_per_epoch(key, dirs, args, exp_perf):
    """Checks directory, get performance per epoch.

    NOTE! Unlike with the BC03 case, here the `eval.log` will look like this:

        {train AND eval logs}  // epoch 000
        {train only}
        ...
        {train only}
        {train AND eval logs}  // epoch 025
        {train only}
        ...
        ...
        ...
        {train only}
        {train AND eval logs}  // epoch 500


    Thus, subsample lines bhased on if they have 'info_done_final' recorded.
    There should be 21 such values if we did 500(+1) epochs.
    """

    # Use stats_xyz[k] for k in {0,1,2,3,4} as the random seed for key `xyz`.
    stats_info_done_final = []
    if len(dirs) != 5:
        print(f'WARNING, {len(dirs)} for {dirs}')

    # The dirs should not be the same, did I copy and paste incorrectly?
    if len(dirs) > 1:
        are_all_same = all(elem == dirs[0] for elem in dirs)
        assert not are_all_same, f'Something is wrong, check: {dirs}'

    # One `dirpath` has results for one random seed.
    for _,dirpath in enumerate(dirs):
        check_single_dir(key, dirpath, args.suppress_warning)

        # One line = one set of eval statistics with 10 evaluation episodes
        # Use `ast.literal_eval` to convert a dict's string to an actual dict.
        eval_log = join(dirpath, 'eval.log')
        with open(eval_log) as fh:
            lines = fh.readlines()
            lines = [ast.literal_eval(line.rstrip()) for line in lines]
            lines = [line for line in lines if 'info_done_final' in line]

        # Stats for this batch. Each `log` is the dict for one eval epoch.
        # In BC04, log['info_done_final'] is the average over 25 episodes, so
        # it will be x/25 for x in {0, 1, ..., 25}.
        info_done_final = [log['info_done_final'] for log in lines]
        stats_info_done_final.append(info_done_final)

    # Checks for ragged list / array, w/uneven numbers.
    lengths = list(set([len(x) for x in stats_info_done_final]))
    if len(lengths) > 1:
        print(f'WARNING, RAGGED ARRAY. For lengths: {lengths} take min?')
        minl = min(lengths)
        stats_info_done_final = [x[:minl] for x in stats_info_done_final]

    # Create np arrays.
    info_done_final = np.array(stats_info_done_final)
    if len(dirs) == 5:
        assert info_done_final.shape == (5,21), info_done_final.shape

    # Can do the average of the info_done_final so this means considering
    # (usually) 5 random seeds, then averaging so 25x5 = 125 test episdoes.
    # Actually I think we want perf_epoch_stde to be for normalized stats.
    perf_epoch_raw = np.mean(info_done_final, axis=0)  # (21,)
    perf_epoch_stde = np.std(info_done_final / exp_perf, axis=0) / np.sqrt(len(dirs))

    # 08/24/2022: let's just record another version with stde for raw stats.
    perf_epoch_stde_raw = np.std(info_done_final, axis=0) / np.sqrt(len(dirs))

    # Now just compute other statistics from this. NOTE: perf_max_idx is of
    # form array([x]), sometimes with more items if epochs have equal value,
    # in which case it's in ascending order.
    perf_epoch_norm = perf_epoch_raw / exp_perf
    perf_raw_max = np.max(perf_epoch_raw)
    perf_norm_max = np.max(perf_epoch_norm)
    perf_max_idx = np.where(perf_epoch_norm == perf_norm_max)[0]

    # Another way of computing statistics: we report the average across all
    # epochs (except 1st one as that should be all 0) and that gives us one
    # stat for all runs, take average of the 5.
    all_stats = info_done_final[:,1:]
    if not all_stats.shape == (5,20):
        print(f'WARNING: all_stats: {all_stats.shape} for {key}, dir: {dirs[0]}')
    perfavg_raw = np.mean(all_stats, axis=1)  # (5,) -- one per random seed
    perfavg_norm = perfavg_raw / exp_perf # (5,) -- one per random seed
    perfavg_stde = np.std(perfavg_norm) / np.sqrt(len(dirs))  # stde NORM
    perfavg_stde_raw = np.std(perfavg_raw) / np.sqrt(len(dirs))  # stde RAW

    # Reminder, `perf_max_idx` is our best performing epoch, but we use [0]
    # since we can have ties, and so we just want the first 'max idx.'
    stuff = {
        'perf_epoch_raw': perf_epoch_raw,   # avg raw perf over epochs
        'perf_epoch_norm': perf_epoch_norm, # avg norm perf over epochs
        'perf_epoch_stde': perf_epoch_stde, # std error mean over epochs
        'perf_raw_max': perf_raw_max,       # max avg raw perf
        'perf_norm_max': perf_norm_max,     # max avg norm perf
        'perf_max_idx': perf_max_idx,       # max perf epoch (will be same)
        'perf_max_stderr': perf_epoch_stde[perf_max_idx[0]],
        'perf_max_stderr_raw': perf_epoch_stde_raw[perf_max_idx[0]],
        # These are now the same as the above but instead of picking one epoch,
        # we get an average over all epochs after the 1st. Then that gives us
        # one number for each of the 5, then we can take a stderr.
        'perfavg_raw': np.mean(perfavg_raw),  # (5,) --> scalar
        'perfavg_norm': np.mean(perfavg_norm),  # (5,) --> scalar
        'perfavg_stde': perfavg_stde,
        'perfavg_stde_raw': perfavg_stde_raw, # same but RAW pef.
        'null': np.array([0]),  # just to get something here
    }
    return stuff


def format_number_str(x):
    """For the maximum epoch. Just use nothing if this is the other eval statistic
    which doesn't take a max over epoch, though.
    """
    x = str(x).zfill(2)  # need 0 idx here
    if x == '00':
        x = '  '
    return x


def table_rollout_performance(args):
    """Now try and print in a table-like format.

    Also report it like 20 rollouts per snapshot, etc.
    """
    keys1 = sorted(list(MM_ONE_SPHERE.keys()))
    keys2 = sorted(list(MM_ONE_SPHERE_6DOF.keys()))
    keys3 = sorted(list(POUR_WATER.keys()))
    keys4 = sorted(list(POUR_WATER_6DOF.keys()))
    allkeys = list(set(keys1+keys2+keys3+keys4))

    # Key (e.g., 'DENSE_TF_MSE') --> dict of stats, w/keys: 'perf_epoch_raw', etc.
    stats_s4d = {}  # scooping 4DoF
    stats_s6d = {}  # scooping 6DoF
    stats_p3d = {}  # pouring 3DoF
    stats_p6d = {}  # pouring 6DoF

    # Iterate through keys, compute various statistics.
    for key in allkeys:
        if key in MM_ONE_SPHERE:
            dirs = form_paths(MM_ONE_SPHERE[key])
            if len(dirs) > 0:
                perfdict = check_dir_get_performance_per_epoch(
                        key, dirs, args, DEMO_PERF['MMOneSphere'])
                stats_s4d[key] = perfdict
        if key in MM_ONE_SPHERE_6DOF:
            dirs = form_paths(MM_ONE_SPHERE_6DOF[key])
            if len(dirs) > 0:
                perfdict = check_dir_get_performance_per_epoch(
                        key, dirs, args, DEMO_PERF['MMOneSphere_6DoF'])
                stats_s6d[key] = perfdict
        if key in POUR_WATER:
            dirs = form_paths(POUR_WATER[key])
            if len(dirs) > 0:
                perfdict = check_dir_get_performance_per_epoch(
                        key, dirs, args, DEMO_PERF['PourWater'])
                stats_p3d[key] = perfdict
        if key in POUR_WATER_6DOF:
            dirs = form_paths(POUR_WATER_6DOF[key])
            if len(dirs) > 0:
                perfdict = check_dir_get_performance_per_epoch(
                        key, dirs, args, DEMO_PERF['PourWater_6DoF'])
                stats_p6d[key] = perfdict

    # Print the table using a specified ordering / subset of the keys above.
    rowkeys = [  # BASELINES
        'DIRECT_VECTOR_MSE',
        'DIRECT_VECTOR_PW',
        'DENSE_TRANSFORMATION_MSE',
        'DENSE_TRANSFORMATION_PW',
        'D_DIRECT_VECTOR',          # new for rebuttal
        'D_SEGM_DIRECT_VECTOR',     # new for rebuttal
        'RGB_DIRECT_VECTOR',
        'RGB_SEGM_DIRECT_VECTOR',   # new for rebuttal
        'RGBD_DIRECT_VECTOR',       # new for rebuttal
        'RGBD_SEGM_DIRECT_VECTOR',  # new for rebuttal
        None,  # ABLATIONS
        'FLOW3D_SVD_PW_CONSIST_NOSKIP',
        'FLOW3D_MSE_AFTER_SVD_CONSISTENCY',
        'FLOW3D_PW_BEFORE_SVD_NOCONSISTENCY',
        None,  # OUR METHOD, with different lambda values (including no consistency ablation).
        'FLOW3D_SVD_PW_CONSIST_0_0',
        'FLOW3D_SVD_PW_CONSIST_0_1',
        'FLOW3D_SVD_PW_CONSIST_0_5',
        'FLOW3D_SVD_PW_CONSIST_1_0',
        None,  # state experiments (supplement)
        'STATE_GT_DIRECT_VECTOR',
        'STATE_LEARNED_DIRECT_VECTOR',
        None,  # use these for confirming / verifying existing results with newer API from mid-Sept 2022 onwards
        'PCL_DIRECT_VECTOR_MSE_NEWAPI_AXANG',  # this is extrinsic (local)
        'PCL_DIRECT_VECTOR_MSE_INTRIN_AXANG',  # this is intrinsic (local)
        'RGB_DIRECT_VECTOR_MSE_NEWAPI_AXANG',
        'PCL_DENSE_TRANSF_MSE_EXTRIN_AXANG',  # doing extrinsic but dense transf
        'PCL_DENSE_TRANSF_MSE_INTRIN_AXANG',  # doing intrinsic but dense transf
        None,  # NEWER ROTATION REPRESENTATIONS, for late Sept 2022 onwards.
        'PCL_DIRECT_VECTOR_MSE_FRO_4D_ROTS',
        'PCL_DIRECT_VECTOR_MSE_FRO_6D_ROTS',
        'PCL_DIRECT_VECTOR_MSE_FRO_9D_ROTS',
        'PCL_DIRECT_VECTOR_MSE_FRO_10D_ROTS',
        None,  # segless results, October 2022.
        'SEGLESS_NAIVE',
        'SEGLESS_WEIGHT_CONSIST',
        'SEGLESS_WEIGHT_CONSIST_SOFTMAX',
        None,  # augmentation, but we should only do this if we have point cloud augmentations
        'RGB_DIRECT_VECTOR_AUGM',  # move to new table
        'RGBD_DIRECT_VECTOR_AUGM', # move to new table, new for rebuttal
        None,  # (smaller-scale experiment) this stuff is mainly debugging scales
        'FLOW3D_SVD_PW_CONSIST_0_1_NOSCALE',
        'DIRECT_VECTOR_MSE_NOSCALE',
        'DIRECT_VECTOR_MSE_MULT_ROT_250',  # really only for 4DoF scooping, as we already did this for pouring
        'RGB_DIRECT_VECTOR_NOSCALE',
        None,  # (smaller-scale experiment)
        'FLOW3D_GAUSSNOISE_005',
        'FLOW3D_GAUSSNOISE_010',
        None,  # (smaller-scale experiment)
        'FLOW3D_SVD_PW_CONSIST_010_DEMOS',
        'FLOW3D_SVD_PW_CONSIST_050_DEMOS',
        None,  # (smaller-scale experiment)
        'FLOW3D_REDUCETOOL_010',
    ]

    # Two tables: (1) using the best epochs, (2) average over epochs.
    def _print_results(statistic):
        if statistic == 'best':
            k_raw = 'perf_raw_max'
            k_norm = 'perf_norm_max'
            k_stde = 'perf_max_stderr'
            k_stde_r = 'perf_max_stderr_raw'
            k_idx = 'perf_max_idx'
        elif statistic == 'average':
            k_raw = 'perfavg_raw'
            k_norm = 'perfavg_norm'
            k_stde = 'perfavg_stde'
            k_stde_r = 'perfavg_stde_raw'
            k_idx = 'null'  # in this case we don't have a max idx
        else:
            raise NotImplementedError()

        print()
        print('='*150)
        if args.show_raw_perf:
            header_row = (f"{'Model':<35}  {'ScoopRaw (4D)          '} "
                    f"  {'ScoopRaw (6D)          '}  {'PourRaw (3D)        '} "
                    f"  {'PourRaw (6D)          '}   {'AvgRaw'}")
        else:
            header_row = (f"{'Model':<35}  {'ScoopNorm (4D)         '} "
                    f"  {'ScoopNorm (6D)         '}  {'PourNorm (3D)       '} "
                    f"  {'PourNorm (6D)         '}   {'AvgNorm'}")
        print(header_row)
        print('-'*150)
        empty = ' '*20  # use 20 if reporting $\pm$, or 18 if just +/-

        # For each row, build a string, then print.
        for rkey in rowkeys:
            if rkey is None:
                print('-'*120)
                continue

            # Build a string.
            row = f'{rkey:<35} &'

            # --------------------------- Scooping 4DoF
            if rkey in stats_s4d:
                i_s4d = stats_s4d[rkey]
                s4d_raw = i_s4d[k_raw]
                s4d_norm = i_s4d[k_norm]
                s4d_stde = i_s4d[k_stde]
                s4d_stde_r = i_s4d[k_stde_r]
                s4d_idx = format_number_str(i_s4d[k_idx][0])
                if args.show_raw_perf:
                    row += f'  {s4d_raw:0.3f} $\pm$ {s4d_stde_r:0.2f} ({s4d_idx}) &'
                else:
                    row += f'  {s4d_norm:0.3f} $\pm$ {s4d_stde:0.2f} ({s4d_idx}) &'
            else:
                row += f'  {empty}  &'

            # --------------------------- Scooping 6DoF
            if rkey in stats_s6d:
                i_s6d = stats_s6d[rkey]
                s6d_raw = i_s6d[k_raw]
                s6d_norm = i_s6d[k_norm]
                s6d_stde = i_s6d[k_stde]
                s6d_stde_r = i_s6d[k_stde_r]
                s6d_idx = format_number_str(i_s6d[k_idx][0])
                if args.show_raw_perf:
                    row += f'  {s6d_raw:0.3f} $\pm$ {s6d_stde_r:0.2f} ({s6d_idx}) &'
                else:
                    row += f'  {s6d_norm:0.3f} $\pm$ {s6d_stde:0.2f} ({s6d_idx}) &'
            else:
                row += f'  {empty}  &'

            # --------------------------- Pouring 3DoF
            if rkey in stats_p3d:
                i_p3d = stats_p3d[rkey]
                p3d_raw = i_p3d[k_raw]
                p3d_norm = i_p3d[k_norm]
                p3d_stde = i_p3d[k_stde]
                p3d_stde_r = i_p3d[k_stde_r]
                p3d_idx = format_number_str(i_p3d[k_idx][0])
                if args.show_raw_perf:
                    row += f'  {p3d_raw:0.3f} $\pm$ {p3d_stde_r:0.2f} ({p3d_idx}) &'
                else:
                    row += f'  {p3d_norm:0.3f} $\pm$ {p3d_stde:0.2f} ({p3d_idx}) &'
            else:
                row += f'  {empty}  &'

            # --------------------------- Pouring 6DoF
            if rkey in stats_p6d:
                i_p6d = stats_p6d[rkey]
                p6d_raw = i_p6d[k_raw]
                p6d_norm = i_p6d[k_norm]
                p6d_stde = i_p6d[k_stde]
                p6d_stde_r = i_p6d[k_stde_r]
                p6d_idx = format_number_str(i_p6d[k_idx][0])
                if args.show_raw_perf:
                    row += f'  {p6d_raw:0.3f} $\pm$ {p6d_stde_r:0.2f} ({p6d_idx}) &'
                else:
                    row += f'  {p6d_norm:0.3f} $\pm$ {p6d_stde:0.2f} ({p6d_idx}) &'
            else:
                row += f'  {empty}  &'

            # Avg of raw and norm, requires all keys to be present.
            if (rkey in stats_s4d and rkey in stats_s6d and rkey in stats_p3d and rkey in stats_p6d):
                a_raw  = np.mean(np.array([ s4d_raw,  s6d_raw,  p3d_raw,  p6d_raw]))
                a_norm = np.mean(np.array([s4d_norm, s6d_norm, p3d_norm, p6d_norm]))
                if args.show_raw_perf:
                    row += f'  {a_raw:0.3f}'
                else:
                    row += f'  {a_norm:0.3f}'
            else:
                row += f' '

            row += f'  \\\\'
            print(row)
        print('='*150)

    # Print two tables.
    _debug_str = 'RAW_PERFORMANCE' if args.show_raw_perf else 'NORMALIZED_PERFORMANCE'
    print(f'\n\n ------------- THE BEST EPOCH ({_debug_str}) ----------------- :')
    _print_results(statistic='best')
    if args.show_avg_epoch:
        print('\nAVERAGE OVER ALL EVAL EPOCHS:')
        _print_results(statistic='average')
    print(f'\nNOTE: using the following args:\n{args}')


def table_rollout_performance_3dof_demos(args):
    """Just the 3DoF demos."""
    stats_s = {}

    # Iterate through keys, compute various statistics.
    for key in list(MM_ONE_SPHERE_TRANS_ONLY.keys()):
        dirs = form_paths(MM_ONE_SPHERE_TRANS_ONLY[key])
        if len(dirs) > 0:
            perfdict = check_dir_get_performance_per_epoch(
                    key, dirs, args, DEMO_PERF['MMOneSphereTransOnly'])
            stats_s[key] = perfdict

    # Print the table using a specified ordering / subset of the keys above.
    rowkeys = [
        'FLOW3D_SVD_PW_CONSIST_0_1',
        'SEGM_PN2_AVG_LAYER',
        'NAIVE_CLASS_VECTOR_MSE',
    ]

    # Special case for this 3DoF demo for scooping.
    def _print_results():
        k_raw = 'perf_raw_max'
        k_norm = 'perf_norm_max'
        k_stde = 'perf_max_stderr'
        k_idx = 'perf_max_idx'
        # Actually the naive method is even better w.r.t. these metrics below:
        #k_raw = 'perfavg_raw'
        #k_norm = 'perfavg_norm'
        #k_stde = 'perfavg_stde'
        #k_idx = 'null'  # in this case we don't have a max idx

        print()
        print('='*120)
        header_row = (f"{'Model':<35}  {'ScoopRaw'}  {'ScoopNorm'} ")
        print(header_row)
        print('-'*120)
        empty = ' '*18

        # For each row, build a string, then print.
        for rkey in rowkeys:
            if rkey is None:
                print('-'*120)
                continue

            # Build a string.
            row = f'{rkey:<35} &'
            if rkey in stats_s:
                i_s = stats_s[rkey]
                s_raw = i_s[k_raw]
                s_norm = i_s[k_norm]
                s_stde = i_s[k_stde]
                s_idx = str(i_s[k_idx][0]).zfill(2)  # need 0 idx here
                row += f'  {s_raw:0.3f}  &  {s_norm:0.3f} +/- {s_stde:0.2f} ({s_idx})  &'

            row += f'  \\\\'
            print(row)
        print('='*120)
    _print_results()


if __name__ == '__main__':
    """BC04 settings, now we should have standardized at 500 epochs.

    Just run as python analysis/results_bc04.py.  :)
    """
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='MMOneSphere')
    p.add_argument('--suppress_warning', type=int, default=1)
    p.add_argument('--show_raw_perf', action='store_true', default=False)
    p.add_argument('--show_avg_epoch', action='store_true', default=False)
    args = p.parse_args()

    table_rollout_performance(args)
    #table_rollout_performance_3dof_demos(args)
