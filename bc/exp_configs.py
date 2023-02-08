"""Configs used for PHYSICAL experiments.

remove_zeros_PCL: seems like this should usually be True? Not relevant
    if using images. Also in real we don't have this issue right?

method_flow2act: will only matter if we use a method that predicts flow,
    does not affect code otherwise.

scale_pcl_flow: If true, scale all point cloud and flow values by
    `scale_pcl_val` as if we re-express obs xyz and flow in different
    units. Setting as 250 (instead of 1000) so that targets are also in
    (-1,1) and which makes MSEs comparable with other methods (though
    this is technically not accurate I think due to the tool masking
    affecting MSE computations)? NOTE: we only have this due to sim and
    dealing with action repeat > 1, not sure if we want in real. Also be
    careful if we scale flow because the pointwise loss needs to be done
    with points in the same units.

scale_targets: Careful! This has caused a lot of confusion. This rescales
    EE targets to be in a larger range, which has been necessary for data
    with smaller action / flow magnitudes.
"""

# Default config, for which we add values and/or override.
DEFAULT_CFG = dict(
    use_consistency_loss=False,
    lambda_consistency=0.0,
    lambda_pos=1.0,
    lambda_rot=1.0,
    use_geodesic_dist=False,
    method_flow2act='mean',
    remove_zeros_PCL=True,
    scale_pcl_flow=False,
    scale_pcl_val=None,
    scale_targets=False,
    data_augm_PCL='None',
    image_size_crop=100,
)

# Proposed method, segm PN++, flow, SVD, pointwise loss (for 3DoF).
SVD_POINTWISE_3D_FLOW = dict(
    obs_type='point_cloud',
    act_type='flow',
    encoder_type='pointnet_svd_pointwise',
    method_flow2act='svd',
    use_consistency_loss=True,
    lambda_consistency=0.1,
    scale_targets=True,
    data_augm_PCL='rot_gaussian_0.00001',
)

SVD_POINTWISE_3D_FLOW_EDDIE = dict(
    obs_type='point_cloud',
    act_type='flow',
    encoder_type='pointnet_svd_pointwise',
    method_flow2act='svd',
    use_consistency_loss=True,
    lambda_consistency=0.1,
    scale_targets=True,
    # data_augm_PCL='rot_gaussian_0.0001',
)

SVD_POINTWISE_6D_FLOW = dict(
    obs_type='point_cloud',
    act_type='flow',
    encoder_type='pointnet_svd_pointwise_6d_flow',
    method_flow2act='svd',
    use_consistency_loss=True,
    lambda_consistency=0.1,
    scale_targets=False,
)

# ----------------------- Naive 'classification PN++' ---------------------- #

# Naive classif. PN++ regress to vector method, for 3DoF data only! The
# `act_type` of `ee` will only keep the first 3 parts, the translation only.
# For translation-only data with deltas in meters, it seems like we do need
# the `scale_targets=True` for stable learning.
NAIVE_CLASS_PN2_TO_VECTOR_3DoF = dict(
    obs_type='point_cloud',
    act_type='ee',
    encoder_type='pointnet',
    scale_targets=True,
    data_augm_PCL='rot_gaussian_0.0001',
)

# Naive classif. PN++ now for 6D vector targets, so use `eepose`. If scaling
# then we've used lambda_rot of 100 to increase the rotation weight but this
# needs to be checked against whatever scaling we do, and we may want to avoid
# such scaling to start by keeping the lambdas both as 1.
NAIVE_CLASS_PN2_TO_VECTOR_6DoF = dict(
    obs_type='point_cloud',
    act_type='eepose',
    encoder_type='pointnet',
    scale_targets=True,
    lambda_pos=1.0,
    lambda_rot=1.0,
    data_augm_PCL='gaussian_0.00001',
)

# ----------------------- Usable but de-prioritize ---------------------- #

# Segm PN++ to per-point preds, then average. Regress to MSE. This is ONLY
# applicable with 3DoF translation data. In sim this was giving good results,
# though it was similar to the classification PN++ so I don't think we will
# end up testing this a lot.
NAIVE_SEGM_PN2_TO_FLOW_AVG_3DoF = dict(
    obs_type='point_cloud',
    act_type='ee',
    encoder_type='pointnet_avg',
    scale_targets=True,
    data_augm_PCL='rot_gaussian_0.00001',
)