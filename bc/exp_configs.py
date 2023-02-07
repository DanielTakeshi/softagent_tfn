"""Experiment configurations for methods we tested.

Some important parameters:

method_flow2act: will only matter if we use a method that predicts flow,
    does not affect code otherwise.

scale_pcl_flow: If true, scale all point cloud and flow values by
    `scale_pcl_val` as if we re-express obs xyz and flow in different
    units. Setting as 250 (instead of 1000) so that targets are also in
    (-1,1) and which makes MSEs comparable with other methods (though
    this is technically not accurate I think due to the tool masking
    affecting MSE computations)?

scale_targets: Rescales targets to be in the range of (-1,1) per component,
which we found helped the baseline methods (e.g., Direct Vector MSE).
"""

# Default config, for which we add values and/or override.
DEFAULT_CFG = dict(
    use_dense_loss=False,
    use_consistency_loss=False,
    lambda_dense=0.0,
    lambda_consistency=0.0,
    lambda_pos=1.0,
    lambda_rot=1.0,
    rpmg_lambda=0.01,
    use_geodesic_dist=False,
    method_flow2act='mean',
    rotation_representation='axis_angle',
    remove_zeros_PCL=True,
    zero_center_PCL=False,  # zero-center PCLs individually
    scale_pcl_flow=False,
    scale_pcl_val=1,
    scale_targets=False,
    data_augm_img='None',  # a string, not boolean
    data_augm_PCL='None',  # a string, not boolean (applied on _scaled_ PCL)
    image_size_crop=100,
    separate_MLPs_R_t=False,
    dense_transform=False,
    remove_skip_connections=False,
    gaussian_noise_PCL=0.0,
    reduce_tool_PCL='None',
    reduce_tool_points=False,
    tool_point_num=10,
    log_flow_visuals=True,  # if a method doesn't use flow it won't log
)

# ----------------------- SVD and pointwise methods ---------------------- #

# Proposed method, segm PN++, ee2flow, SVD, pointwise loss (for 6DoF).
# Edit: to clarify, PN++ produces 3D flow. We were debating whether we
# use this vs 6D flow, and for the CoRL 2022, submission we used THIS.
# We can also use this for 3D translation only data but for that PLEASE
# USE THE 'flow' ACTION TYPE, NOT 'ee2flow'.
SVD_POINTWISE_EE2FLOW = dict(
    obs_type='point_cloud',
    act_type='ee2flow',
    encoder_type='pointnet_svd_pointwise',
    method_flow2act='svd',
    use_consistency_loss=True,
    lambda_consistency=0.1,  # use 0.5 for pouring?
    scale_pcl_flow=True,
    scale_pcl_val=250,
)

# Same as our 3D flow method but using an extra dense loss (06/08/2022
# meeting). Using the same encoder to reduce code complexity.
SVD_POINTWISE_EE2FLOW_w_DENSE = dict(
    obs_type='point_cloud',
    act_type='ee2flow',
    encoder_type='pointnet_svd_pointwise',
    method_flow2act='svd',
    use_dense_loss=True,
    use_consistency_loss=True,
    lambda_dense=1.0,
    lambda_consistency=0.1,
    scale_pcl_flow=True,
    scale_pcl_val=250,
)

# (Note: should not generally be run since it's for translation only demos)
# Another way of doing flow, except here we have "6D flow"; it may further
# decouple the translation and rotation. ASSUME TRANSLATION ONLY HERE. Still
# use rotation to produce the pointwise loss (so rotation should be identity).
# Since it's 3DoF we assume we can use `method_flow2act=mean`.
SVD_POINTWISE_6D_FLOW = dict(
    obs_type='point_cloud',
    act_type='flow',
    encoder_type='pointnet_svd_pointwise_6d_flow',
    method_flow2act='mean',
    use_consistency_loss=False,
    lambda_consistency=0.1,
    scale_pcl_flow=True,
    scale_pcl_val=250,
    separate_MLPs_R_t=False,
)

# 6D flow, but using SVD to extract actions. This was originally our proposed
# method but then from PourWater, the 3D flow was just as good. This is WITH
# the consistency loss and WITHOUT the separate MLPs for (R,t).
SVD_POINTWISE_6D_EE2FLOW_SVD = dict(
    obs_type='point_cloud',
    act_type='ee2flow',
    encoder_type='pointnet_svd_pointwise_6d_flow',
    method_flow2act='svd',
    use_consistency_loss=True,
    lambda_consistency=0.1,
    scale_pcl_flow=True,
    scale_pcl_val=250,
)

# Same as our method but using an extra dense loss (06/08/2022 meeting).
# Using the same encoder to reduce code complexity but we also need the
# 6d action type, since dense supervision on predicted flow (before SVD)
# requires 6d actions, though pointwise supervision (after SVD) needs 3d
# actions, but we can go from 6d -> 3d by adding the trans and rot parts.
# Detect this case with `use_dense_loss`.
SVD_POINTWISE_6D_EE2FLOW_SVD_w_DENSE = dict(
    obs_type='point_cloud',
    act_type='ee2flow_sep_rt',
    encoder_type='pointnet_svd_pointwise_6d_flow',
    method_flow2act='svd',
    use_dense_loss=True,
    use_consistency_loss=True,
    lambda_dense=1.0,
    lambda_consistency=0.1,
    scale_pcl_flow=True,
    scale_pcl_val=250,
    separate_MLPs_R_t=False,
)

# 6D flow, but using pointwise loss before SVD.
SVD_PRE_POINTWISE_6D_EE2FLOW_SEP_RT_SVD = dict(
    obs_type='point_cloud',
    act_type='ee2flow_sep_rt',
    encoder_type='pointnet_svd_pointwise_6d_flow',
    method_flow2act='svd',
    use_consistency_loss=False,
    scale_pcl_flow=True,
    scale_pcl_val=250,
    separate_MLPs_R_t=False,
    gaussian_noise_PCL=0.0,  # normally 0
)

# 3D flow, but using pointwise loss before SVD. I found it easier to do this
# with a new encoder type, alternative is ee2flow_sep_rt but combining the
# trans and rot (via trans+rot) to go from 6D to 3D. NOTE: no consistency as
# that would add another target to the same trunk (the predicted flows) but
# we could test.
SVD_PRE_POINTWISE_3D_EE2FLOW_SVD = dict(
    obs_type='point_cloud',
    act_type='ee2flow',
    encoder_type='pointnet_svd_pointwise_PW_bef_SVD',
    method_flow2act='svd',
    use_consistency_loss=False,
    scale_pcl_flow=True,
    scale_pcl_val=250,
    separate_MLPs_R_t=False,
    gaussian_noise_PCL=0.0,
)

# ----------------------- SVD / flow abalation methods ---------------------- #
# See above for the 'proposed' SVD / flow methods with 3D and 6D flow.
# Here we will focus on some other ablations.

# Same as `SVD_POINTWISE_SVD` except that we will just take the (R,t) from
# the flow, turn it to a 6D vector, and do MSE on that. MSE instead of PW.
# NOTE(daniel): for now the flow visualizations are not supported, turn off!
# TODO(daniel): we should eventually get the flow viz supported...
SVD_3D_FLOW_EEPOSE_MSE_LOSS = dict(
    obs_type='point_cloud',
    act_type='eepose',
    encoder_type='pointnet_svd',  # just keep this name
    method_flow2act='svd',  # this should be a duplicate of the forward pass
    use_consistency_loss=True,  # actually this makes sense to use
    lambda_consistency=0.1,  # maybe 0.5 for PourWater
    scale_targets=True,
    lambda_pos=1.0,
    lambda_rot=1.0,  # 100 for MixedMedia, 1 for PourWater?
    log_flow_visuals=False,  # important for now
)

# Same as `SVD_POINTWISE_6D_EE2FLOW_SVD` except that we will just take the
# (R,t) produced from the flow, turn it to a 6D vector, and do MSE on that.
# TL;DR test MSE loss instead of pointwise tool flow loss. This will involve
# doing basically the same "flow2act" that we do at inference time, except
# it is done at TRAIN time, and so the forward pass just returns a 6D vector.
SVD_6D_FLOW_EEPOSE_MSE_LOSS = dict(
    obs_type='point_cloud',
    act_type='eepose',
    encoder_type='pointnet_svd_6d_flow_mse_loss',
    method_flow2act='svd',  # this should be a duplicate of the forward pass
    use_consistency_loss=True,  # actually this makes sense to use
    lambda_consistency=0.1,
    scale_targets=True,
    lambda_pos=1.0,
    lambda_rot=100.0,  # 100 for MixedMedia, 1 for PourWater?
)

# Same as `SVD_POINTWISE_6D_EE2FLOW_SVD` except that we remove the skip
# connections in the segmentation PN++. This will test if there is something
# fundamental to the segmentation architecture that lets it work well here.
# (I don't expect this to perform that well.) To keep things simple, use the
# same encoder type, so we don't have to keep checking for special cases.
SVD_POINTWISE_6D_EE2FLOW_SVD_NOSKIP = dict(
    obs_type='point_cloud',
    act_type='ee2flow',
    encoder_type='pointnet_svd_pointwise_6d_flow',
    method_flow2act='svd',
    use_consistency_loss=True,
    lambda_consistency=0.1,
    scale_pcl_flow=True,
    scale_pcl_val=250,
    remove_skip_connections=True,
)

# Same as no skip but with 3D flow.
SVD_POINTWISE_3D_EE2FLOW_SVD_NOSKIP = dict(
    obs_type='point_cloud',
    act_type='ee2flow',
    encoder_type='pointnet_svd_pointwise',
    method_flow2act='svd',
    use_consistency_loss=True,
    lambda_consistency=0.1,
    scale_pcl_flow=True,
    scale_pcl_val=250,
    remove_skip_connections=True,
)

# Same as our method but now testing what happens if we reduce tool points.
# Test this (06/09/2022) to see if # of tool points is a factor.
SVD_POINTWISE_6D_EE2FLOW_SVD_reducetool = dict(
    obs_type='point_cloud',
    act_type='ee2flow',
    encoder_type='pointnet_svd_pointwise_6d_flow',
    method_flow2act='svd',
    use_consistency_loss=True,
    lambda_consistency=0.1,
    scale_pcl_flow=True,
    scale_pcl_val=250,
    reduce_tool_PCL='pouring_v01',
)

# Same as our 3D flow method but reducing # of tool points.
SVD_POINTWISE_EE2FLOW_reducetool = dict(
    obs_type='point_cloud',
    act_type='ee2flow',
    encoder_type='pointnet_svd_pointwise',
    method_flow2act='svd',
    use_consistency_loss=True,
    lambda_consistency=0.5, # might want 0.5 for pouring actually
    scale_pcl_flow=True,
    scale_pcl_val=250,
    reduce_tool_PCL='pouring_v01',
)

# Same as above, but for `ScoopBall` tasks (3D and 6D) only.
SVD_POINTWISE_EE2FLOW_reducetool_mixedmedia = dict(
    obs_type='point_cloud',
    act_type='ee2flow',
    encoder_type='pointnet_svd_pointwise',
    method_flow2act='svd',
    use_consistency_loss=True,
    lambda_consistency=0.1,
    scale_pcl_flow=True,
    scale_pcl_val=250,
    reduce_tool_points=True,
    tool_point_num=10,
)

# ----------------------- Translation-only methods ---------------------- #

# Segm PN++ to flow, then average, regress to MSE (no pointwise loss). NOTE:
# when doing this, set `log_flow_visuals=False` in `launch_bc_mm.py`. This is
# ToolFlowNet (no SVD) as reported in the paper (supplement) for CoRL.
NAIVE_SEGM_PN2_TO_FLOW_AVG_3DoF = dict(
    obs_type='point_cloud',
    act_type='ee',
    encoder_type='pointnet_avg',
    scale_targets=True,
    log_flow_visuals=False,  # important for now
)

# Segm PN++ to flow, per-point flow MSE, average only at test time, so
# averaging is nondifferentiable, unlike PN++ with the averaging layer.
NAIVE_SEGM_PN2_TO_FLOW_MSE_3DoF = dict(
    obs_type='point_cloud',
    act_type='flow',
    encoder_type='pointnet',
    method_flow2act='mean',
    scale_pcl_flow=True,
    scale_pcl_val=250,
)

# ----------------------- Naive 'classification PN++' ---------------------- #

# Naive classif. PN++ regress to vector method. For translation-only demos.
NAIVE_CLASS_PN2_TO_VECTOR_3DoF = dict(
    obs_type='point_cloud',
    act_type='ee',
    encoder_type='pointnet',
    scale_targets=True,
)

# Here, scaling targets means translation only (not the rotation part).
# Since we have rotations, can optionally do geodesic for that part.
# Either way, it requires some weighted loss on the two parts. If scaling
# targets, use weight of 100 for rotation as that's roughly going to make
# our demonstration rotations equal to the largest translation magnitudes.

# DIRECT VECTOR (MSE) BASELINE as reported in the paper.
# UPDATE: actually if we do PourWater, then the magnitudes are going to
# be roughly the same (0.3 ish vs 0.5 ish) so I'd keep them at 1.
NAIVE_CLASS_PN2_TO_VECTOR_6DoF = dict(
    obs_type='point_cloud',  # can use 'point_cloud_gt_v{01,02}'
    act_type='eepose',
    encoder_type='pointnet',
    scale_targets=True,
    use_geodesic_dist=False,
    lambda_pos=1.0,
    lambda_rot=1.0,  # 100 for MixedMedia, 1 for PourWater?
)

# Network directly outputs quaternion (we convert to axis-angle later).
# For this we might want to keep weights equal for pos and rot.
NAIVE_CLASS_PN2_TO_VECTOR_7DoF = dict(
    obs_type='point_cloud',
    act_type='eepose',
    encoder_type='pointnet',
    scale_targets=True,
    use_geodesic_dist=True,
    lambda_pos=1.0,
    lambda_rot=1.0,
)

# Now Naive method BUT WITH POINTWISE LOSS! If this doesn't work we can
# probably conclude it's due to flow instead of the pointwise loss.
NAIVE_CLASS_PN2_TO_VECTOR_POINTWISE = dict(
    obs_type='point_cloud',
    act_type='ee2flow',
    encoder_type='pointnet_classif_6D_pointwise',
    scale_pcl_flow=True,
    scale_pcl_val=250,
)

# --------------------------- CNNs / images -------------------------- #

# Naive, images --> CNN --> regress to vector (translation only, MSE).
NAIVE_CNN_3DoF = dict(
    obs_type='cam_rgb',
    act_type='ee',
    encoder_type='pixel',
    data_augm_img='random_crop',
    scale_targets=True,
)

# Naive, images --> CNN --> regress to 6D action vector (with MSE loss).
# Note: originally I had lambda_rot=100 for Mixed Media (v01) but I think
# using just 1 and then scaling rotation is better (must change env hacks).
# However we should use lambda_rot=100 for consistency if we are doing stuff
# to compare with CoRL submission with MM v01. For MM v02, use lambda_rot=1.
NAIVE_CNN_6DoF = dict(
    obs_type='cam_rgb',
    act_type='eepose',
    encoder_type='pixel',
    scale_targets=True,
    data_augm_img='None',  # 'None' or 'random_crop' (in string form)
    use_geodesic_dist=False,
    lambda_pos=1.0,
    lambda_rot=1.0,
)

# Naive, RGBD --> CNN --> regress to vector (with MSE loss). Same comments
# apply as to the `NAIVE_CNN_6DoF` case regarding `lambda_rot`.
NAIVE_CNN_RGBD_6DoF = dict(
    obs_type='cam_rgbd',
    act_type='eepose',
    encoder_type='pixel',
    scale_targets=True,
    data_augm_img='None',  # 'None' or 'random_crop' (in string form)
    lambda_pos=1.0,
    lambda_rot=1.0,
)

# Depth -> CNN -> regress. Depth will only have 1 channel.
NAIVE_CNN_DEPTH_6DoF = dict(
    obs_type='depth_img',
    act_type='eepose',
    encoder_type='pixel',
    scale_targets=True,
    data_augm_img='None',  # 'None' or 'random_crop' (in string form)
    lambda_pos=1.0,
    lambda_rot=1.0,
)

# Naive, (Depth+Segm) --> CNN --> regress to vector (with MSE loss). Same
# comments apply as to the `NAIVE_CNN_6DoF` case regarding `lambda_rot`.
NAIVE_CNN_DEPTH_SEGM_6DoF = dict(
    obs_type='depth_segm',
    act_type='eepose',
    encoder_type='pixel',
    scale_targets=True,
    data_augm_img='None',  # 'None' or 'random_crop' (in string form)
    lambda_pos=1.0,
    lambda_rot=1.0,
)

# Naive, (RGB + Segm) --> CNN --> regress to vector (with MSE loss)
NAIVE_CNN_RGB_SEGM_6DoF = dict(
    obs_type='rgb_segm_masks',
    act_type='eepose',
    encoder_type='pixel',
    scale_targets=True,
    data_augm_img='None',  # 'None' or 'random_crop' (in string form)
    lambda_pos=1.0,
    lambda_rot=1.0,
)

# Naive, (RGB-D + Segm) --> CNN --> regress to vector (with MSE loss)
NAIVE_CNN_RGB_DEPTH_SEGM_6DoF = dict(
    obs_type='rgbd_segm_masks',
    act_type='eepose',
    encoder_type='pixel',
    scale_targets=True,
    data_augm_img='None',  # 'None' or 'random_crop' (in string form)
    lambda_pos=1.0,
    lambda_rot=1.0,
)

# --------------------------- Dense transformations -------------------------- #

# Treat a single tool point that we always observe (synthetically add it to
# the point cloud) and use that to output the transformation on and use as the
# center of rotation. The easiest seems to be to use the tool tip. We can get
# that from the existing data we have via the keypoints, and (b) we can query
# on-the-fly during evaluation.

# NOTE: please keep 'dense_tf_3D_' or 'dense_tf_6D_' patterns in encoder names.

# 3DoF translation only case, this is probably mainly for debugging.
# Set dense_transf_policy=True  --->  use segment. PN++, not classification.
# The forward pass through segment. PN++ directly returns the (N,3) output.
# Supervise with MSE?
DENSE_TRANSF_POLICY_TIP_3DoF = dict(
    obs_type='point_cloud',
    act_type='ee',
    encoder_type='pointnet_dense_tf_3D_MSE',
    scale_targets=True,
    dense_transform=True,
)

# 6DoF translation and (axis-angle) rotation using dense transformation.
# Set dense_transf_policy=True  --->  use segment. PN++, not classification.
# The forward pass through segment. PN++ directly returns the (N,6) output.
DENSE_TRANSF_POLICY_TIP_6DoF_MSE = dict(
    obs_type='point_cloud',
    act_type='eepose',
    encoder_type='pointnet_dense_tf_6D_MSE',
    scale_targets=True,
    lambda_pos=1.0,
    lambda_rot=1.0,  # 100 for MixedMedia, 1 for PourWater?
    dense_transform=True,
)

# Pointwise loss seems more principled. Note: this also requires scaling
# the pcl flow (not just the targets) and the action type is 'flow'.
DENSE_TRANSF_POLICY_TIP_6DoF_PW = dict(
    obs_type='point_cloud',
    act_type='ee2flow',
    encoder_type='pointnet_dense_tf_6D_pointwise',
    scale_pcl_flow=True,
    scale_pcl_val=250,
    dense_transform=True,
)

# Dense Transformation (MSE), now with EXTRINSIC axis-angle for all envs.
DENSE_TRANSF_EXTRINSIC_AXIS_ANGLE_6DoF_MSE = dict(
    obs_type='point_cloud',
    act_type='eepose_convert',
    rotation_representation='axis_angle',
    encoder_type='pointnet_dense_tf_6D_MSE',
    scale_targets=True,
    lambda_pos=1.0,
    lambda_rot=1.0,
    dense_transform=True,
)

# Dense Transformation (MSE), now with INTRINSIC axis-angle for all envs.
DENSE_TRANSF_INTRINSIC_AXIS_ANGLE_6DoF_MSE = dict(
    obs_type='point_cloud',
    act_type='eepose_convert',
    rotation_representation='intrinsic_axis_angle',
    encoder_type='pointnet_dense_tf_6D_MSE',
    scale_targets=True,
    lambda_pos=1.0,
    lambda_rot=1.0,
    dense_transform=True,
)

# --------------------------- State-based policy baseline -------------------------- #

# State-based policy baseline, uses state --> MLP --> action.
# Use the NEW observation type 'state' and a new encoder type.
STATE_POLICY_BASELINE = dict(
    obs_type='state',
    act_type='eepose',
    encoder_type='mlp',
    scale_targets=True,
    lambda_pos=1.0,
    lambda_rot=1.0,
)

# State-based policy baseline, now: PCL -> (PN++) -> (state). Train this FIRST.
# THEN we do the state policy baseline, but we never see ground truth state.
# We only see PCLs that are passed through the PN++ to produce the state.
# Use the POINT CLOUD observation but a different encoder.
# This is really a two-step procedure. First step should train the state predictor.
LEARNED_STATE_POLICY_BASELINE = dict(
    obs_type='point_cloud',
    act_type='eepose',
    encoder_type='state_predictor_then_mlp',
    scale_targets=True,
    lambda_pos=1.0,
    lambda_rot=1.0,
)

# ------------------- Rotation representation experiments ----------------------- #

# NOTE: [use_geodesic_dist] will not work with any of the following configs
# NOTE: only test different rotations with obs_type='point_cloud'.

# DIRECT VECTOR (MSE) BASELINE with EXTRINSIC AXIS-ANGLE FOR ALL ENVIRONMENTS
DIRECT_VECTOR_AXIS_ANGLE = dict(
    obs_type='point_cloud',
    act_type='eepose_convert',
    rotation_representation='axis_angle',
    encoder_type='pointnet',
    scale_targets=True,
    lambda_pos=1.0,
    lambda_rot=1.0,
)

# DIRECT VECTOR (MSE) BASELINE with _intrinsic_ AXIS-ANGLE FOR ALL ENVIRONMENTS
DIRECT_VECTOR_INTRINSIC_AXIS_ANGLE = dict(
    obs_type='point_cloud',
    act_type='eepose_convert',
    rotation_representation='intrinsic_axis_angle',
    encoder_type='pointnet',
    scale_targets=True,
    lambda_pos=1.0,
    lambda_rot=1.0,
)

# Direct Vector (MSE) with 4D rotations, converted to rotation matrices for losses.
# Using RPMG backprop changes
# Note the new encoder type here.
DIRECT_VECTOR_4D_ROTATION = dict(
    obs_type='point_cloud',
    act_type='eepose_convert',
    rotation_representation='rotation_4D',
    encoder_type='pointnet_rpmg',
    scale_targets=True,
    lambda_pos=1.0,
    lambda_rot=1.0,
    rpmg_lambda=0.01, # as used in the RPMG paper
)

# Direct Vector (MSE) with 6D rotations, converted to rotation matrices for losses.
# Using RPMG backprop changes
# Note the new encoder type here.
DIRECT_VECTOR_6D_ROTATION = dict(
    obs_type='point_cloud',
    act_type='eepose_convert',
    rotation_representation='rotation_6D',
    encoder_type='pointnet_rpmg',
    scale_targets=True,
    lambda_pos=1.0,
    lambda_rot=1.0,
    rpmg_lambda=0.01, # as used in the RPMG paper
)

# Direct Vector (MSE) with 6D rotations, using pointwise loss
# Using RPMG backprop changes
# Note the new encoder type here.
DIRECT_VECTOR_6D_ROTATION_POINTWISE = dict(
    obs_type='point_cloud',
    act_type='ee2flow',
    rotation_representation='rpmg_flow_6D',
    encoder_type='pointnet_rpmg_pointwise',
    # scale_pcl_flow=True,
    scale_pcl_flow=False,
    # scale_pcl_val=250,
    lambda_pos=1.0,
    lambda_rot=1.0,
    rpmg_lambda=0.01, # as used in the RPMG paper
)

# Direct Vector (MSE) with 9D rotations, converted to rotation matrices for losses.
# Using RPMG backprop changes
# Note the new encoder type here.
DIRECT_VECTOR_9D_ROTATION = dict(
    obs_type='point_cloud',
    act_type='eepose_convert',
    rotation_representation='rotation_9D',
    encoder_type='pointnet_rpmg',
    scale_targets=True,
    lambda_pos=1.0,
    lambda_rot=1.0,
    rpmg_lambda=0.01, # as used in the RPMG paper
)

# Direct Vector (MSE) with 10D rotations, converted to rotation matrices for losses.
# Using RPMG backprop changes
# Note the new encoder type here.
DIRECT_VECTOR_10D_ROTATION = dict(
    obs_type='point_cloud',
    act_type='eepose_convert',
    rotation_representation='rotation_10D',
    encoder_type='pointnet_rpmg',
    scale_targets=True,
    lambda_pos=1.0,
    lambda_rot=1.0,
    rpmg_lambda=0.01, # as used in the RPMG paper
)

# Direct Vector (MSE) with INTRINSIC 6D rotations, converted to rotation matrices for losses.
# Using RPMG backprop changes
DIRECT_VECTOR_INTRINSIC_6D_ROTATION = dict(
    obs_type='point_cloud',
    act_type='eepose_convert',
    rotation_representation='intrinsic_rotation_6D',
    encoder_type='pointnet_rpmg',
    scale_targets=True,
    lambda_pos=1.0,
    lambda_rot=1.0,
    rpmg_lambda=0.01, # as used in the RPMG paper
)

# Direct Vector (MSE) with INTRINSIC 6D rotations, converted to rotation matrices for losses.
# Using tau_gt
# Using RPMG backprop changes
DIRECT_VECTOR_INTRINSIC_6D_ROTATION_TAUGT = dict(
    obs_type='point_cloud',
    act_type='eepose_convert',
    rotation_representation='intrinsic_rotation_6D',
    encoder_type='pointnet_rpmg_taugt',
    scale_targets=True,
    lambda_pos=1.0,
    lambda_rot=1.0,
    rpmg_lambda=0.01, # as used in the RPMG paper
)

# Direct Vector (MSE) with 6D rotations, converted to rotation matrices for losses.
# Only using RPMG forward, not RPMG backward -- should probably not use this.
# Equivalent to 6D ablation in the RPMG paper
# Note the new encoder type here.
DIRECT_VECTOR_6D_ROTATION_FORWARD_ONLY = dict(
    obs_type='point_cloud',
    act_type='eepose_convert',
    rotation_representation='rotation_6D',
    encoder_type='pointnet_rpmg_forward',
    scale_targets=True,
    lambda_pos=1.0,
    lambda_rot=1.0,
)

# 6D rotation DIRECTLY supervised, no RPMG
DIRECT_VECTOR_6D_ROTATION_DIRECT = dict(
    obs_type='point_cloud',
    act_type='eepose_convert',
    rotation_representation='no_rpmg_6D',
    encoder_type='pointnet',
    scale_targets=True,
    lambda_pos=1.0,
    lambda_rot=1.0,
)
