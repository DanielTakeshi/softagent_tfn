from xml.dom.minidom import Identified
import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


# NOTE(daniel): get these by checking shape after going through conv layers
# for 128 x 128 inputs
OUT_DIM_128 = {4: 57}
# for 100 x 100 inputs
OUT_DIM_100 = {4: 43}
# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations.

    Update 08/24/2022: support depth_segm as observation type.
    """

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32,
            output_logits=False, depth_segm=False):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.depth_segm = depth_segm
        print(f'Making Image CNN. Using a segm? {self.depth_segm}, obs: {self.obs_shape}')

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        # Handle compatibility with output of convs and the FC layers portion.
        if obs_shape[-1] == 64:
            out_dim = OUT_DIM_64[num_layers]
        elif obs_shape[-1] == 84:
            out_dim = OUT_DIM[num_layers]
        elif obs_shape[-1] == 100:
            out_dim = OUT_DIM_100[num_layers]
        elif obs_shape[-1] == 128:
            out_dim = OUT_DIM_128[num_layers]
        else:
            raise NotImplementedError
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        """Forward pass through CNN.

        NOTE(daniel): since we support RGBD input now, only divide by 255 for the
        image based components. With just RGBD we can do: obs = obs / 255.
        """
        if self.depth_segm:
            # Special case, we actually don't do anything as the image should already
            # be binary 0-1 for the masks, and between [0,x] for depth, where x ~ 1.
            # E.g., for MMOneSphere, image channels are: depth, tool mask, item mask.
            # Update: also doing this for rgb_segm_masks, rgbd_segm_masks, as we assume
            # the input will be in the correct range.
            pass
        else:
            obs[:,:3,:,:] /= 255.  # (batch, channels, H, W)
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()
        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            out = h_norm
        else:
            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, output_logits,
            depth_segm=False):
        super().__init__()
        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


class IdentityPNEncoder(nn.Module):
    """NOTE(daniel) for PointNet++, to allow for len(obs_shape) = 2.
    But this will still be an identity function. Just to make code consistent.
    """
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, output_logits,
            depth_segm=False):
        super().__init__()
        assert len(obs_shape) == 2
        self.feature_dim = obs_shape

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


# NOTE(daniel): this shoudl be fixed at some point, after deadline. :)
_AVAILABLE_ENCODERS = {'pixel': PixelEncoder,
                       'segm': PixelEncoder,
                       'mlp': IdentityEncoder,
                       'state_predictor_then_mlp': IdentityEncoder,
                       'identity': IdentityEncoder,
                       'pointnet': IdentityPNEncoder,
                       'pointnet_rpmg': IdentityPNEncoder,
                       'pointnet_avg': IdentityPNEncoder,
                       'pointnet_svd': IdentityPNEncoder,
                       'pointnet_svd_centered': IdentityPNEncoder,
                       'pointnet_svd_pointwise': IdentityPNEncoder,
                       'pointnet_svd_pointwise_6d_flow': IdentityPNEncoder,
                       'pointnet_dense_tf_3D_MSE': IdentityPNEncoder,
                       'pointnet_dense_tf_6D_MSE': IdentityPNEncoder,
                       'pointnet_dense_tf_6D_pointwise': IdentityPNEncoder,
                       'pointnet_classif_6D_pointwise': IdentityPNEncoder,
                       'pointnet_rpmg_pointwise': IdentityPNEncoder,
                       'pointnet_rpmg_taugt': IdentityPNEncoder,
                       'pointnet_svd_6d_flow_mse_loss': IdentityPNEncoder,
                       'pointnet_svd_pointwise_PW_bef_SVD': IdentityPNEncoder,
}


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False,
    depth_segm=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits, depth_segm=depth_segm
    )
