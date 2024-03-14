# import kornia
import torch
from torchvision import models, transforms
import torch.nn.functional as F


###### estimate linear color transform
def match_colors_for_image_set(content_images, style_img):
    content_sub = content_images.view(-1, 3)
    style_sub = style_img.view(-1, 3).to(content_sub.device)

    mu_c = content_sub.mean(0, keepdim=True)
    mu_s = style_sub.mean(0, keepdim=True)

    cov_c = torch.matmul((content_sub - mu_c).transpose(1, 0), content_sub - mu_c) / float(content_sub.size(0))
    cov_s = torch.matmul((style_sub - mu_s).transpose(1, 0), style_sub - mu_s) / float(style_sub.size(0))

    u_c, sig_c, _ = torch.svd(cov_c)
    u_s, sig_s, _ = torch.svd(cov_s)

    u_c_i = u_c.transpose(1, 0)
    u_s_i = u_s.transpose(1, 0)

    scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
    scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))

    tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
    tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T

    content_sub = content_sub @ tmp_mat.T + tmp_vec.view(1, 3)
    content_sub = content_sub.contiguous().clamp_(0.0, 1.0)

    tf = torch.eye(4).float().to(tmp_mat.device)
    tf[:3, :3] = tmp_mat
    tf[:3, 3:4] = tmp_vec.T
    return content_sub, tf


class VGG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg16(pretrained=True).eval()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_feats(self, x, layers=[], supress_assert=True):
        # Layer indexes:
        # Conv1_*: 1,3
        # Conv2_*: 6,8
        # Conv3_*: 11, 13, 15
        # Conv4_*: 18, 20, 22
        # Conv5_*: 25, 27, 29

        if not supress_assert:
            assert x.min() >= 0.0 and x.max() <= 1.0, "input is expected to be an image scaled between 0 and 1"

        x = self.normalize(x)
        final_ix = max(layers)
        outputs = []

        for ix, layer in enumerate(self.vgg.features):
            x = layer(x)
            if ix in layers:
                outputs.append(x)

            if ix == final_ix:
                break

        return outputs


def cos_distance(a, b, center=True):
    """a: [b, c, hw],
    b: [b, c, h2w2]
    """
    # """cosine distance
    if center:
        a = a - a.mean(2, keepdims=True)
        b = b - b.mean(2, keepdims=True)

    a_norm = ((a * a).sum(1, keepdims=True) + 1e-8).sqrt()
    b_norm = ((b * b).sum(1, keepdims=True) + 1e-8).sqrt()

    a = a / (a_norm + 1e-8)
    b = b / (b_norm + 1e-8)

    d_mat = 1.0 - torch.matmul(a.transpose(2, 1), b)
    # """"

    """
    a_norm_sq = (a * a).sum(1).unsqueeze(2)
    b_norm_sq = (b * b).sum(1).unsqueeze(1)

    d_mat = a_norm_sq + b_norm_sq - 2.0 * torch.matmul(a.transpose(2, 1), b)
    """
    return d_mat


def cos_loss(a, b):
    # """cosine loss
    a_norm = (a * a).sum(1, keepdims=True).sqrt()
    b_norm = (b * b).sum(1, keepdims=True).sqrt()
    a_tmp = a / (a_norm + 1e-8)
    b_tmp = b / (b_norm + 1e-8)
    cossim = (a_tmp * b_tmp).sum(1)
    cos_d = 1.0 - cossim
    return cos_d.mean()
    # """

    # return ((a - b) ** 2).mean()


def feat_replace(a, b):
    n, c, h, w = a.size()
    n2, c, h2, w2 = b.size()

    assert (n == 1) and (n2 == 1)

    a_flat = a.view(n, c, -1)
    b_flat = b.view(n2, c, -1)
    b_ref = b_flat.clone()

    z_new = []

    # Loop is slow but distance matrix requires a lot of memory
    for i in range(n):
        z_dist = cos_distance(a_flat[i : i + 1], b_flat[i : i + 1])

        z_best = torch.argmin(z_dist, 2)
        del z_dist

        z_best = z_best.unsqueeze(1).repeat(1, c, 1)
        feat = torch.gather(b_ref, 2, z_best)

        z_new.append(feat)

    z_new = torch.cat(z_new, 0)
    z_new = z_new.view(n, c, h, w)
    return z_new


def guided_feat_replace(a, b, trgt):
    n, c, h, w = a.size()
    n2, c2, h2, w2 = b.size()
    n3, c3, h3, w3 = trgt.size()
    assert (n == 1) and (n2 == 1) and (c == c2) and (n3 == 1) and (h2 == h3) and (w2 == w3)

    a_flat = a.view(n, c, -1)
    b_flat = b.view(n2, c2, -1)
    trgt = trgt.view(n3, c3, -1)

    z_new = []

    # Loop is slow but distance matrix requires a lot of memory
    for i in range(n):
        z_dist = cos_distance(a_flat[i : i + 1], b_flat[i : i + 1])

        z_best = torch.argmin(z_dist, 2)
        del z_dist

        z_best = z_best.unsqueeze(1).repeat(1, c3, 1)
        feat = torch.gather(trgt, 2, z_best)

        z_new.append(feat)

    z_new = torch.cat(z_new, 0)
    z_new = z_new.view(n, c3, h, w)
    return z_new


def nn_loss(outputs, styles, vgg, blocks=[2]):

    blocks.sort()
    block_indexes = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]
    total_loss = 0.0

    all_layers = []
    for block in blocks:
        all_layers += block_indexes[block]

    x_feats_all = vgg.get_feats(outputs, all_layers)
    with torch.no_grad():
        s_feats_all = vgg.get_feats(styles, all_layers)

    ix_map = {}
    for a, b in enumerate(all_layers):
        ix_map[b] = a

    for block in blocks:
        layers = block_indexes[block]
        x_feats = torch.cat([x_feats_all[ix_map[ix]] for ix in layers], 1)
        s_feats = torch.cat([s_feats_all[ix_map[ix]] for ix in layers], 1)

        target_feats = feat_replace(x_feats, s_feats)
        total_loss += cos_loss(x_feats, target_feats)

    return total_loss


def gram_matrix(feature_maps, center=False):
    """
    feature_maps: b, c, h, w
    gram_matrix: b, c, c
    """
    b, c, h, w = feature_maps.size()
    features = feature_maps.view(b, c, h * w)
    if center:
        features = features - features.mean(dim=-1, keepdims=True)
    G = torch.bmm(features, torch.transpose(features, 1, 2))
    return G


class NNLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg = VGG().to(device)

    def forward(
        self,
        outputs,
        styles,
        blocks=[
            2,
        ],
        loss_names=["nn_loss"],  # can also include 'gram_loss', 'content_loss'
        contents=None,
    ):
        blocks.sort()
        block_indexes = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]

        all_layers = []
        for block in blocks:
            all_layers += block_indexes[block]

        x_feats_all = self.vgg.get_feats(outputs, all_layers)
        with torch.no_grad():
            s_feats_all = self.vgg.get_feats(styles, all_layers)
            if "content_loss" in loss_names:
                content_feats_all = self.vgg.get_feats(contents, all_layers)

        ix_map = {}
        for a, b in enumerate(all_layers):
            ix_map[b] = a

        nn_loss = 0.0
        gram_loss = 0.0
        content_loss = 0.0
        for block in blocks:
            layers = block_indexes[block]
            x_feats = torch.cat([x_feats_all[ix_map[ix]] for ix in layers], 1)
            s_feats = torch.cat([s_feats_all[ix_map[ix]] for ix in layers], 1)

            if "nn_loss" in loss_names:
                target_feats = feat_replace(x_feats, s_feats)
                nn_loss += cos_loss(x_feats, target_feats)

            if "gram_loss" in loss_names:
                gram_loss += torch.mean((gram_matrix(x_feats) - gram_matrix(s_feats)) ** 2)

            if "content_loss" in loss_names:
                content_feats = torch.cat([content_feats_all[ix_map[ix]] for ix in layers], 1)
                content_loss += torch.mean((content_feats - x_feats) ** 2)

        return nn_loss, gram_loss, content_loss

    def get_style_nn(
        self,
        outputs,
        styles,
        blocks=[
            2,
        ],
    ):
        blocks.sort()
        block_indexes = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]

        all_layers = []
        for block in blocks:
            all_layers += block_indexes[block]

        x_feats_all = self.vgg.get_feats(outputs, all_layers)
        with torch.no_grad():
            s_feats_all = self.vgg.get_feats(styles, all_layers)

        ix_map = {}
        for a, b in enumerate(all_layers):
            ix_map[b] = a

        trgt_feats = []
        for block in blocks:
            layers = block_indexes[block]
            x_feats = torch.cat([x_feats_all[ix_map[ix]] for ix in layers], 1)
            s_feats = torch.cat([s_feats_all[ix_map[ix]] for ix in layers], 1)

            _, _, h, w = s_feats.size()
            styles_resample = F.interpolate(styles, (h, w), mode="bilinear")

            feats = guided_feat_replace(x_feats, s_feats, styles_resample)
            trgt_feats.append(feats)

        return trgt_feats


if __name__ == "__main__":
    vgg = VGG().cuda()
    fake_output = torch.rand(1, 3, 256, 256).cuda()
    fake_style = torch.rand(1, 3, 256, 256).cuda()

    # Blocks are 0-indexed (Conv1 = 0, Conv2 = 1, etc)
    loss = nn_loss(fake_output, fake_style, vgg, blocks=[2])
    print(loss)
