import torch
import torch.nn as nn
import torchvision
import config

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1,
            init_identity=False):
    """3x3 convolution with padding"""
    conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)
    if init_identity:
        """ init as identity kernel, not working yet
        0 0 0
        0 1 0
        0 0 0
        """
        identity_weight = conv.weight.new_zeros(3, 3)
        identity_weight[1, 1] = 1. / in_planes
        identity_weight = identity_weight.view(
            1, 1, 3, 3).expand(conv.weight.size())
        with torch.no_grad():
            conv.weight = nn.Parameter(identity_weight)
    return conv


img_model = torchvision.models.resnet50(pretrained=True)

for param in img_model.parameters():
    param.requires_grad = False

class GridFeatBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_back = img_model
        self.feature = nn.Sequential(*list(self.img_back.children())[:-2])

        self.grid_encoder = nn.Sequential(
            conv3x3(config.backbone_channel_in_size, config.hidden_size),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        bsz, c, h, w = x.shape

        res5_features = self.feature(x)
        # print(res5_features.shape)
        grid = self.grid_encoder(res5_features)

        #grid = self.grid_encoder(grid_feat_outputs)  # (B * n_frm, C, H, W)
        new_c, new_h, new_w = grid.shape[-3:]
        n_frms = 1
        # if n_frms != 0:
        grid = grid.view(bsz, n_frms, new_c, new_h, new_w)  # (B, n_frm, C, H, W)

        grid = grid.permute(0, 1, 3, 4, 2)  # (B, n_frm=3, H, W, C)

        return grid
    
class VisualInputEmbedding(nn.Module):
    """
    Takes input of both image and video (multi-frame)
    """
    def __init__(self, config):
        super(VisualInputEmbedding, self).__init__()
        self.config = config

        # sequence embedding
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.row_position_embeddings = nn.Embedding(
            config.max_grid_row_position_embeddings,
            config.hidden_size)
        self.col_position_embeddings = nn.Embedding(
            config.max_grid_col_position_embeddings,
            config.hidden_size)
        self.token_type_embeddings = nn.Embedding(1, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, grid):
        """
        Args:
            grid: (B, n_frm, H, W, C), note that #frm can be 1

        Returns:

        """
#         print(grid.shape)
        bsz, _, _, _, hsz = grid.shape

        # temporal mean pooling
        grid = grid.mean(1)  # (B, H, W, d)
        grid = self.add_2d_positional_embeddings(grid)  # (B, H, W, d)
        # image token sequence
        visual_tokens = grid.view(bsz, -1, hsz)  # (B, H*W, d)

 
        visual_tokens_shape = visual_tokens.shape[:-1]  # (B, H*W)
        device = visual_tokens.device

        # image token type embeddings.
        token_type_ids = torch.zeros(
            visual_tokens_shape, dtype=torch.long, device=device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # embeddings = visual_tokens + position_embeddings + token_type_embeddings
        embeddings = visual_tokens + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings  # (B, H*W, d)

    def add_temporal_postion_embeddings(self, grid):
        """
        Args:
            grid: (B, n_frms, H, W, d)

        Returns:
            (B, n_frms, H, W, d)
        """
        n_frms, height, width, hsz = grid.shape[-4:]

        # add row-wise position embeddings
        temporal_position_ids = torch.arange(
            n_frms, dtype=torch.long, device=grid.device)  # (n_frms, )
        t_position_embeddings = self.temporal_position_embeddings(
            temporal_position_ids)  # (n_frms, d)
        new_shape = (1, n_frms, 1, 1, hsz)  # (1, n_frms, 1, 1, d)
        grid = grid + t_position_embeddings.view(
            *new_shape)  # broadcast automatically

        return grid

    def add_2d_positional_embeddings(self, grid):
        """
        Args:
            grid: (B, *, H, W, d)

        Returns:
            (B, *, H, W, d)
        """
        height, width, hsz = grid.shape[-3:]

        # add row-wise position embeddings
        row_position_ids = torch.arange(
            height, dtype=torch.long, device=grid.device)  # (H, )
        row_position_embeddings = self.row_position_embeddings(
            row_position_ids)  # (H, d)
        row_shape = (1, ) * (len(grid.shape) - 3) + (
            height, 1, hsz)  # (1, *1, H, 1, d)
        grid = grid + row_position_embeddings.view(
            *row_shape)  # broadcast automatically

        # add column-wise position embeddings
        col_position_ids = torch.arange(
            width, dtype=torch.long, device=grid.device)  # (W, )
        col_position_embeddings = self.col_position_embeddings(
            col_position_ids)  # (W, d)
        col_shape = (1, ) * (len(grid.shape) - 3) + (
            1, width, hsz)  # (1, *1, 1, W, d)
        grid = grid + col_position_embeddings.view(
            *col_shape)  # broadcast automatically
        return grid
    
if __name__ == "__main__":
    # Test image festure extractor
    test_input = torch.randn((2,3,224,224))
    grid_feat = GridFeatBackbone(img_model)
    test_output = grid_feat(test_input)
    print(test_output.shape) # torch.Size([2, 1, 3, 3, 768])
    
    # Test Visual Embedding Module
    grid_feat = VisualInputEmbedding(config)
    test_output = grid_feat(test_output)
    print(test_output.shape) # torch.Size([2, 9, 768])


