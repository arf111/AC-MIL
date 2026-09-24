import torch
import torch.nn as nn
from torch.autograd import Function
from monai.networks.nets.resnet import _resnet, ResNetBlock, get_inplanes


class ResNet3D(nn.Module):
    """
    Wrapper for MONAI's ResNet to extract features (no FC/classification layer).
    Despite the name, `spatial_dims` controls whether it runs as a 2D or 3D
    convolutional network; this repo uses it in 2D mode over image patches.
    """
    def __init__(self, block, layers, block_inplanes, spatial_dims, n_input_channels, shortcut_type="B", bias_downsample=True,
                 act='prelu'):
        super().__init__()
        self.backbone = _resnet(
            arch="resnet_custom",
            block=block,
            layers=layers,
            block_inplanes=block_inplanes,
            spatial_dims=spatial_dims,
            n_input_channels=n_input_channels,
            shortcut_type=shortcut_type,
            feed_forward=False,
            bias_downsample=bias_downsample,
            act=act,
            progress=True,
            pretrained=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act(x)
        if not self.backbone.no_max_pool:
            x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)  # 64
        x = self.backbone.layer2(x)  # 128
        x = self.backbone.layer3(x)  # 256
        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    """
    @staticmethod
    def forward(ctx, x, alpha):
        """
        Forward pass: identity function.

        Args:
            ctx (torch.autograd.function.Context): Context object to save variables for backward pass.
            input (torch.Tensor): The input tensor.
            alpha (float): The hyperparameter to scale the reversed gradient.

        Returns:
            torch.Tensor: The input tensor (unchanged).
        """
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: reverse and scale the gradient.

        Args:
            ctx (torch.autograd.function.Context): Context object with saved variables.
            grad_output (torch.Tensor): The gradient from the subsequent layer.

        Returns:
            torch.Tensor: The reversed and scaled gradient.
            None: Gradient for the alpha argument (not needed).
        """
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer that reverses gradients during backpropagation.

    Args:
        alpha: Scaling factor for reversed gradients (default: 1.0)
    """
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        """
        Apply the gradient reversal.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor with gradient reversal applied.
        """
        return GradientReversalFunction.apply(x, self.alpha)

    def set_alpha(self, alpha):
        """Allows dynamic adjustment of alpha during training"""
        self.alpha = alpha

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha})"


class AttentionForHAMILQA(nn.Module):
    def __init__(self, L, D, K=1, temperature=1.0):
        super(AttentionForHAMILQA, self).__init__()
        self.attention_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_w = nn.Linear(D, K)
        self.temperature = temperature

    def forward(self, x):
        A = self.attention_w(self.attention_V(x))
        return torch.softmax(A / self.temperature, dim=1)

class AttentionMILPseudoBagTier1Unsup(nn.Module):
    def __init__(
        self,
        encoder_name,
        n_input_channels=1,
        num_classes=4,
        concept_dim=64,
        spatial_dims=None,
    ):
        super(AttentionMILPseudoBagTier1Unsup, self).__init__()

        # --- A. Shared Encoder ---
        self.encoder_2d = None
        self.feature_dim = 64
        self.concept_dim = concept_dim
        self.num_classes = num_classes

        if encoder_name == 'resnet':
            # ResNet10, not pretrained (trained from scratch).
            self.encoder_2d = ResNet3D(
                block=ResNetBlock,
                layers=[1, 1, 1, 1],
                block_inplanes=get_inplanes(),
                spatial_dims=spatial_dims,
                n_input_channels=n_input_channels,
                shortcut_type="B",
                bias_downsample=False,
            )

            self.feature_dim = 256
        else:
            raise NotImplementedError('Only ResNet is implemented')

        self.dropout = nn.Dropout(0.5)

        # --- B. The Fork (Concept Projections) ---
        def create_projection_head():
            return nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim // 2),
                nn.PReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.feature_dim // 2, self.concept_dim),
                nn.LayerNorm(self.concept_dim)
            )

        self.proj_sharpness = create_projection_head()
        self.proj_nulling = create_projection_head()
        self.proj_aorta = create_projection_head()
        self.proj_unsup = create_projection_head()

        # --- C. The Route (Concept-Specific Attention) ---
        self.att_sharpness = AttentionForHAMILQA(L=self.concept_dim, D=32, K=1)
        self.att_nulling = AttentionForHAMILQA(L=self.concept_dim, D=32, K=1)
        self.att_aorta = AttentionForHAMILQA(L=self.concept_dim, D=32, K=1)
        self.att_unsup = AttentionForHAMILQA(L=self.concept_dim, D=32, K=1)

        # --- D. Concept Classifiers ---

        self.clf_sharpness = nn.Linear(self.concept_dim, num_classes-1)
        self.clf_nulling = nn.Linear(self.concept_dim, num_classes-1)
        self.clf_aorta = nn.Linear(self.concept_dim, num_classes-1)

        # --- E. Adversarial Branch (GRL) ---
        self.grl = GradientReversalLayer(alpha=1.0)

        def create_adversary():
            return nn.Sequential(
                nn.Linear(self.concept_dim, self.concept_dim // 2),
                nn.LayerNorm(self.concept_dim // 2),
                nn.PReLU(),
                nn.Dropout(0.7),

                nn.Linear(self.concept_dim // 2, self.concept_dim // 4),
                nn.LayerNorm(self.concept_dim // 4),
                nn.PReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.concept_dim // 4, num_classes-1)
            )

        self.adversaries = nn.ModuleDict({
            'sharpness': create_adversary(),
            'nulling': create_adversary(),
            'aorta': create_adversary()
        })

        self.initialize_parameters()

    def initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def calculate_spatial_diversity_loss(self, att_maps):
        """
        Calculates Spatial Attention Diversity (SAD) loss.
        Penalizes attention maps from different concepts looking at the same patches.

        Args:
            att_maps: Dict of {concept_name: Tensor(Batch*Bags, Patches, 1)}
        """
        loss = 0.0

        # 1. Flatten maps to (Batch*Bags, Patches)
        # We assume input is (B, N, 1), so we squeeze the last dim
        flat_maps = {k: v.view(v.size(0), -1) for k, v in att_maps.items()}

        # 2. L2-Normalize each map to unit length
        # This ensures the dot product represents Cosine Similarity (0 to 1)
        # regardless of how "spiky" or "flat" the attention distribution is.
        norm_maps = {}
        for k, v in flat_maps.items():
            norm = torch.norm(v, p=2, dim=1, keepdim=True) + 1e-8
            norm_maps[k] = v / norm

        # 3. Define pairs that MUST be spatially distinct
        # We explicitly penalize overlap between Sharpness and the anatomical concepts
        pairs = [
            ("aorta", "nulling"),     # These are distinct tissues; strictly enforce separation.
            ('aorta', 'unsup'),      # Unsupervised concept should not focus on the same patches as aorta.
            ('nulling', 'unsup')     # Unsupervised concept should not focus on the same patches as nulling.
        ]

        count = 0
        for (c1, c2) in pairs:
            if c1 in norm_maps and c2 in norm_maps:
                # Dot product: sum(a * b)
                # If maps are disjoint, overlap ~ 0. If identical, overlap ~ 1.
                overlap = torch.sum(norm_maps[c1] * norm_maps[c2], dim=1)
                loss += torch.mean(overlap)
                count += 1

        return loss / max(count, 1)

    def forward(self, x):
        # -----------------------------------------------
        # 1. Reshape & Encode
        # -----------------------------------------------
        if len(x.shape) == 4:
            x = x.unsqueeze(0)

        bs, n_bags, n_patches, c, h, w = x.shape
        x_flat = x.reshape(bs * n_bags * n_patches, c, h, w)

        h = self.encoder_2d(x_flat)
        if isinstance(h, list):
            h = h[-1]
        h = h.reshape(bs * n_bags, n_patches, -1)

        h = self.dropout(h)

        # -----------------------------------------------
        # 2. The Fork (Projection)
        # -----------------------------------------------
        h_sharp = self.proj_sharpness(h)
        h_null = self.proj_nulling(h)
        h_aorta = self.proj_aorta(h)
        h_unsup = self.proj_unsup(h)

        # -----------------------------------------------
        # 4. The Route (Attention & Aggregation)
        # -----------------------------------------------
        att_sharp = self.att_sharpness(h_sharp)
        att_null = self.att_nulling(h_null)
        att_aorta = self.att_aorta(h_aorta)
        att_unsup = self.att_unsup(h_unsup)

        # Dictionary to hold the final maps used for Z
        att_maps_for_z = {
            "sharpness": att_sharp,
            "nulling": att_null,
            "aorta": att_aorta,
            "unsup": att_unsup
        }

        # Weighted Sum: Z = sum(alpha * h)
        z_sharp = torch.sum(att_maps_for_z["sharpness"] * h_sharp, dim=1)
        z_null = torch.sum(att_maps_for_z["nulling"] * h_null, dim=1)
        z_aorta = torch.sum(att_maps_for_z["aorta"] * h_aorta, dim=1)
        z_unsup = torch.sum(att_maps_for_z["unsup"] * h_unsup, dim=1)

        # Now the adversary predicts from the Bag-level vector (Shape: Batch*Bags, 64)
        z_unsup_grl = self.grl(z_unsup)

        adv_logits_sharp = self.adversaries['sharpness'](z_unsup_grl)
        adv_logits_null = self.adversaries['nulling'](z_unsup_grl)
        adv_logits_aorta = self.adversaries['aorta'](z_unsup_grl)
        # -----------------------------------------------
        # 5. Concept Supervision
        # -----------------------------------------------

        logits_sharp = self.clf_sharpness(z_sharp)
        logits_null = self.clf_nulling(z_null)
        logits_aorta = self.clf_aorta(z_aorta)

        # Collect maps into a clean dictionary
        current_att_maps = {
            "nulling": att_null,
            "aorta": att_aorta,
            "unsup": att_unsup
        }
        loss_diversity = self.calculate_spatial_diversity_loss(current_att_maps)

        # -----------------------------------------------
        # 6. Fusion
        # -----------------------------------------------
        z_unsup_fused = z_unsup
        v_fused = torch.cat([z_sharp.detach(),
                             z_null.detach(),
                             z_aorta.detach(),
                             z_unsup_fused], dim=1)
        v_fused = v_fused.view(bs, n_bags, -1)

        results = {
            "fused_features": v_fused,
            "concept_logits": {
                "sharpness": logits_sharp,
                "nulling": logits_null,
                "aorta": logits_aorta
            },
            "adv_logits": {
                "sharpness": adv_logits_sharp,
                "nulling": adv_logits_null,
                "aorta": adv_logits_aorta
            },
            "aux_loss": loss_diversity, # Always 0.0
            "attention_maps": {
                "sharpness": att_maps_for_z["sharpness"],
                "nulling": att_maps_for_z["nulling"],
                "unsup": att_maps_for_z["unsup"],
                "aorta": att_maps_for_z["aorta"]
            }
        }

        return results

class AttentionMILPseudoBagTier2Unsup(nn.Module):
    def __init__(self, num_classes=4, fused_dim=256):
        super(AttentionMILPseudoBagTier2Unsup, self).__init__()

        # Input is the fused vector (4 * 64 = 256)
        self.L = fused_dim
        self.D = 64
        self.K = 1

        self.bag_attention = AttentionForHAMILQA(L=self.L, D=self.D, K=self.K)

        # Final QA Classifier
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(self.L, num_classes-1)

        self.initialize_parameters()

    def initialize_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        # x shape: (Batch, Bags, Fused_Dim=256)

        # 1. Attention over Slices (Bags)
        # alpha: (Batch, Bags, 1)
        att_weights = self.bag_attention(x)

        # 2. Aggregate Slices -> Volume Vector
        # Z_vol = sum(alpha * x) -> (Batch, 256)
        z_vol = torch.sum(att_weights * x, dim=1)

        # 3. Final Prediction
        logits = self.classifier(z_vol)

        return logits, att_weights
