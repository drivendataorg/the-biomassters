import torch
import torch.nn as nn
import timm
from segmentation_models_pytorch.base import initialization as init
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder


class AttentionPooling(nn.Module):
    def __init__(self, embedding_size, hid=None):
        super().__init__()

        hid = embedding_size if hid is None else hid

        self.attn = nn.Sequential(
            nn.Linear(embedding_size, hid),
            nn.LayerNorm(hid),
            nn.GELU(),
            nn.Linear(hid, 1)
        )

    def forward(self, x, bs, mask=None):
        # x: [B*T, D, H, W]
        # mask: [B, T]
        _, d, h, w = x.size()
        x = x.view(bs, -1, d, h, w)
        x = x.permute(0, 1, 3, 4, 2)  # [bs, t, h, w, d]

        # x:    [B, T, *, D]
        attn_logits = self.attn(x)  # [B, T, *, 1]
        if mask is not None:
            attn_logits[mask] = -torch.inf

        attn_weights = attn_logits.softmax(dim=1)  # [B, T, *, 1]

        x = attn_weights * x  # [B, T, *, D]
        x = x.sum(dim=1)   # [B, *, D]

        x = x.permute(0, 3, 1, 2)     # [bs, d, h, w]

        return x, attn_weights


class TimmEncoder(nn.Module):
    def __init__(self, cfg, output_stride=32):
        super().__init__()

        depth = len(cfg.out_indices)
        self.model = timm.create_model(
            cfg.backbone,
            in_chans=cfg.in_channels,
            pretrained=True,
            num_classes=0,
            features_only=True,
            output_stride=output_stride if output_stride != 32 else None,
            out_indices=cfg.out_indices,
        )
        self._in_channels = cfg.in_channels
        self._out_channels = [
            cfg.in_channels,
        ] + self.model.feature_info.channels()
        self._depth = depth
        self._output_stride = output_stride  # 32

    def forward(self, x):
        features = self.model(x)

        return features

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)


class UnetVFLOW(nn.Module):
    def __init__(self, args, decoder_use_batchnorm: bool = True):
        super().__init__()

        encoder_name = args.backbone

        self.encoder = TimmEncoder(args)

        encoder_depth = len(self.encoder.out_channels) - 1

        self.attn = nn.ModuleList(
            [
                AttentionPooling(i)
                for i in self.encoder.out_channels[1:]
            ]
        )

        decoder_channels = args.dec_channels[:encoder_depth]
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=args.dec_attn_type,
        )
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], args.n_classes, kernel_size=3, padding=1)

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, x, mask):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        bs, _, d, h, w = x.size()
        x = x.view(-1, d, h, w)
        x = x.to(memory_format=torch.channels_last)
        features = self.encoder(x)

        features = [None] + [
            attn(f, bs, mask)[0]
            for f, attn in zip(features, self.attn)
        ]

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        return masks
