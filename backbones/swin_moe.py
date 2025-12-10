import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from torch.nn import Conv2d, Dropout

try:
    import os, sys
    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse
except Exception as e:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed.")



class BasicMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TopKRouter(nn.Module):
    def __init__(self, in_features, num_experts, k=2, 
                 router_z_loss_coef=1e-3, load_balance_loss_coef=1e-2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.gate = nn.Linear(in_features, num_experts, bias=False)
        trunc_normal_(self.gate.weight, std=0.02)
        self.router_z_loss_coef = router_z_loss_coef
        self.load_balance_loss_coef = load_balance_loss_coef

    def forward(self, x):
        logits = self.gate(x)
        
        p = torch.softmax(logits, dim=1) 
        router_z_loss = self.router_z_loss_coef * torch.mean(logits**2)
        
        topk_values, topk_indices = torch.topk(logits, self.k, dim=1)

        dispatch_mask = torch.zeros_like(logits)
        row_indices = torch.arange(x.size(0), device=x.device).unsqueeze(-1)
        dispatch_mask[row_indices, topk_indices] = 1.0
        
        T = x.size(0)
        importance = p.sum(dim=0)     
        load = dispatch_mask.sum(dim=0) 
        load_balance_loss = self.load_balance_loss_coef * self.num_experts * torch.sum(importance * load) / (T * T)
        
        return dispatch_mask, topk_values, topk_indices, router_z_loss, load_balance_loss



class Expert(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MoEMlp(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_features=None, 
                 out_features=None, 
                 act_layer=nn.GELU, 
                 drop=0., 
                 num_experts=4,
                 k=2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.num_experts = num_experts
        self.k = k

        self.experts = nn.ModuleList([
            Expert(in_features, hidden_features, out_features, act_layer, drop)
            for _ in range(num_experts)
        ])

        self.router = TopKRouter(in_features, num_experts, k=k)

    def forward(self, x):
        original_shape = x.shape
        if len(x.shape) == 3:
            B, N, C = x.shape
            x = x.view(B * N, C)
        
        dispatch_mask, topk_values, topk_indices, router_z_loss, load_balance_loss = self.router(x)
        aux_loss = router_z_loss + load_balance_loss

        T = x.size(0)
        out_dim = self.experts[0].fc2.out_features
        expert_assignments = dispatch_mask.nonzero(as_tuple=False) 

        if expert_assignments.shape[0] == 0:
            combined_output = x.new_zeros((T, out_dim))
            if len(original_shape) == 3:
                combined_output = combined_output.view(B, N, -1)
            return combined_output, aux_loss

        sorted_assignments = expert_assignments[expert_assignments[:, 1].sort()[1]]
        token_indices_sorted = sorted_assignments[:, 0]  
        expert_indices_sorted = sorted_assignments[:, 1]   

        expert_input_sorted = x[token_indices_sorted]      

        row_offset = self.num_experts * token_indices_sorted  
        sorted_pair_id = row_offset + expert_indices_sorted

        row_ids = torch.arange(T, device=x.device)
        row_ids = row_ids.unsqueeze(1).expand(-1, self.k)
        flat_row_ids    = row_ids.reshape(-1)
        flat_expert_ids = topk_indices.reshape(-1)

        flat_pair_id = flat_row_ids * self.num_experts + flat_expert_ids
        flat_gate_vals = topk_values.reshape(-1)

        gate_lookup = x.new_zeros(T * self.num_experts)  
        gate_lookup[flat_pair_id] = flat_gate_vals

        sorted_gate_vals = gate_lookup[sorted_pair_id]

        unique_experts, counts = torch.unique(expert_indices_sorted, return_counts=True)

        combined_output = x.new_zeros((T, out_dim))
        cur_index = 0
        for i, e in enumerate(unique_experts):
            e = e.item()
            c = counts[i].item()
            range_start = cur_index
            range_end = cur_index + c
            cur_index = range_end

            expert_input_this = expert_input_sorted[range_start:range_end]
            gate_vals_this = sorted_gate_vals[range_start:range_end].unsqueeze(-1)

            expert_output_this = self.experts[e](expert_input_this)
            scaled_output_this = expert_output_this * gate_vals_this

            token_indices_this = token_indices_sorted[range_start:range_end]
            combined_output[token_indices_this] += scaled_output_this
            
        if len(original_shape) == 3:
            B, N, _ = original_shape
            combined_output = combined_output.view(B, N, out_dim)
        
        return combined_output, aux_loss


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0]*self.window_size[1],
               self.window_size[0]*self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn[:, :, :self.window_size[0]*self.window_size[1], :self.window_size[0]*self.window_size[1]] += relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_//nW, nW, self.num_heads, self.window_size[0]*self.window_size[1],
                             self.window_size[0]*self.window_size[1]) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads,
                 window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False,
                 use_moe=False,  
                 num_experts=3,   
                 k=2          
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), 
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_moe:
            self.mlp = MoEMlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                out_features=dim,
                act_layer=act_layer,
                drop=drop,
                num_experts=num_experts,
                k=k
            )
        else:
            self.mlp = BasicMlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                out_features=dim,
                act_layer=act_layer,
                drop=drop
            )

        if self.shift_size > 0:
            H_, W_ = self.input_resolution
            img_mask = torch.zeros((1, H_, W_, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))
                x_windows = window_partition(shifted_x, self.window_size)
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows[:, :25]
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1,2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        
        mlp_output = self.mlp(self.norm2(x))
        if isinstance(mlp_output, tuple):
            mlp_out, aux_loss = mlp_output
        else:
            mlp_out, aux_loss = mlp_output, 0.0
        
        self.aux_loss = aux_loss
        x = x + self.drop_path(mlp_out)
        return x, self.aux_loss



class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, 
                 use_checkpoint=False, fused_window_process=False,
                 deep_prompt_dim=41,
                 use_moe=False,    
                 num_experts=4,  
                 k=2):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=(input_resolution[0], input_resolution[1]),
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop, 
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                fused_window_process=fused_window_process,
                use_moe=use_moe,
                num_experts=num_experts,
                k=k
            )
            for i in range(depth)
        ])

        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x):
        total_aux_loss = 0.0
        for blk in self.blocks:
            x, aux = blk(x)
            total_aux_loss += aux
        if self.downsample is not None:
            x = self.downsample(x)
        return x, total_aux_loss


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}x{W}) doesn't match ({self.img_size[0]}x{self.img_size[1]})"
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x
    
    def flops(self):
        H, W = self.img_size
        Kh, Kw = self.patch_size
        Cin = self.in_chans
        Cout = self.embed_dim
        Hout, Wout = self.patches_resolution

        flops = 2 * Cout * Hout * Wout * Cin * Kh * Kw

        if self.norm is not None:
            flops += Cout * Hout * Wout 

        return flops


class SwinTransformer(nn.Module):
    def __init__(self, img_size=112, patch_size=9, in_chans=3, num_classes=512,
                 embed_dim=128, depths=[8,12,2], num_heads=[4,16,16],
                 window_size=6, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.05,
                 norm_layer=nn.LayerNorm, ape=True, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False,
                 use_moe=False,            
                 num_experts=3,           
                 k=2,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2**(self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            input_resolution = (patches_resolution[0] // (2 ** i_layer),
                                patches_resolution[1] // (2 ** i_layer))
            if input_resolution[0] == 20:
                prompt_dim = 41
            elif input_resolution[0] == 10:
                prompt_dim = 21
            else:
                prompt_dim = 11

            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=input_resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale,
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]): sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                fused_window_process=fused_window_process,
                deep_prompt_dim=prompt_dim,
                use_moe=use_moe,
                num_experts=num_experts,
                k=k
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.reso = kwargs.get('reso', img_size)

        self.feature_layer = nn.Sequential(
            nn.BatchNorm1d(embed_dim * 100),
            nn.Dropout(0.25),
            nn.Linear(embed_dim * 100, 512),
            nn.BatchNorm1d(512)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        total_aux_loss = 0.0
        for layer in self.layers:
            x, aux_loss_layer = layer(x)
            total_aux_loss += aux_loss_layer
        
        x = self.norm(x)
        x = x.reshape(B, -1)
        self.aux_loss = total_aux_loss
        return x

    def forward(self, x):
        features = self.forward_features(x)
        x = self.feature_layer(features)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2**self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


@register_model
def swin_180_topk_moe_loss(**kwargs):
    kwargs['reso'] = 180
    kwargs['use_moe'] = True
    kwargs.setdefault('num_experts', 2)
    kwargs.setdefault('k', 1)
    model = SwinTransformer(
        img_size=180, patch_size=9, in_chans=3, num_classes=512,
        embed_dim=160, depths=[4, 8, 18], num_heads=[8, 16, 16],
        window_size=5,
        **kwargs
    )
    return model

@register_model
def swin_120_topk_moe_loss(**kwargs):
    kwargs['reso'] = 120
    kwargs['use_moe'] = True
    kwargs.setdefault('num_experts', 3)
    kwargs.setdefault('k', 2)
    model = SwinTransformer(
        img_size=120, patch_size=6, in_chans=3, num_classes=512,
        embed_dim=384, depths=[2, 18, 2], num_heads=[8, 16, 16],
        window_size=5,
        **kwargs
    )
    return model


if __name__ == "__main__":
    model = swin_120_topk_moe_loss(k=2, num_experts=3) 
    dummy_input = torch.randn(2, 3, 120, 120)
    local_embeddings = model(dummy_input)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in Swin 120 MoE: {num_params}")
    print("Output shape:", local_embeddings.shape)
    # print("Auxiliary loss:", aux_loss.item())