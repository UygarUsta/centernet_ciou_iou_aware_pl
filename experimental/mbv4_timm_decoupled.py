import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import time # Added import for the main block

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.bn_depthwise = nn.BatchNorm2d(in_channels)
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        # Removed Sequential wrapper as BatchNorm should be applied after depthwise

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_depthwise(x) # Apply BatchNorm after depthwise
        x = self.pointwise(x)
        return x

def normal_init(module, mean=0, std=1, bias=0):
    if isinstance(module, (nn.Conv2d, nn.Linear)): # Check if it's a standard Conv/Linear
        nn.init.normal_(module.weight, mean, std)
        if module.bias is not None:
            nn.init.constant_(module.bias, bias)
    elif isinstance(module, SeparableConv2d): # Handle SeparableConv2d specifically
        # Initialize the pointwise layer, as depthwise often doesn't need special init
        nn.init.normal_(module.pointwise.weight, mean, std)
        if module.pointwise.bias is not None:
            nn.init.constant_(module.pointwise.bias, bias)


class ComplexUpsample(nn.Module):
    def __init__(self, input_dim=128, outpt_dim=128):
        super().__init__()

        self.conv1 = nn.Sequential(SeparableConv2d(input_dim, outpt_dim, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(outpt_dim),
                                   nn.ReLU(inplace=True)
                                   )

        self.conv2 = nn.Sequential(SeparableConv2d(input_dim, outpt_dim, kernel_size=5, stride=1, padding=2, bias=False),
                                   nn.BatchNorm2d(outpt_dim),
                                   nn.ReLU(inplace=True)
                                   )

    def forward(self, inputs):
        # do preprocess

        x = self.conv1(inputs)
        y = self.conv2(inputs)

        z = x + y

        z = nn.functional.interpolate(z, scale_factor=2,mode='bilinear', align_corners=False ) # Added align_corners=False

        return z


class Fpn(nn.Module):
    def __init__(self,input_dims=[24,32,96,320],head_dims=[128,128,128] ):
        super().__init__()

        # Ensure head_dims has enough elements if accessed by index
        assert len(head_dims) >= 3, "head_dims must have at least 3 elements"
        self.output_channels = head_dims[0] # Store the final output channels of FPN

        self.latlayer2=nn.Sequential(SeparableConv2d(input_dims[0],head_dims[0]//2,kernel_size=5,padding=2, bias=False), # Added bias=False for consistency
                                      nn.BatchNorm2d(head_dims[0]//2),
                                      nn.ReLU(inplace=True))


        self.latlayer3=nn.Sequential(SeparableConv2d(input_dims[1],head_dims[1]//2,kernel_size=5,padding=2, bias=False), # Added bias=False
                                      nn.BatchNorm2d(head_dims[1]//2),
                                      nn.ReLU(inplace=True))

        self.latlayer4 = nn.Sequential(SeparableConv2d(input_dims[2], head_dims[2] // 2,kernel_size=5,padding=2, bias=False), # Added bias=False
                                       nn.BatchNorm2d(head_dims[2] // 2),
                                       nn.ReLU(inplace=True))


        # Upsample layers output half the channels needed for concatenation
        self.upsample3=ComplexUpsample(head_dims[1],head_dims[0]//2)
        self.upsample4 =ComplexUpsample(head_dims[2],head_dims[1]//2)
        self.upsample5 = ComplexUpsample(input_dims[3],head_dims[2]//2)


    def forward(self, inputs):
        ## Expected input_dims=[24,32,96,320] for mobilenetv4_conv_small features[-4:]
        c2, c3, c4, c5 = inputs

        c4_lat = self.latlayer4(c4) # Output: head_dims[2]//2
        c3_lat = self.latlayer3(c3) # Output: head_dims[1]//2
        c2_lat = self.latlayer2(c2) # Output: head_dims[0]//2


        upsample_c5=self.upsample5(c5) # Output: head_dims[2]//2

        # Concatenate: (head_dims[2]//2 + head_dims[2]//2) = head_dims[2]
        p4=torch.cat([c4_lat,upsample_c5],dim=1)


        upsample_p4=self.upsample4(p4) # Input: head_dims[2], Output: head_dims[1]//2

        # Concatenate: (head_dims[1]//2 + head_dims[1]//2) = head_dims[1]
        p3=torch.cat([c3_lat,upsample_p4],dim=1)

        upsample_p3 = self.upsample3(p3) # Input: head_dims[1], Output: head_dims[0]//2

        # Concatenate: (head_dims[0]//2 + head_dims[0]//2) = head_dims[0]
        p2 = torch.cat([c2_lat, upsample_p3],dim=1)


        return p2 # Final output has head_dims[0] channels

# --- Modified CenterNetHead ---
class CenterNetHead(nn.Module):
    """
    Decoupled Head for CenterNet tasks.
    Applies a shared convolution, then separate branches for classification and regression.
    """
    def __init__(self, in_channels, nc, feat_channels=128):
        super().__init__()
        # Shared convolution layer
        # self.shared_conv = nn.Sequential(
        #     SeparableConv2d(in_channels, feat_channels, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(feat_channels),
        #     nn.ReLU(inplace=True)
        # )

        # Classification branch
        self.cls_branch = nn.Sequential(
            SeparableConv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True)
        )
        self.cls_pred = SeparableConv2d(feat_channels, nc, kernel_size=3, stride=1, padding=1, bias=True)

        # Regression branch (for wh, offset, iou)
        self.reg_branch = nn.Sequential(
            SeparableConv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True)
        )
        self.wh_pred = SeparableConv2d(feat_channels, 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.offset_pred = SeparableConv2d(feat_channels, 2, kernel_size=3, stride=1, padding=1, bias=True)
        # Using standard Conv2d for IoU head as in original code, applied to reg_branch output
        self.iou_pred = nn.Conv2d(feat_channels, 1, kernel_size=3, stride=1, padding=1, bias=True)

        # Initialize final prediction layers
        normal_init(self.cls_pred, 0, 0.01, -2.19) # Initialize pointwise of SeparableConv2d
        normal_init(self.wh_pred, 0, 0.01, 0)      # Initialize pointwise of SeparableConv2d
        normal_init(self.offset_pred, 0, 0.01, 0)  # Initialize pointwise of SeparableConv2d
        normal_init(self.iou_pred, 0, 0.01, 0)     # Initialize standard Conv2d

    def forward(self, inputs):
        # Shared features
        shared_feat =  inputs #self.shared_conv(inputs)

        # Classification path
        cls_feat = self.cls_branch(shared_feat)
        cls = self.cls_pred(cls_feat).sigmoid_()

        # Regression path
        reg_feat = self.reg_branch(shared_feat)
        wh = self.wh_pred(reg_feat)
        offset = self.offset_pred(reg_feat)
        iou_aware_head = self.iou_pred(reg_feat).sigmoid_().squeeze(1) # Apply sigmoid and squeeze

        return cls, wh, offset, iou_aware_head

# --- End Modified CenterNetHead ---


class CenterNet(nn.Module):
    def __init__(self,nc):
        super().__init__()

        self.nc = nc
        input_dims = []
        ###model structure
        # Use a specific revision or tag if needed for reproducibility
        self.backbone =  timm.create_model(
            'mobilenetv4_conv_small.e1200_r224_in1k',
            pretrained=True,
            features_only=True,
            exportable=True
        )
        feature_info = self.backbone.feature_info
        # Extract channel dimensions for the last 4 feature levels
        # Indices might need adjustment depending on the specific timm model version
        # Typically, features_only=True returns 5 levels for MobileNets
        # We usually skip the first level (stride 2)
        input_dims = [info['num_chs'] for info in feature_info.info[1:]] # Get channels for C2, C3, C4, C5

        # Define FPN head dimensions
        fpn_head_dims = [128, 128, 128] # Example dimensions, adjust as needed
        self.fpn=Fpn(input_dims=input_dims, head_dims=fpn_head_dims)

        # The input channels for the head is the output channels of the FPN
        fpn_output_channels = self.fpn.output_channels
        head_feat_channels = 128 # Intermediate feature channels within the head

        # Instantiate the Decoupled Head
        self.head = CenterNetHead(
            in_channels=fpn_output_channels,
            nc=self.nc,
            feat_channels=head_feat_channels
        )


        self.device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    def forward(self, inputs):
        # Get features from backbone, typically last 4 for FPN
        # Ensure the correct number of feature maps are selected
        fms = self.backbone(inputs)[-4:] # Select last 4 feature maps

        fpn_fm=self.fpn(fms)
        cls, wh, offset, iou_aware_head= self.head(fpn_fm)


        return cls, wh, offset, iou_aware_head

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # import time # Already imported at the top
    model = CenterNet(10) # Example: 10 classes

    ### load your weights if available
    # model.load_state_dict(torch.load('your_weights.pth'))
    model.eval()

    batch_size = 1
    input_height = 320 # Example input size
    input_width = 320
    device = torch.device('cpu')
    model.to(device)

    modelparams = count_parameters(model)
    print(f"Total Model Params: {modelparams:,}")

    dummy_input = torch.randn(batch_size, 3, input_height, input_width).to(device)

    # --- Standard Inference ---
    print("\n--- Standard Model Inference ---")
    with torch.no_grad():
        out = model(dummy_input)
    print("Output shapes (cls, wh, offset, iou):")
    print(f"  cls:    {out[0].shape}")
    print(f"  wh:     {out[1].shape}")
    print(f"  offset: {out[2].shape}")
    print(f"  iou:    {out[3].shape}") # iou_aware_head is squeezed

    # --- JIT Tracing ---
    print("\n--- JIT Tracing ---")
    try:
        print("Start Tracing...")
        traced_model = torch.jit.trace(model, dummy_input)
        print("End Tracing.")
        # Optional: Save traced model
        # traced_model.save("traced_centernet_decoupled.pt")
        # Test traced model
        with torch.no_grad():
            traced_out = traced_model(dummy_input)
        print("Traced model output shapes (cls, wh, offset, iou):")
        print(f"  cls:    {traced_out[0].shape}")
        print(f"  wh:     {traced_out[1].shape}")
        print(f"  offset: {traced_out[2].shape}")
        print(f"  iou:    {traced_out[3].shape}")
    except Exception as e:
        print(f"Could not trace model: {e}")
        traced_model = model # Fallback to original model if tracing fails

    # --- Dynamic Quantization (CPU) ---
    # Note: Dynamic quantization works best on Linear/LSTM, Conv support is limited.
    # SeparableConv2d might not be directly supported. Quantizing only specific layers.
    # Static quantization is generally preferred for ConvNets but requires calibration.
    print("\n--- Dynamic Quantization (Attempt) ---")
    try:
        # Quantize only layers typically supported well by dynamic quantization
        # This might include the pointwise conv inside SeparableConv2d if bias=True,
        # and the final nn.Conv2d (iou_pred).
        # BatchNorm layers prevent naive fusion needed for static quantization.
        model_quantized = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8 # Specify layers to quantize
        )
        model_quantized.eval()
        model_quantized.to(device) # Ensure it's on CPU for qint8

        print("Quantization applied (dynamically to specified layers).")

        # Warm-up runs
        print("Warm-up runs...")
        with torch.no_grad():
            for i in range(5):
                _ = model_quantized(dummy_input)
                # print(f"Warm-up {i+1}: cls shape {_o[0].shape}") # Optional print

        # Timing settings
        num_runs = 100 # Reduced runs for quicker testing
        start_time = time.time()

        # Run the model multiple times and measure the total time
        with torch.no_grad():
            for _ in range(num_runs):
                outputs = model_quantized(dummy_input)

        end_time = time.time()
        total_time = end_time - start_time
        fps = num_runs / total_time

        print(f"Total inference time for {num_runs} runs (quantized): {total_time:.2f} seconds")
        print(f"Average FPS (quantized): {fps:.2f}")

    except Exception as e:
        print(f"Could not perform dynamic quantization or run inference: {e}")
        print("Skipping quantization timing.")

