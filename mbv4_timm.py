import torch
import torch.nn as nn
import torch.nn.functional as F
import timm 

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                    bias=bias),
                                    nn.BatchNorm2d(in_channels))
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


def normal_init(module, mean=0, std=1, bias=0):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias'):
            nn.init.constant_(module.bias, bias)


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

        z = nn.functional.interpolate(z, scale_factor=2,mode='bilinear' )

        return z
    

class Fpn(nn.Module):
    def __init__(self,input_dims=[24,32,96,320],head_dims=[128,128,128] ):
        super().__init__()





        self.latlayer2=nn.Sequential(SeparableConv2d(input_dims[0],head_dims[0]//2,kernel_size=5,padding=2),
                                      nn.BatchNorm2d(head_dims[0]//2),
                                      nn.ReLU(inplace=True))


        self.latlayer3=nn.Sequential(SeparableConv2d(input_dims[1],head_dims[1]//2,kernel_size=5,padding=2),
                                      nn.BatchNorm2d(head_dims[1]//2),
                                      nn.ReLU(inplace=True))

        self.latlayer4 = nn.Sequential(SeparableConv2d(input_dims[2], head_dims[2] // 2,kernel_size=5,padding=2),
                                       nn.BatchNorm2d(head_dims[2] // 2),
                                       nn.ReLU(inplace=True))



        self.upsample3=ComplexUpsample(head_dims[1],head_dims[0]//2)

        self.upsample4 =ComplexUpsample(head_dims[2],head_dims[1]//2)

        self.upsample5 = ComplexUpsample(input_dims[3],head_dims[2]//2)




    def forward(self, inputs):
        ##/24,32,96,320
        c2, c3, c4, c5 = inputs

        c4_lat = self.latlayer4(c4)
        c3_lat = self.latlayer3(c3)
        c2_lat = self.latlayer2(c2)


        upsample_c5=self.upsample5(c5)

        p4=torch.cat([c4_lat,upsample_c5],dim=1)


        upsample_p4=self.upsample4(p4)

        p3=torch.cat([c3_lat,upsample_p4],dim=1)

        upsample_p3 = self.upsample3(p3)

        p2 = torch.cat([c2_lat, upsample_p3],dim=1)


        return p2

class CenterNetHead(nn.Module):
    def __init__(self,nc,head_dims ):
        super().__init__()



        self.cls =SeparableConv2d(head_dims[0], nc, kernel_size=3, stride=1, padding=1, bias=True)
        self.wh =SeparableConv2d(head_dims[0], 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.offset =SeparableConv2d(head_dims[0], 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.iou_head = nn.Conv2d(head_dims[0], 1, kernel_size=3, stride=1, padding=1, bias=True)

        normal_init(self.cls.pointwise, 0, 0.01,-2.19)
        normal_init(self.wh.pointwise, 0, 0.01, 0)



    def forward(self, inputs):


        cls = self.cls(inputs).sigmoid_()
        wh = self.wh(inputs)
        offset = self.offset(inputs)
        iou_aware_head = self.iou_head(inputs).sigmoid_().squeeze(1)
        return cls,wh,offset,iou_aware_head

class CenterNet(nn.Module):
    def __init__(self,nc):
        super().__init__()

        self.nc = nc 
        input_dims = []
        ###model structure
        self.backbone =  timm.create_model('mobilenetv4_conv_small.e1200_r224_in1k', pretrained=True, features_only=True,exportable=True) #e1200_r224_in1k - 050
        feature_info = self.backbone.feature_info
        for idx, info in enumerate(feature_info.info):
            #print(f"Seviye {idx+1}: {info['module']}, kanal sayısı={info['num_chs']}")
            if idx > 0:
                input_dims.append(info['num_chs'])
        self.fpn=Fpn(input_dims=input_dims,head_dims=[128,128,128])
        self.head = CenterNetHead(self.nc,head_dims=[128,128,128])


        self.device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    def forward(self, inputs):

        fms = self.backbone(inputs)[-4:]
        fpn_fm=self.fpn(fms)
        cls, wh, offset, iou_aware_head= self.head(fpn_fm)


        return cls, wh, offset, iou_aware_head
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    import time 
    model = CenterNet(10)

    ### load your weights
    model.eval()
    # Providing input and output names sets the display names for values
    # within the model's graph. Setting these does not change the semantics
    # of the graph; it is only for readability.
    #
    # The inputs to the network consist of the flat list of inputs (i.e.
    # the values you would pass to the forward() method) followed by the
    # flat list of parameters. You can partially specify names, i.e. provide
    # a list here shorter than the number of inputs to the model, and we will
    # only set that subset of names, starting from the beginning.
    # Move model to CPU
    batch_size = 1
    input_height = 320
    input_width = 320
    device = torch.device('cpu')
    model.to(device)
    modelparams = count_parameters(model)
    print(f"Total Model Params: {modelparams:,}")
    dummy_input = torch.randn(batch_size, 3, input_height, input_width).to(device)
    out = model(dummy_input)
    print(out[0].shape)
    print(out[1].shape)
    print("Start Tracing")
    model = torch.jit.trace(model, dummy_input)
    print("End Tracing")

    # Create dummy input data
    # Quantize the model for faster CPU inference
    model_quantized = torch.quantization.quantize_dynamic(
        model, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
    )
    model_quantized.eval()
    model_quantized.to(device)



    # Warm-up runs (to exclude initialization overhead)
    with torch.no_grad():
        for _ in range(10):
            _ = model_quantized(dummy_input)
            print(_[0].shape)

    # Timing settings
    num_runs = 100
    start_time = time.time()

    # Run the model multiple times and measure the total time
    with torch.no_grad():
        for _ in range(num_runs):
            outputs = model_quantized(dummy_input)

    end_time = time.time()
    total_time = end_time - start_time
    fps = num_runs / total_time

    print(f"Total inference time for {num_runs} runs: {total_time:.2f} seconds")
    print(f"Average FPS: {fps:.2f}")