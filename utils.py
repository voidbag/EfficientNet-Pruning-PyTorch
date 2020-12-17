import copy
import efficientnet_repo.Yet_Another_EfficientDet_Pytorch.efficientnet.utils_extra as yet_utils
import efficientnet_repo.Yet_Another_EfficientDet_Pytorch.efficientnet.model as yet_model

import efficientnet_repo.EfficientNet_PyTorch.efficientnet_pytorch.utils as ept_utils
import efficientnet_repo.EfficientNet_PyTorch.efficientnet_pytorch.model as ept_model


_origin_from_name = ept_model.EfficientNet.from_name

def _from_name_hook(model_name, cfg=None, override_params=dict(num_classes=1000)):
    override_params = copy.deepcopy(override_params)
    return _origin_from_name(model_name, cfg=cfg, **override_params)

class EfficientNetModule(object):
    def __init__(self, repo):
        self.repo = repo 

    def MBConvBlock(self):
        ret_cls = None
        if self.repo == "yet": #Yet another efficientDet pytorch
            ret_cls = yet_model.MBConvBlock
        elif self.repo == "ept": #Efficientnet PyTorch
            ret_cls = ept_model.MBConvBlock
        else:
            ret_cls = None

        assert ret_cls is not None
        return ret_cls

    def Conv2dStaticSamePadding(self):
        ret_cls = None
        if self.repo == "yet":
            ret_cls = yet_utils.Conv2dStaticSamePadding
        elif self.repo == "ept":
            ret_cls = ept_utils.Conv2dStaticSamePadding
        else:
            ret_cls = None

        assert ret_cls is not None
        return ret_cls

    def EfficientNet(self, need_hook=False):
        ret_cls = None
        if self.repo == "yet":
            ret_cls = yet_model.EfficientNet
        elif self.repo == "ept":
            ret_cls = ept_model.EfficientNet
            if need_hook:
                global _from_name_hook
                ept_model.EfficientNet.from_name = _from_name_hook

        assert ret_cls is not None
        return ret_cls
    
    
        

    def from_pruned(self, name, path, override_params):
        efficientnet = self.EfficientNet()
        return efficientnet.from_name_pruned(name, state_dict_path=path, override_params=override_params)

def test_enet_module():
    yet_module = EfficientNetModule("yet")
    assert yet_module.MBConvBlock() == yet_model.MBConvBlock
    assert yet_module.Conv2dStaticSamePadding() == yet_utils.Conv2dStaticSamePadding
    
    
    ept_module = EfficientNetModule("ept")
    assert ept_module.MBConvBlock() == ept_model.MBConvBlock
    assert ept_module.Conv2dStaticSamePadding() == ept_utils.Conv2dStaticSamePadding


def main():
    test_enet_module()

if __name__ == "__main__":
    main()

