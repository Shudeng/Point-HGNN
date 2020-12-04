"""
From Point-GNN
"""

import torch
from torch import nn

def multi_layer_fc_fn(Ks=[300, 64, 32, 64], num_classes=4, is_logits=False, num_layers=4):
    assert len(Ks) == num_layers
    linears = []
    for i in range(1, len(Ks)):
        linears += [
                nn.Linear(Ks[i-1], Ks[i]),
                nn.ReLU(),
                nn.BatchNorm1d(Ks[i])
                ]

    if is_logits:
        linears += [
                nn.Linear(Ks[-1], num_classes)]
    else:
        linears += [
                nn.Linear(Ks[-1], num_classes),
                nn.ReLU(),
                nn.BatchNorm1d(num_classes)
                ]
    return nn.Sequential(*linears)


class ClassAwarePredictor(nn.Module):
    def __init__(self, num_classes, box_encoding_len):
        super(ClassAwarePredictor, self).__init__()
        # self.cls_fn = multi_layer_fc_fn(Ks=[300, 64], num_layers=2, num_classes=num_classes, is_logits=True)
        self.cls_fn = multi_layer_fc_fn(Ks=[64, 32], num_layers=2, num_classes=num_classes, is_logits=True)
        self.loc_fns = nn.ModuleList()
        self.num_classes = num_classes
        self.box_encoding_len = box_encoding_len

        for i in range(num_classes):
            # self.loc_fns += [
            #         multi_layer_fc_fn(Ks=[300, 300, 64], num_layers=3, num_classes=box_encoding_len, is_logits=True)]
            self.loc_fns += [
                    multi_layer_fc_fn(Ks=[64, 64, 32], num_layers=3, num_classes=box_encoding_len, is_logits=True)]

    def forward(self, features):
        logits = self.cls_fn(features)
        box_encodings_list = []
        for loc_fn in self.loc_fns:
            box_encodings = loc_fn(features).unsqueeze(1)
            box_encodings_list += [box_encodings]

        box_encodings = torch.cat(box_encodings_list, dim=1)
        return logits, box_encodings
