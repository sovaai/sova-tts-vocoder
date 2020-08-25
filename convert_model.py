# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************
import sys
import copy
import torch


def _check_model_old_version(model):
    if hasattr(model.WN[0], 'res_layers') or hasattr(model.WN[0], 'cond_layers'):
        return True
    else:
        return False


def _update_model_res_skip(old_model, new_model):
    for idx in range(0, len(new_model.WN)):
        wavenet = new_model.WN[idx]
        n_channels = wavenet.n_channels
        n_layers = wavenet.n_layers
        wavenet.res_skip_layers = torch.nn.ModuleList()
        for i in range(0, n_layers):
            if i < n_layers - 1:
                res_skip_channels = 2 * n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            skip_layer = torch.nn.utils.remove_weight_norm(wavenet.skip_layers[i])
            if i < n_layers - 1:
                res_layer = torch.nn.utils.remove_weight_norm(wavenet.res_layers[i])
                res_skip_layer.weight = torch.nn.Parameter(torch.cat([res_layer.weight, skip_layer.weight]))
                res_skip_layer.bias = torch.nn.Parameter(torch.cat([res_layer.bias, skip_layer.bias]))
            else:
                res_skip_layer.weight = torch.nn.Parameter(skip_layer.weight)
                res_skip_layer.bias = torch.nn.Parameter(skip_layer.bias)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            wavenet.res_skip_layers.append(res_skip_layer)
        del wavenet.res_layers
        del wavenet.skip_layers


def _update_model_cond(old_model, new_model):
    for idx in range(0, len(new_model.WN)):
        wavenet = new_model.WN[idx]
        n_channels = wavenet.n_channels
        n_layers = wavenet.n_layers
        n_mel_channels = wavenet.cond_layers[0].weight.shape[1]
        cond_layer = torch.nn.Conv1d(n_mel_channels, 2 * n_channels * n_layers, 1)
        cond_layer_weight = []
        cond_layer_bias = []
        for i in range(0, n_layers):
            _cond_layer = torch.nn.utils.remove_weight_norm(wavenet.cond_layers[i])
            cond_layer_weight.append(_cond_layer.weight)
            cond_layer_bias.append(_cond_layer.bias)
        cond_layer.weight = torch.nn.Parameter(torch.cat(cond_layer_weight))
        cond_layer.bias = torch.nn.Parameter(torch.cat(cond_layer_bias))
        cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
        wavenet.cond_layer = cond_layer
        del wavenet.cond_layers


def update_model(old_model):
    if not _check_model_old_version(old_model):
        return old_model
    new_model = copy.deepcopy(old_model)
    if hasattr(old_model.WN[0], 'res_layers'):
        _update_model_res_skip(old_model, new_model)
    if hasattr(old_model.WN[0], 'cond_layers'):
        _update_model_cond(old_model, new_model)
    return new_model


if __name__ == '__main__':
    old_model_path = sys.argv[1]
    new_model_path = sys.argv[2]
    model = torch.load(old_model_path, map_location='cpu')
    model['model'] = update_model(model['model'])
    torch.save(model, new_model_path)
