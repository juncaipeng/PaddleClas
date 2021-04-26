# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..', '..', '..')))
sys.path.append(
    os.path.abspath(os.path.join(__dir__, '..', '..', '..', 'tools')))

import paddle
from paddleslim.dygraph.quant import QAT

from ppcls.data import Reader
from ppcls.utils.config import get_config, print_config
from ppcls.utils.save_load import init_model, save_model
from ppcls.utils import logger
import program

from pact_helper import get_default_quant_config


def parse_args():
    parser = argparse.ArgumentParser("PaddleClas train script")
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='configs/ResNet/ResNet50.yaml',
        help='config file path')
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        default=[],
        help='config options to be overridden')
    args = parser.parse_args()
    return args


def main(args):
    # vars
    use_quant = True
    use_pact = True
    lr = 0.0001
    epoch = 1
    run_batch = 100
    batch_size = 64
    num_workers = 1
    data_dir = '/dataset/ILSVRC2012/'
    train_file_list = '/dataset/ILSVRC2012/train_list.txt'
    valid_file_list = '/dataset/ILSVRC2012/val_list.txt'
    pretrain_dir = './pretrain'

    # ---------
    paddle.seed(12345)

    config = get_config(args.config, overrides=args.override, show=False)
    config['TRAIN']['batch_size'] = batch_size
    config['TRAIN']['file_list'] = train_file_list
    config['TRAIN']['data_dir'] = data_dir
    config['TRAIN']['num_workers'] = num_workers
    config['VALID']['batch_size'] = batch_size
    config['VALID']['file_list'] = valid_file_list
    config['VALID']['data_dir'] = data_dir
    config['VALID']['num_workers'] = num_workers
    config['epochs'] = epoch
    config['LEARNING_RATE']['params']['lr'] = lr
    pretrained_model = os.path.join(pretrain_dir, config['ARCHITECTURE']['name'] + "_pretrained")
    assert os.path.exists(pretrained_model + ".pdparams")
    config['pretrained_model'] = pretrained_model
    print_config(config)

    # assign the place
    use_gpu = config.get("use_gpu", True)
    place = paddle.set_device('gpu' if use_gpu else 'cpu')

    trainer_num = paddle.distributed.get_world_size()
    use_data_parallel = trainer_num != 1
    config["use_data_parallel"] = use_data_parallel

    if config["use_data_parallel"]:
        paddle.distributed.init_parallel_env()

    net = program.create_model(config.ARCHITECTURE, config.classes_num)
    optimizer, lr_scheduler = program.create_optimizer(
        config, parameter_list=net.parameters())
    init_model(config, net, optimizer)

    # prepare to quant
    if use_quant:
        quant_config = get_default_quant_config()
        quant_config["activation_preprocess_type"] = "PACT" if use_pact else None 
        quanter = QAT(config=quant_config)
        quanter.quantize(net)

    if config["use_data_parallel"]:
        net = paddle.DataParallel(net)

    train_dataloader = Reader(config, 'train', places=place)()

    if config.validate:
        valid_dataloader = Reader(config, 'valid', places=place)()

    last_epoch_id = config.get("last_epoch", -1)
    best_top1_acc = 0.0  # best top1 acc record
    best_top1_epoch = last_epoch_id
    for epoch_id in range(last_epoch_id + 1, config.epochs):
        net.train()
        # 1. train with train dataset
        program.run(train_dataloader, config, net, optimizer, lr_scheduler,
                    epoch_id, 'train', run_batch=run_batch)

        # 2. validate with validate dataset
        if config.validate and epoch_id % config.valid_interval == 0:
            net.eval()
            with paddle.no_grad():
                top1_acc = program.run(valid_dataloader, config, net, None,
                                       None, epoch_id, 'valid', run_batch=run_batch)
            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                best_top1_epoch = epoch_id
                model_path = os.path.join(config.model_save_dir,
                                          config.ARCHITECTURE["name"])
                #save_model(net, optimizer, model_path, "best_model")
            message = "The best top1 acc {:.5f}, in epoch: {:d}".format(
                best_top1_acc, best_top1_epoch)
            logger.info(message)
        
        '''
        # 3. save the persistable model
        if epoch_id % config.save_interval == 0:
            model_path = os.path.join(config.model_save_dir,
                                      config.ARCHITECTURE["name"])
            save_model(net, optimizer, model_path, epoch_id)
        '''
        if use_quant:
            save_model_path = os.path.join(config.model_save_dir, config.ARCHITECTURE["name"])
            save_model_path = save_model_path + "_pact" if use_pact else ""
            quanter.save_quantized_model(net, model_path, 
                input_spec=[paddle.static.InputSpec(shape=[None, 3, 224, 224], dtype='float32')])
            print('finish')


if __name__ == '__main__':
    args = parse_args()
    main(args)
