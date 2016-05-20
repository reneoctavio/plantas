import os
import os.path
import fileinput
import shutil
import math

'''
networks = {'AlexNet': ['Default'],
            'CaffeNet': ['Default', 'PReLU'],
            'GoogLeNet': ['Default', 'Finetuned', 'PReLU', 'Finetuned-PReLU'],
            'Inception': ['Default', 'ELU', 'PReLU'],
            'ResNet-50': ['Default', 'PReLU', 'Finetuned'],
            'ResNet-101': ['Default']}

'''

networks = {'GoogLeNet': ['Finetuned-Zero'], 'CaffeNet': ['Finetuned-Zero']}

core_dataset = '/usr/share/digits/digits/jobs/20160326-172626-a652'
full_dataset = '/usr/share/digits/digits/jobs/20160326-143203-b20e'
archictetures_path = '/home/reneoctavio/Documents/Plantas/Training/Architectures'
archictetures_core_path = '/home/reneoctavio/Documents/Plantas/Training/Core'
archictetures_full_path = '/home/reneoctavio/Documents/Plantas/Training/Full'

script_path = '/home/reneoctavio/Documents/Plantas/Training/run.sh'

finetune_files = {'CaffeNet': '/home/reneoctavio/Downloads/bvlc_reference_caffenet.caffemodel',
                    'GoogLeNet': '/home/reneoctavio/Downloads/bvlc_googlenet.caffemodel',
                    'ResNet-50': '/home/reneoctavio/Downloads/ResNet-50-model.caffemodel'}

core_size = {'train': 6585, 'test': 1410}
full_size = {'train': 23158, 'test': 4944}

batch_size = {'AlexNet': 100.0,
            'CaffeNet': 256.0,
            'GoogLeNet': 32.0,
            'Inception': 12.0,
            'ResNet-50': 6.0,
            'ResNet-101': 4.0}

epochs = 30
steps = 3

def treat_files(read_file, write_file, set_path, arch, deriv, batch_size):
    # Treat derivations
    if 'PReLU' in deriv:
        for line in read_file.readlines():
            if 'mean_file:' in line:
                mean_path = os.path.join(set_path, 'mean.binaryproto')
                line = '    mean_file: "' + mean_path + '"\n'
            if 'source:' in line and 'train_db' in line:
                source_path = os.path.join(set_path, 'train_db')
                line = '    source: "' + source_path + '"\n'
            if 'source:' in line and 'val_db' in line:
                source_path = os.path.join(set_path, 'val_db')
                line = '    source: "' + source_path + '"\n'
            if 'ReLU' in line:
                line = '  type: "PReLU"' + '\n'
            if 'batch_size:' in line:
                line = '    batch_size: ' + str(int(batch_size[arch])) + '\n'
            write_file.write(line)

    elif 'ELU' in deriv:
        for line in read_file.readlines():
            if 'mean_file:' in line:
                mean_path = os.path.join(set_path, 'mean.binaryproto')
                line = '    mean_file: "' + mean_path + '"\n'
            if 'source:' in line and 'train_db' in line:
                source_path = os.path.join(set_path, 'train_db')
                line = '    source: "' + source_path + '"\n'
            if 'source:' in line and 'val_db' in line:
                source_path = os.path.join(set_path, 'val_db')
                line = '    source: "' + source_path + '"\n'
            if 'batch_size:' in line:
                line = '    batch_size: ' + str(int(batch_size[arch])) + '\n'
            if 'ReLU' in line:
                line = '  type: "ELU" elu_param{ alpha: 0.1 }' + '\n'
            write_file.write(line)

    elif 'Zero' in deriv:
        last_layer = False
        for line in read_file.readlines():
            # Check if last line before classification
            if ('fc50' in line) or ('classifier_new' in line):
                last_layer = True
            if 'layer' in line:
                last_layer = False
            if 'mean_file:' in line:
                mean_path = os.path.join(set_path, 'mean.binaryproto')
                line = '    mean_file: "' + mean_path + '"\n'
            if 'source:' in line and 'train_db' in line:
                source_path = os.path.join(set_path, 'train_db')
                line = '    source: "' + source_path + '"\n'
            if 'source:' in line and 'val_db' in line:
                source_path = os.path.join(set_path, 'val_db')
                line = '    source: "' + source_path + '"\n'
            if 'batch_size:' in line:
                line = '    batch_size: ' + str(int(batch_size[arch])) + '\n'
            if ('lr_mult' in line) and (not last_layer):
                line = '  lr_mult: 0.0' + '\n'

            write_file.write(line)

    else:
        for line in read_file.readlines():
            if 'mean_file:' in line:
                mean_path = os.path.join(set_path, 'mean.binaryproto')
                line = '    mean_file: "' + mean_path + '"\n'
            if 'source:' in line and 'train_db' in line:
                source_path = os.path.join(set_path, 'train_db')
                line = '    source: "' + source_path + '"\n'
            if 'source:' in line and 'val_db' in line:
                source_path = os.path.join(set_path, 'val_db')
                line = '    source: "' + source_path + '"\n'
            if 'batch_size:' in line:
                line = '    batch_size: ' + str(int(batch_size[arch])) + '\n'
            write_file.write(line)

def copy_and_modify(arch_path, arch_dest_path, set_path, script_f, set_size, batch_size, networks, epochs, steps, finetune_files):
    for arch, derivs in networks.iteritems():
        for deriv in derivs:
            # Create path for file
            path = os.path.join(arch_dest_path, arch + '-' + deriv)
            if not os.path.exists(path):
                os.makedirs(path)

            # Copy caffe files
            path_original = os.path.join(arch_path, arch)

            # Files to modify
            train_val_f_origin = open(os.path.join(path_original, 'train_val.prototxt'), 'r')
            train_val_f_target = open(os.path.join(path, 'train_val.prototxt'), 'w')

            deploy_f_origin = open(os.path.join(path_original, 'deploy.prototxt'), 'r')
            deploy_f_target = open(os.path.join(path, 'deploy.prototxt'), 'w')

            solver_f_target = open(os.path.join(path, 'solver.prototxt'), 'w')

            # Treat train_val
            treat_files(train_val_f_origin, train_val_f_target, set_path, arch, deriv, batch_size)
            # Treat deploy
            treat_files(deploy_f_origin, deploy_f_target, set_path, arch, deriv, batch_size)

            # Solver
            test_iter = int(math.ceil(set_size['test'] / batch_size[arch]))
            test_interval = int(math.ceil(set_size['train'] / batch_size[arch]))
            display = int(test_interval / 8)
            max_iter = int(test_interval * epochs)
            stepsize = int(round(max_iter / 3))
            snapshot = int(max_iter / 2)
            solver_txt = 'test_iter: ' + str(test_iter) + '\n' + \
                         'test_interval: ' + str(test_interval) + '\n' + \
                         'base_lr: 0.01' + '\n' + \
                         'display: ' + str(display) + '\n' + \
                         'max_iter: ' + str(max_iter) + '\n' + \
                         'lr_policy: "step"' + '\n' + \
                         'gamma: 0.1' + '\n' + \
                         'momentum: 0.9' + '\n' + \
                         'weight_decay: 0.0005' + '\n' + \
                         'stepsize: ' + str(stepsize) + '\n' + \
                         'snapshot: ' + str(snapshot) + '\n' + \
                         'snapshot_prefix: "' + os.path.join(path, 'snapshot') + '"\n' + \
                         'solver_mode: GPU' + '\n' + \
                         'net: "' + os.path.join(path, 'train_val.prototxt') + '"\n' + \
                         'solver_type: SGD' + '\n'
            solver_f_target.write(solver_txt)

            # Script file
            log_path = os.path.join(path, 'caffe_output.log')
            if os.path.isfile(log_path):
                os.remove(log_path)

            script_txt = './build/tools/caffe train -solver ' + \
                os.path.join(path, 'solver.prototxt')
            if 'Finetuned' in deriv:
                script_txt += ' -weights ' + finetune_files[arch]
            script_txt += ' 2>&1 | tee -a ' + os.path.join(path, 'caffe_output.log')
            script_f.write(script_txt + '\n')

# Open script file
os.remove(script_path)
script_f = open(script_path, 'a')
script_f.write('#!/bin/bash\n')

# Core
copy_and_modify(archictetures_path, archictetures_core_path, core_dataset, script_f, core_size, batch_size, networks, epochs, steps, finetune_files)

# Full
copy_and_modify(archictetures_path, archictetures_full_path, full_dataset, script_f, full_size, batch_size, networks, epochs, steps, finetune_files)
