import numpy as np

def isd(input_name, output_name, input_shape, output_shape, dtype):
    return {'input_name': input_name,
            'output_name': output_name,
            'input_shape': input_shape, 
            'output_shape': output_shape, 
            'dtype': dtype}

def models(batch):
    return {
        'test': isd('input_1', None, (batch, 296), None, np.float32),
        '4lstm2fcn_8in2out': isd('input_1', 'reshape', (batch, 8, 48), (batch, 2, 12), np.float64),
        'cnn_cpsurf44m': isd('input1', 'conv2d_transpose_10', (batch, 101, 82, 9), (batch, 101, 82, 1), np.float32),
        'tcnn_surfml_212m': isd('input_1', 'time_distributed_53', (batch, 2, 101, 82, 9), (batch, 2, 101, 82, 1), np.float32)
    }
