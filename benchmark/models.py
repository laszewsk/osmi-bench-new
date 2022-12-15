import numpy as np

def isd(input_name, output_name, input_shape, output_shape, dtype):
    return {'input_name': input_name,
            'output_name': output_name,
            'input_shape': input_shape, 
            'output_shape': output_shape, 
            'dtype': dtype}

def models(batch):
    return {
        'lineml_fcn3datasets': isd('input_1', None, (batch, 296), None, np.float64),
        'lineml_fcn_rev9and12': isd('input_2', None, (batch, 296), None, np.float64),
        '4lstm2fcn_8in2out': isd('input_1', 'reshape', (batch, 8, 48), (batch, 2, 12), np.float64),
        'cnn_cpsurf44m': isd('input1', 'conv2d_transpose_10', (batch, 101, 82, 9), (batch, 101, 82, 1), np.float32),
        'tcnn_surfml_212m': isd('input_1', 'time_distributed_53', (batch, 2, 101, 82, 9), (batch, 2, 101, 82, 1), np.float32),
        'swmodel': isd('dense_input', None, (batch, 3778), None, np.float32),
        'lwmodel': isd('dense_input', None, (batch, 1426), None, np.float32)
    }
