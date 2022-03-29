import logging

from argparse import ArgumentParser
from ml_genn_tf.converters import (ConverterType, DataNorm, FewSpike, 
                                   InputType, Simple)

from functools import partial

def parse_arguments(model_description='ML GeNN model'):
    '''
    Parses command line arguments for common ML GeNN options, and returns them in namespace form.
    '''

    parser = ArgumentParser(description=model_description)

    # compilation options
    parser.add_argument('--log-level', type=str, default="warning")
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--rng-seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--input-type', default='poisson',
                        choices=[i.value for i in InputType])
    parser.add_argument('--prefer-in-memory-connect', action='store_true')
    parser.add_argument('--kernel-profiling', action='store_true')

    # ANN conversion options
    parser.add_argument('--converter', default='few-spike',
                        choices=[i.value for i in ConverterType])
    parser.add_argument('--n-norm-samples', type=int, default=256)

    # evaluation options
    parser.add_argument('--n-train-samples', type=int, default=None)
    parser.add_argument('--n-test-samples', type=int, default=None)

    # TensorFlow options
    parser.add_argument('--reuse-tf-model', action='store_true')
    parser.add_argument('--record-tensorboard', action='store_true')
    parser.add_argument('--augment-training', action='store_true')

    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=args.log_level.upper())

    def build_converter(self, norm_data, signed_input=False, k=8, 
                        evaluate_timesteps=500):
        if self.converter == 'few-spike':
            return FewSpike(k=k, signed_input=signed_input, 
                            norm_data=[norm_data])
        if args.converter == 'data-norm':
            return DataNorm(evaluate_timesteps=evaluate_timesteps, 
                            signed_input=signed_input,
                            norm_data=[norm_data],
                            input_type=self.input_type)
        #elif args.converter == 'spike-norm':
        #    return SpikeNorm(norm_data=[norm_data], norm_time=norm_time, 
        #                     signed_input=signed_input, input_type=self.input_type)
        else:
            return Simple(signed_input=signed_input, input_type=self.input_type)

    args.build_converter = partial(build_converter, args)

    return args
