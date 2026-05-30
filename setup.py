from setuptools import setup

setup(
    name='pascal',
    version='0.1.0',
    description='PASCAL - Pan-Arctic Behavioural and Life-history Simulator for Calanus',
    py_modules=[
        'coupler_parallel',
        'coupler',
        'individual',
        'pascal_drift',
        'utils',
        'data_logger',
        'pascal42_mod_growthdevelopmentmetabolism',
        'pascal42_mod_growthdevelopmentmetabolism_feedback',
        'pascal42_mod_survival',
        'pascal42_mod_verticalmigration',
    ],
)
