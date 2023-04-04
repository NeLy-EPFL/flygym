import pickle
import argparse
from pathlib import Path


legs = [f'{side}{pos}' for side in 'LR' for pos in 'FMH']
joint_name_lookup = {
    'ThC_yaw': 'Coxa_yaw',
    'ThC_pitch': 'Coxa',
    'ThC_roll': 'Coxa_roll',
    'CTr_pitch': 'Femur',
    'CTr_roll': 'Femur_roll',
    'FTi_pitch': 'Tibia',
    'TiTa_pitch': 'Tarsus1'
}


def parse_args():
    """ Parse arguments from command line """
    parser = argparse.ArgumentParser(
        description=('Selecte portion of DF3DPostProcessing joint angle output '
                     'and convert it to NeuroMechFly DoF naming convention')
    )
    parser.add_argument('input', help='Input file joint angles file')
    parser.add_argument('output', help='Output file joint angles file')
    parser.add_argument('-dt', '--timestep', default=5e-4, type=float,
                        help='Timestep of the input file')
    parser.add_argument('-s', '--start', default=4.0, type=float,
                        help='Start time of the period to be selected')
    parser.add_argument('-e', '--end', default=5.0, type=float,
                        help='End time of the period to be selected')
    return parser.parse_args()


def main():
    args = parse_args()
    input_file = args.input
    output_file = args.output
    timestep = args.timestep
    start_time = args.start
    end_time = args.end
    
    with open(input_file, 'rb') as f:
        input_data = pickle.load(f)
    
    start_idx = int(start_time / timestep)
    end_idx = int(end_time / timestep)
    out_dict = {'meta': {'timestep': timestep,
                         'source_file': input_file,
                         'time_range': (start_time, end_time)}}
    
    for leg in legs:
        for src_dof_name, tgt_dof_name in joint_name_lookup.items():
            ts = input_data[f'{leg}_leg'][src_dof_name]
            out_dict[f'joint_{leg}{tgt_dof_name}'] = ts[start_idx:end_idx]
    
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(out_dict, f)


if __name__ == '__main__':
    main()
