"""Replay the bundled RBY1/G1 sample motions on OpenArm v1."""
from openarm_teleop import SAMPLE_MOTIONS
from .teleop import parser as common_parser, run


def parse_args(argv=None):
    p = common_parser()
    p.description = __doc__
    p.set_defaults(device='replay')
    p.add_argument('--sample', choices=list(SAMPLE_MOTIONS), default='ipman_roll',
                   help='Bundled recording; ignored when --frames or --csv is given')
    # Existing robot-demo spellings remain convenient when switching robots.
    p.add_argument('--max_frames', dest='steps', type=int, default=argparse.SUPPRESS)
    p.add_argument('--max_fr', dest='rate', type=float, default=argparse.SUPPRESS)
    p.add_argument('--playback_speed', dest='playback_speed', type=float, default=argparse.SUPPRESS)
    p.add_argument('--no-loop', dest='loop', action='store_false')
    args = p.parse_args(argv)
    if args.device != 'replay':
        p.error('offline replay requires --device replay; use openarm-teleop for live input')
    if args.frames is None and args.csv is None:
        args.frames = str(SAMPLE_MOTIONS[args.sample])
    return args


def main():
    run(parse_args())


import argparse
if __name__ == '__main__':
    main()
