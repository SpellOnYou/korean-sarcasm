import json
from argparse import ArgumentParser


def main():
    parser = ArgumentParser('MODEL')
    parser.add_argument('-c', '--config_path', type=str, default=None)
    subparsers = parser.add_subparsers()

    args = parser.parse_args()

    if args.config_path is not None:
        with open(args.config_path) as f:
            config = json.load(f)

        default_config = vars(args)
        for key, default_value in default_config.items():
            if key not in config:
                config[key] = default_value
    else:
        config = vars(args)  # convert to dictionary
        
    args.function(**config, config=config)


if __name__ == '__main__':
    main()
