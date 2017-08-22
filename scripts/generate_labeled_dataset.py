import argparse
import os


def arg():
    parser = argparse.ArgumentParser(description='Labeled Dataset Generator.')
    parser.add_argument('--input', '-i', type=str,
                        help='input dataset that contains labeled dataset.')
    parser.add_argument('--labeled', '-l', default=True,
                        help='Whether the dataset directories are named.')
    parser.add_argument('--names', '-n', type=str, default='',
                        help='labeled list of names.')
    parser.add_argument('--output', '-o', type=str, default='output.txt',
                        help='labeled output txt.')
    return parser.parse_args()


def main():
    args = arg()
    dir_list = os.listdir(args.input)
    print(dir_list)
    if args.names:
        with open(args.names, 'w') as f:
            for name in dir_list:
                f.write(str(name) + '\n')

    result_list = []
    for n_class, dir_name in enumerate(dir_list):
        target_dir = os.path.abspath(args.input) + '/' + dir_name
        img_list = os.listdir(target_dir)
        for img in img_list:
            abs_path_to_img = target_dir + '/' + img
            result_list.append((abs_path_to_img, str(n_class)))

    with open(args.output, 'w') as fo:
        for abspath, n_class in result_list:
            fo.write('{} {}\n'.format(abspath, n_class))

if __name__ == '__main__':
    main()
