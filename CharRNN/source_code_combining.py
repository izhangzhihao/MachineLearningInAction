import os
import sys

# walk_dir = sys.argv[1]
walk_dir = '/Users/zhazhang/Downloads/guava-master'

print('walk_dir = ' + walk_dir)

# If your current working directory may change during script execution, it's recommended to
# immediately convert program arguments to an absolute path. Then the variable root below will
# be an absolute path as well. Example:
# walk_dir = os.path.abspath(walk_dir)
print('walk_dir (absolute) = ' + os.path.abspath(walk_dir))

save_path = '/Users/zhazhang/Downloads/TensorFlowPractice/CharRNN/data/guava.txt'

with open(save_path, 'wb') as save_file:
    # noinspection SpellCheckingInspection
    for root, subdirs, files in os.walk(walk_dir):
        print('--\nroot = ' + root)
        list_file_path = os.path.join(root, '/guava-master')
        print('list_file_path = ' + list_file_path)

        with open(list_file_path, 'wb') as list_file:
            for subdir in subdirs:
                print('\t- subdirectory ' + subdir)

            for filename in files:
                if filename.__contains__('.java'):
                    file_path = os.path.join(root, filename)

                    print('\t- file %s (full path: %s)' % (filename, file_path))

                    with open(file_path, 'rb') as f:
                        f_content = f.read()
                        save_file.write(f_content)
