import os
import tarfile

matrix_data_dir = 'data/matrices'

# Extract tar files
print('Extracting tar files to mtx files...')
for root, dirs, files in os.walk(matrix_data_dir):
    for _fname in files:
        fname = os.path.join(root, _fname)
        if fname.endswith('.tar.gz'):
            print('Extracting file: {fname}')
            tar = tarfile.open(fname, 'r:gz')
            tar.extractall(path=matrix_data_dir)
            tar.close()
            os.remove(fname)

# Move mtx files from directories to data/matrices (basically flat extraction)
for root, dirs, files in os.walk(matrix_data_dir):
    for _dname in dirs:
        dname = os.path.join(root, _dname)
        for root_sub, dirs_sub, files_sub in os.walk(dname):
            for _fname_sub in files_sub:
                fname_sub = os.path.join(dname, _fname_sub)
                fname = os.path.join(matrix_data_dir, _fname_sub)
                os.rename(fname_sub, fname)
        os.rmdir(dname)
print('Done.')
