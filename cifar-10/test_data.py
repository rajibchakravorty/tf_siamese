import numpy

from skimage.io import imread, imsave

import os

def get_file_list_label( folder, label ):

    all_files = list( os.walk( folder ) )

    all_files = all_files[0][2]

    all_files = [ os.path.join( folder, x ) for x in all_files ]

    return all_files

def test_and_resave( file_list ):

    rectified_list = list()

    for f in file_list:

        try:
            im = imread( f )

            if im is None:
                continue

            elif len( im.shape ) == 2:
                print f, im.shape
                continue

            elif len( im.shape ) > 3:
                print f, im.shape
                continue

            elif im.shape[2] != 3:
                print f, im.shape
                continue

            rectified_list.append( f )
        except:
            continue
    print 'Started with {0}, final list {1}'.format( len( file_list), len( rectified_list) )

if __name__ == '__main__':
    main_path = '/opt/ml_data/PetImages'

    cat_folder = os.path.join( main_path, '0' )

    cat_list = get_file_list_label( cat_folder, 0 )

    test_and_resave( cat_list )
