import numpy as np

from os.path import join

def raw_class_text_to_list( text_file, root_data_folder ):

    with open( text_file , 'rb ') as f:

        lines = f.readlines()

        raw_list = list()
        for line in lines:

            filename, label = line.split(',')

            label = int( label )

            filename = join( root_data_folder,filename)

            raw_list.append( (filename, label ) )

    return raw_list
