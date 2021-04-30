import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from .robotcar import get_robotcar_dicts

def register_robotcar(root):
    root = os.path.join(root, 'RobotCar')
    register_split('train', root)
    register_split('eval', root)
    
def register_split(split_name, root):
    data_root = os.path.join(root, 'object')
    split_path = os.path.join(root, 'ImageSets', split_name+'.txt')
    DatasetCatalog.register('robotcar_' + split_name, lambda data_root=data_root, 
        split_path=split_path: get_robotcar_dicts(data_root, split_path))
    MetadataCatalog.get('robotcar_' + split_name).set(thing_classes=['car'],
        data_root=data_root, split_path=split_path, evaluator_type='robotcar', 
        thing_dataset_id_to_contiguous_id={1:0})

register_robotcar('./data')