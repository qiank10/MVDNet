# MVDNet
Robust Multimodal Vehicle Detection in Foggy Weather Using Complementary Lidar and Radar Signals, CVPR 2021.

## Prepare Data

Download the [Oxford Radar RobotCar Dataset](https://oxford-robotics-institute.github.io/radar-robotcar-dataset). Currently, only the vehicles in the first data record (Data: 10/01/2019, Time: 11:46:21 GMT) are labeled. After unzipping the files, the directory should look like this:
```
# Oxford Radar RobotCar Data Record
|-- DATA_PATH
    |-- gt
    |-- radar
    |-- velodyne_left
    |-- velodyne_right
    |-- vo
    |-- radar.timestamps
    |-- velodyne_left.timestamps
    |-- velodyne_right.timestamps
    |-- ...
```

Prepare the radar data:
```
python data/sdk/prepare_radar_data.py --data_path DATA_PATH --image_size 320 --resolution 0.2
```

Prepare the lidar data:
```
python data/sdk/prepare_lidar_data.py --data_path DATA_PATH
```

Prepare the foggy lidar test set with specified fog density, e.g., 0.05:
```
python data/sdk/prepare_fog_data.py --data_path DATA_PATH --beta 0.05
```

The processed data is organized as follows:
```
# Oxford Radar RobotCar Data Record
|-- DATA_PATH
    |-- processed
        |-- radar
            |-- 1547120789640420.jpg
            |-- ...
        |-- radar_history
            |-- 1547120789640420_k.jpg   # The k-th radar frame preceding the frame at the timestamp 1547120789640420, k=1,2,3,4.
            |-- ...
        |-- lidar
            |-- 1547120789640420.bin
            |-- ...
        |-- lidar_history
            |-- 1547120789640420_k.bin   # Link to the k-th lidar frame preceding the frame at the timestamp 1547120789640420, k=1,2,3,4.
            |-- 1547120789640420_k_T.bin # Transform matrix between the k-th preceding lidar frame and the current frame.
            |-- ...
        |-- lidar_fog_0.05               # Foggy lidar data with fog density as 0.05
            |-- 1547120789640420.bin
            |-- ...
        |-- lidar_history_fog_0.05
            |-- 1547120789640420_k.bin
            |-- 1547120789640420_k_T.bin
            |-- ...
```

Both 2D and 3D labels are in
```
./data/RobotCar/object/
```
