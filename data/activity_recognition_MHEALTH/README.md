# Activity Recognition MHEALTH dataset

The MHEALTH (Mobile HEALTH) dataset comprises body motion and vital signs recordings for ten volunteers of the diverse profile while performing several physical activities. Sensors placed on the subject's chest, right wrist, and left ankle are used to measure the motion experienced by diverse body parts, namely, acceleration, rate of turn, and magnetic field orientation. The sensor positioned on the chest also provides 2-lead ECG measurements, which can be potentially used for basic heart monitoring, checking for various arrhythmias, or looking at the effects of exercise on the ECG.

Summary:
Activities: 12
Sensor devices: 3
Subjects: 10

The collected dataset comprises body motion and vital signs recordings for ten volunteers of the diverse profile while performing 12 physical activities (Table 1). Shimmer2 [BUR10] wearable sensors were used for the recordings. The sensors were respectively placed on the subject's chest, right wrist, and left ankle and attached by using elastic straps (as shown in the figure in the attachment). The use of multiple sensors permits us to measure the motion experienced by diverse body parts, namely, the acceleration, the rate of turn, and the magnetic field orientation, thus better capturing the body dynamics. The sensor positioned on the chest also provides 2-lead ECG measurements which are not used for the development of the recognition model but rather collected for future work purposes. This information can be used, for example, for basic heart monitoring, checking for various arrhythmias, or looking at the effects of exercise on the ECG. All sensing modalities are recorded at a sampling rate of 50 Hz, which is considered sufficient for capturing human activity. Each session was recorded using a video camera. This dataset is found to generalize to common activities of daily living, given the diversity of body parts involved in each one (e.g., the frontal elevation of arms vs. knees bending), the intensity of the actions (e.g., cycling vs. sitting and relaxing) and their execution speed or dynamicity (e.g., running vs. standing still). The activities were collected in an out-of-lab environment with no constraints on the way these must be executed, with the exception that the subject should try their best when executing them.

## Columns

Column 1: acceleration from the chest sensor (X-axis)

Column 2: acceleration from the chest sensor (Y axis)

Column 3: acceleration from the chest sensor (Z axis)

Column 4: electrocardiogram signal (lead 1)

Column 5: electrocardiogram signal (lead 2)

Column 6: acceleration from the left-ankle sensor (X-axis)

Column 7: acceleration from the left-ankle sensor (Y axis)

Column 8: acceleration from the left-ankle sensor (Z axis)

Column 9: gyro from the left-ankle sensor (X-axis)

Column 10: gyro from the left-ankle sensor (Y axis)

Column 11: gyro from the left-ankle sensor (Z axis)

Column 13: magnetometer from the left-ankle sensor (X-axis)

Column 13: magnetometer from the left-ankle sensor (Y axis)

Column 14: magnetometer from the left-ankle sensor (Z axis)

Column 15: acceleration from the right-lower-arm sensor (X-axis)

Column 16: acceleration from the right-lower-arm sensor (Y axis)

Column 17: acceleration from the right-lower-arm sensor (Z axis)

Column 18: gyro from the right-lower-arm sensor (X-axis)

Column 19: gyro from the right-lower-arm sensor (Y axis)

Column 20: gyro from the right-lower-arm sensor (Z axis)

Column 21: magnetometer from the right-lower-arm sensor (X-axis)

Column 22: magnetometer from the right-lower-arm sensor (Y axis)

Column 23: magnetometer from the right-lower-arm sensor (Z axis)

Column 24: Label (0 for the null class)
L1: Standing still (1 min)
L2: Sitting and relaxing (1 min)
L3: Lying down (1 min)
L4: Walking (1 min)
L5: Climbing stairs (1 min)
L6: Waist bends forward (20x)
L7: Frontal elevation of arms (20x)
L8: Knees bending (crouching) (20x)
L9: Cycling (1 min)
L10: Jogging (1 min)
L11: Running (1 min)
L12: Jump front & back (20x)

Units: Acceleration (m/s^2), gyroscope (deg/s), magnetic field (local), ECG (mV) *

## Acknowledgement

Banos, O., Garcia, R., Holgado, J. A., Damas, M., Pomares, H., Rojas, I., Saez, A., Villalonga, C. mHealthDroid: a novel framework for agile development of mobile health applications. Proceedings of the 6th International Work-conference on Ambient Assisted Living and Active Ageing (IWAAL 2014), Belfast, Northern Ireland, December 2-5, (2014).

Nguyen, L. T., Zeng, M., Tague, P., Zhang, J. (2015). Recognizing New Activities with Limited Training Data. In IEEE International Symposium on Wearable Computers (ISWC).

Banos, O., Garcia, R., Holgado, J. A., Damas, M., Pomares, H., Rojas, I., Saez, A., Villalonga, C. mHealthDroid: a novel framework for agile development of mobile health applications. Proceedings of the 6th International Work-conference on Ambient Assisted Living an Active Ageing (IWAAL 2014), Belfast, Northern Ireland, December 2-5, (2014).

Banos, O., Villalonga, C., Garcia, R., Saez, A., Damas, M., Holgado, J. A., Lee, S., Pomares, H., Rojas, I. Design, implementation and validation of a novel open framework for agile development of mobile health applications. BioMedical Engineering OnLine, vol. 14, no. S2:S6, pp. 1-20 (2015).

We recommend referring to this dataset as the 'MHEALTH dataset' in publications.
