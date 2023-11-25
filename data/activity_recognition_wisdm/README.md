# Activity Recognition WISDM dataset

The dataset, acquired from WISDM Lab, consists of data collected from 36 different users performing six types of human activities (ascending and descending stairs, sitting, walking, jogging, and standing) for specific periods of time.

These data were acquired from accelerometers, which are able of detecting the orientation of the device measuring the acceleration along the three different dimensions. They were collected using a sample rate of 20 Hz (1 sample every 50 millisecond) that is equivalent to 20 samples per second.

These time-series data can be used to perform various techniques, such as human activity recognition.


## Columns

user: the user who acquired the data (integer from 1 to 36).

activity: the activity that the user was carrying out. It could be:
1. walking
2. Jogging
3. Sitting
4. Standing
5. Upstairs
6. Downstairs.

timestamp: generally the phone's uptime in nanoseconds.

x-axis: The acceleration in the x direction as measured by the android phone's accelerometer.
Floating-point values between -20 and 20. A value of 10 = 1g = 9.81 m/s^2, and 0 = no acceleration.
The acceleration recorded includes gravitational acceleration toward the center of the Earth, so that when the phone is at rest on a flat surface the vertical axis will register +-10.

y-axis: same as x-axis, but along y axis.

z-axis: same as x-axis, but along z axis.

## Acknowledgement

Data were fetched from the WISDM dataset website, and they were cleaned, deleting missing values, replacing inconsistent strings and converting the dataset to csv.

Jeffrey W. Lockhart, Tony Pulickal, and Gary M. Weiss (2012).
"Applications of Mobile Activity Recognition,"
Proceedings of the ACM UbiComp International Workshop
on Situation, Activity, and Goal Awareness, Pittsburgh,
PA.

Gary M. Weiss and Jeffrey W. Lockhart (2012). "The Impact of
Personalization on Smartphone-Based Activity Recognition,"
Proceedings of the AAAI-12 Workshop on Activity Context
Representation: Techniques and Languages, Toronto, CA.

Jennifer R. Kwapisz, Gary M. Weiss and Samuel A. Moore (2010).
"Activity Recognition using Cell Phone Accelerometers,"
Proceedings of the Fourth International Workshop on
Knowledge Discovery from Sensor Data (at KDD-10), Washington
DC.