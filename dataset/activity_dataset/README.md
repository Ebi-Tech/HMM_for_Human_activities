# Activity dataset

Phone accelerometer recordings (Total Acceleration, gravity included) of four activities:
standing, walking, jumping and still. Recorded with the Sensor Logger app on one Android phone
held at waist level. The phone has no gyroscope.

- Real sampling rate of the phone: about 27.5 Hz (asked for 50 Hz, the phone gave less).
- Every clip was converted to an even 25 Hz grid.
- Every clip is 5 seconds long (125 samples) and was cut from a longer recording.
- train/ and test/ are made from different recordings, so the test clips are unseen.
- manifest.csv lists every file and the recording it came from.

Columns: timestamp (seconds, starts at 0), acc_x, acc_y, acc_z, activity.
