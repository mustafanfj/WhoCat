# E12a shortcut tests (TAR@FAR=1%, test cats; change = changed minus clean, points)

| model                 | protocol   | test               |   tar_clean |   tar_changed |   change |     lo |     hi | significant   |
|:----------------------|:-----------|:-------------------|------------:|--------------:|---------:|-------:|-------:|:--------------|
| petface               | P1         | background removed |       86.72 |         85.92 |    -0.8  |  -1.03 |  -0.61 | True          |
| petface               | P1         | lighting changed   |       86.72 |         80.89 |    -5.83 |  -6.21 |  -5.47 | True          |
| petface               | P1         | night-style        |       86.72 |         62.56 |   -24.16 | -24.74 | -23.63 | True          |
| petface               | P3         | background removed |       59.13 |         56.29 |    -2.84 |  -3.29 |  -2.35 | True          |
| petface               | P3         | lighting changed   |       59.13 |         47.23 |   -11.9  | -12.53 | -11.17 | True          |
| petface               | P3         | night-style        |       59.13 |         31.26 |   -27.87 | -28.7  | -26.91 | True          |
| arc_random_best       | P1         | background removed |       92.67 |         89.87 |    -2.8  |  -3.01 |  -2.58 | True          |
| arc_random_best       | P1         | lighting changed   |       92.67 |         90.47 |    -2.21 |  -2.4  |  -2.03 | True          |
| arc_random_best       | P1         | night-style        |       92.67 |         73.34 |   -19.33 | -19.8  | -18.84 | True          |
| arc_random_best       | P3         | background removed |       69.98 |         63.25 |    -6.72 |  -7.28 |  -6.16 | True          |
| arc_random_best       | P3         | lighting changed   |       69.98 |         65.84 |    -4.14 |  -4.63 |  -3.71 | True          |
| arc_random_best       | P3         | night-style        |       69.98 |         41.54 |   -28.44 | -29.32 | -27.52 | True          |
| arc_coat_best         | P1         | background removed |       88.18 |         86.44 |    -1.74 |  -1.94 |  -1.51 | True          |
| arc_coat_best         | P1         | lighting changed   |       88.18 |         81.39 |    -6.79 |  -7.08 |  -6.43 | True          |
| arc_coat_best         | P1         | night-style        |       88.18 |         25.01 |   -63.17 | -63.9  | -62.49 | True          |
| arc_coat_best         | P3         | background removed |       56.39 |         53.93 |    -2.46 |  -2.88 |  -2.05 | True          |
| arc_coat_best         | P3         | lighting changed   |       56.39 |         45.69 |   -10.7  | -11.39 | -10.18 | True          |
| arc_coat_best         | P3         | night-style        |       56.39 |         10.24 |   -46.15 | -46.89 | -44.91 | True          |
| arc_coat_door_best    | P1         | background removed |       91.82 |         88.94 |    -2.88 |  -3.08 |  -2.69 | True          |
| arc_coat_door_best    | P1         | lighting changed   |       91.82 |         90.88 |    -0.94 |  -1.08 |  -0.83 | True          |
| arc_coat_door_best    | P1         | night-style        |       91.82 |         88.98 |    -2.84 |  -3.05 |  -2.66 | True          |
| arc_coat_door_best    | P3         | background removed |       74.08 |         67.17 |    -6.9  |  -7.29 |  -6.3  | True          |
| arc_coat_door_best    | P3         | lighting changed   |       74.08 |         72.72 |    -1.36 |  -1.64 |  -0.99 | True          |
| arc_coat_door_best    | P3         | night-style        |       74.08 |         70.31 |    -3.76 |  -4.06 |  -3.23 | True          |
| arc_random_door_best  | P1         | background removed |       90.61 |         88.86 |    -1.75 |  -1.97 |  -1.53 | True          |
| arc_random_door_best  | P1         | lighting changed   |       90.61 |         89.12 |    -1.49 |  -1.67 |  -1.28 | True          |
| arc_random_door_best  | P1         | night-style        |       90.61 |         86.99 |    -3.61 |  -3.81 |  -3.38 | True          |
| arc_random_door_best  | P3         | background removed |       66.94 |         63.2  |    -3.74 |  -4.13 |  -3.37 | True          |
| arc_random_door_best  | P3         | lighting changed   |       66.94 |         64.56 |    -2.38 |  -2.74 |  -2.06 | True          |
| arc_random_door_best  | P3         | night-style        |       66.94 |         61.95 |    -4.99 |  -5.45 |  -4.47 | True          |
| arc_coat_door_bg_best | P1         | background removed |       91.15 |         90.99 |    -0.16 |  -0.24 |  -0.09 | True          |
| arc_coat_door_bg_best | P1         | lighting changed   |       91.15 |         89.96 |    -1.19 |  -1.32 |  -1.03 | True          |
| arc_coat_door_bg_best | P1         | night-style        |       91.15 |         87.96 |    -3.19 |  -3.42 |  -2.99 | True          |
| arc_coat_door_bg_best | P3         | background removed |       73.43 |         73.19 |    -0.23 |  -0.53 |  -0.05 | True          |
| arc_coat_door_bg_best | P3         | lighting changed   |       73.43 |         71.78 |    -1.65 |  -2.07 |  -1.45 | True          |
| arc_coat_door_bg_best | P3         | night-style        |       73.43 |         69.58 |    -3.85 |  -4.26 |  -3.45 | True          |