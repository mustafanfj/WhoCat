# WhoCat results

## TAR @ FAR=1% (test cats)

| model                    |   ('P1', 'clean') |   ('P1', 'door') |   ('P1', 'gray') |   ('P2', 'clean') |   ('P2', 'door') |   ('P2', 'gray') |   ('P3', 'clean') |   ('P3', 'door') |   ('P3', 'gray') |
|:-------------------------|------------------:|-----------------:|-----------------:|------------------:|-----------------:|-----------------:|------------------:|-----------------:|-----------------:|
| arc_coat_best            |              88.2 |             43.6 |             25   |              81   |             36.4 |             18.3 |              56.4 |             24.9 |             10.2 |
| arc_coat_door_best       |              91.8 |             86.4 |             89   |              88.4 |             81.3 |             85   |              74.1 |             65.1 |             70.3 |
| arc_coat_door_bg_best    |              91.2 |             84.9 |             88   |              87.6 |             79.8 |             83.8 |              73.4 |             63.9 |             69.6 |
| arc_random_best          |              92.7 |             69.4 |             73.3 |              88.6 |             59.1 |             64   |              70   |             39.7 |             41.5 |
| arc_random_door_best     |              90.6 |             83.6 |             87   |              85.1 |             75.2 |             79.9 |              66.9 |             55.8 |             62   |
| dinov2                   |              64.6 |             38.2 |             51.7 |              46.5 |             22.1 |             31.1 |              26.6 |             10.6 |             14.3 |
| megadesc                 |              45.4 |             14.5 |             15.2 |              32.5 |              8.8 |              8.9 |              18.1 |              4.9 |              3.6 |
| petface                  |              86.7 |             60.2 |             62.6 |              79   |             48.3 |             49.6 |              59.1 |             31.9 |             31.3 |
| student_coat_door_best   |              87.1 |             76.8 |             82.6 |              79.9 |             65.7 |             73.5 |              57.6 |             42.9 |             51   |
| student_random_door_best |              86   |             75   |             82.4 |              74.8 |             57.9 |             68.1 |              51.6 |             36   |             43.9 |

## PetFace official verification AUC (sanity)

| model                    |   auc |
|:-------------------------|------:|
| petface                  | 97.71 |
| megadesc                 | 88.23 |
| dinov2                   | 95.58 |
| arc_random_best          | 98.82 |
| arc_coat_best            | 98.34 |
| arc_coat_door_best       | 98.44 |
| student_coat_door_best   | 97.95 |
| arc_random_door_best     | 98.36 |
| student_random_door_best | 97.51 |
| arc_coat_door_bg_best    | 98.26 |

## Strangers let in at the P1-tuned threshold (target 1%)

| model                    |   ('P1', 'clean') |   ('P1', 'door') |   ('P1', 'gray') |   ('P2', 'clean') |   ('P2', 'door') |   ('P2', 'gray') |   ('P3', 'clean') |   ('P3', 'door') |   ('P3', 'gray') |
|:-------------------------|------------------:|-----------------:|-----------------:|------------------:|-----------------:|-----------------:|------------------:|-----------------:|-----------------:|
| arc_coat_best            |              0.93 |             0.93 |             1.26 |              3.61 |             2.38 |             2.39 |             22.36 |             7.89 |             5.55 |
| arc_coat_door_best       |              0.97 |             0.92 |             0.91 |              3.46 |             3.18 |             3.25 |             18.57 |            14.06 |            15.25 |
| arc_coat_door_bg_best    |              1    |             0.96 |             0.94 |              3.48 |             3.21 |             3.25 |             17.89 |            13.39 |            14.49 |
| arc_random_best          |              0.98 |             0.94 |             0.87 |              3.82 |             2.87 |             2.36 |             22.52 |            10.44 |             8.25 |
| arc_random_door_best     |              0.99 |             0.95 |             0.95 |              4.31 |             4    |             4.22 |             21.42 |            15.84 |            17.74 |
| dinov2                   |              1.21 |             0.27 |             0.46 |              5.6  |             1.35 |             2.29 |             25.23 |             6.25 |            10.83 |
| megadesc                 |              0.98 |             0.2  |             0.19 |              3.62 |             0.64 |             0.47 |             12.93 |             1.76 |             1.41 |
| petface                  |              0.97 |             0.9  |             0.86 |              4.22 |             3.09 |             2.75 |             18.3  |             9.89 |             9.06 |
| student_coat_door_best   |              1.01 |             0.95 |             0.96 |              4.05 |             3.67 |             3.85 |             21.07 |            15.18 |            17.62 |
| student_random_door_best |              0.99 |             1.06 |             1.06 |              5.15 |             5.19 |             5.45 |             22.38 |            17.6  |            20.59 |

## Same-coat strangers let in, by coat (%)

| coat               |   ('arc_coat_best', 'clean') |   ('arc_coat_best', 'gray') |   ('arc_coat_door_best', 'clean') |   ('arc_coat_door_best', 'gray') |   ('arc_coat_door_bg_best', 'clean') |   ('arc_coat_door_bg_best', 'gray') |   ('arc_random_best', 'clean') |   ('arc_random_best', 'gray') |   ('arc_random_door_best', 'clean') |   ('arc_random_door_best', 'gray') |   ('dinov2', 'clean') |   ('dinov2', 'gray') |   ('megadesc', 'clean') |   ('megadesc', 'gray') |   ('petface', 'clean') |   ('petface', 'gray') |   ('student_coat_door_best', 'clean') |   ('student_coat_door_best', 'gray') |   ('student_random_door_best', 'clean') |   ('student_random_door_best', 'gray') |
|:-------------------|-----------------------------:|----------------------------:|----------------------------------:|---------------------------------:|-------------------------------------:|------------------------------------:|-------------------------------:|------------------------------:|------------------------------------:|-----------------------------------:|----------------------:|---------------------:|------------------------:|-----------------------:|-----------------------:|----------------------:|--------------------------------------:|-------------------------------------:|----------------------------------------:|---------------------------------------:|
| black              |                         5.09 |                        4.75 |                              4.77 |                             4.49 |                                 4.85 |                                4.64 |                           5.43 |                          4.38 |                                6.51 |                               7    |                  6.93 |                 3.11 |                    1    |                   0.07 |                   7.21 |                  5.26 |                                  6.05 |                                 6.37 |                                    9.81 |                                  11.2  |
| calico             |                         2.52 |                        0.45 |                              2.44 |                             2.15 |                                 2.54 |                                2.21 |                           2.67 |                          0.71 |                                2.79 |                               2.28 |                  3.35 |                 1.02 |                    1.21 |                   0.22 |                   2.41 |                  0.81 |                                  2.38 |                                 1.87 |                                    2.61 |                                   2.32 |
| gray               |                        23.98 |                       26.6  |                             22.58 |                            21.21 |                                22.62 |                               21.33 |                          29.1  |                         30.51 |                               35.74 |                              36.29 |                 20.39 |                 8.91 |                    2.23 |                   0.16 |                  32.34 |                 36.6  |                                 23.24 |                                24.3  |                                   36.33 |                                  42.3  |
| pied_blackandwhite |                         3.26 |                        1.71 |                              3.38 |                             3.31 |                                 3.19 |                                3.02 |                           3.59 |                          2.4  |                                3.88 |                               3.82 |                  7.35 |                 3.91 |                    2.07 |                   0.12 |                   3.21 |                  2.74 |                                  3.76 |                                 3.58 |                                    4.1  |                                   4.23 |
| seal_point         |                        19.05 |                        4.43 |                             18.23 |                            16.83 |                                17.73 |                               16.35 |                          20.81 |                         13.3  |                               29.59 |                              27.85 |                 31.89 |                19.24 |                    3.28 |                   0.45 |                  25.05 |                 17.73 |                                 17.05 |                                15.99 |                                   30.32 |                                  29.14 |
| tabby_brown        |                         2.32 |                        1.13 |                              1.97 |                             1.87 |                                 2.12 |                                1.94 |                           1.9  |                          0.77 |                                2    |                               1.81 |                  5.15 |                 1.93 |                    6.17 |                   0.54 |                   1.79 |                  0.79 |                                  2.5  |                                 2.47 |                                    2.15 |                                   2.21 |
| tabby_gray         |                         2.12 |                        4.09 |                              1.93 |                             1.71 |                                 1.97 |                                1.69 |                           1.91 |                          2.1  |                                1.93 |                               1.63 |                  3.33 |                 1.13 |                    2.07 |                   0.31 |                   1.82 |                  2.28 |                                  2.28 |                                 2.01 |                                    2.15 |                                   1.83 |
| tabby_orange       |                         3.18 |                        0.45 |                              3.05 |                             2.87 |                                 3.2  |                                3.04 |                           3.07 |                          0.74 |                                3.15 |                               2.94 |                  3.57 |                 0.64 |                    6.92 |                   1.02 |                   3.6  |                  0.6  |                                  3.94 |                                 3.07 |                                    3.79 |                                   3.69 |
| tortoiseshell      |                         5.44 |                        1.73 |                              4.72 |                             4.69 |                                 5.23 |                                5.35 |                           7.67 |                          3.12 |                                9.51 |                               9.33 |                  0.49 |                 0.02 |                    0.21 |                   0.02 |                   8.77 |                  4.33 |                                  4.74 |                                 4.96 |                                   11.29 |                                  12.19 |
| white              |                         7.57 |                        3.37 |                              8.61 |                             8.13 |                                 7.68 |                                7.4  |                          10.36 |                          5.75 |                               11.9  |                              12.85 |                 14.19 |                 5.73 |                    7.45 |                   2.08 |                  12.82 |                  7.59 |                                 10.24 |                                10.79 |                                   15.33 |                                  18.66 |

## Face ID: owner rejected at FAR=1% (%)

| model                    |   (1, 'clean') |   (1, 'door') |   (1, 'gray') |   (3, 'clean') |   (3, 'door') |   (3, 'gray') |
|:-------------------------|---------------:|--------------:|--------------:|---------------:|--------------:|--------------:|
| arc_coat_best            |          30.15 |         66.67 |         78.42 |          18.46 |         60.57 |         81.15 |
| arc_coat_door_best       |          16.1  |         22.85 |         19.07 |           7.94 |         13.68 |         10.73 |
| arc_coat_door_bg_best    |          16.97 |         24.02 |         20.18 |           8.25 |         14.75 |         11.8  |
| arc_random_best          |          18.07 |         46.89 |         37.07 |           8.75 |         38.8  |         31.78 |
| arc_random_door_best     |          20.73 |         29.82 |         24.67 |          10.97 |         19.35 |         15.17 |
| dinov2                   |          60.22 |         80.49 |         70.88 |          52.48 |         77.91 |         69.27 |
| megadesc                 |          72.57 |         91.4  |         91.18 |          66.34 |         90.18 |         91.88 |
| petface                  |          28.11 |         55.84 |         53.89 |          17.47 |         49.11 |         50.34 |
| student_coat_door_best   |          28.52 |         40.94 |         33.77 |          17.68 |         32.25 |         24.44 |
| student_random_door_best |          33.32 |         47.23 |         39.1  |          22.01 |         37.73 |         29.74 |

## E5 stranger-aware enrolment (%, recurring look-alike neighbours)

| model                    | probe   |   margin |   owner_frr_base |   owner_frr_aware |   stranger_far_base |   stranger_far_aware |   homes_breached_base |   homes_breached_aware |
|:-------------------------|:--------|---------:|-----------------:|------------------:|--------------------:|---------------------:|----------------------:|-----------------------:|
| petface                  | clean   |    -0.14 |            13.37 |             15.35 |               23.13 |                 9.09 |                 47.87 |                  21.67 |
| petface                  | door    |    -0.14 |            39.66 |             40.65 |               12.8  |                 7.93 |                 29.95 |                  19.23 |
| megadesc                 | clean   |    -0.1  |            54.75 |             56.92 |               16.24 |                10.86 |                 32.97 |                  25.11 |
| megadesc                 | door    |    -0.1  |            92.67 |             92.97 |                2.38 |                 1.94 |                  6.47 |                   5.34 |
| dinov2                   | clean   |    -0.08 |            33.11 |             35.09 |               29.96 |                20.95 |                 51.67 |                  41.66 |
| dinov2                   | door    |    -0.08 |            74.68 |             75.05 |                8.09 |                 6.91 |                 20    |                  17.49 |
| arc_random_best          | clean   |    -0.16 |             7.34 |              9.92 |               28.64 |                 8.33 |                 57.39 |                  20.61 |
| arc_random_best          | door    |    -0.16 |            30.55 |             31.63 |               13.99 |                 8.32 |                 31.85 |                  19.76 |
| arc_coat_best            | clean   |    -0.16 |            12.14 |             14.66 |               28.46 |                11.78 |                 57.4  |                  28.7  |
| arc_coat_best            | door    |    -0.16 |            56.47 |             57.27 |               10.28 |                 7.36 |                 25.07 |                  18.39 |
| arc_coat_door_best       | clean   |    -0.12 |             8.27 |             11.03 |               23.98 |                 5.84 |                 50.5  |                  14.58 |
| arc_coat_door_best       | door    |    -0.12 |            13.44 |             15.72 |               18.68 |                 6.41 |                 41.44 |                  15.91 |
| student_coat_door_best   | clean   |    -0.16 |            12.91 |             15.41 |               26.51 |                10.85 |                 54.07 |                  26.29 |
| student_coat_door_best   | door    |    -0.16 |            22.64 |             24.28 |               19.61 |                11.28 |                 42.92 |                  26.83 |
| arc_random_door_best     | clean   |    -0.14 |             9.42 |             12.12 |               27.19 |                 8.09 |                 54.8  |                  19.86 |
| arc_random_door_best     | door    |    -0.14 |            16.11 |             18.11 |               20.48 |                 8.85 |                 43.78 |                  21.28 |
| student_random_door_best | clean   |    -0.18 |            13.97 |             16.14 |               27.66 |                13.5  |                 54.77 |                  31.4  |
| student_random_door_best | door    |    -0.18 |            23.11 |             24.51 |               22.03 |                14.43 |                 45.91 |                  32.61 |
| arc_coat_door_bg_best    | clean   |    -0.12 |             8.85 |             11.63 |               23.25 |                 5.95 |                 49.36 |                  14.93 |
| arc_coat_door_bg_best    | door    |    -0.12 |            14.79 |             16.98 |               17.94 |                 6.51 |                 39.99 |                  16.08 |

## Paired differences in TAR@FAR=1% (points, 95% CI)

| test                        | protocol   | probe   |   diff_pts |     lo |     hi | significant   |
|:----------------------------|:-----------|:--------|-----------:|-------:|-------:|:--------------|
| H1 coat vs random           | P1         | clean   |      -4.49 |  -4.82 |  -4.25 | True          |
| H1 coat vs random           | P2         | clean   |      -7.65 |  -8.05 |  -7.34 | True          |
| H1 coat vs random           | P3         | clean   |     -13.59 | -14.19 | -12.96 | True          |
| door aug                    | P1         | clean   |       3.64 |   3.31 |   3.92 | True          |
| door aug                    | P1         | door    |      42.84 |  42.25 |  43.39 | True          |
| door aug                    | P1         | gray    |      63.97 |  63.33 |  64.65 | True          |
| door aug                    | P3         | clean   |      17.69 |  16.78 |  18.27 | True          |
| door aug                    | P3         | door    |      40.25 |  39.37 |  40.99 | True          |
| door aug                    | P3         | gray    |      60.07 |  59.1  |  60.98 | True          |
| ours vs PetFace             | P1         | clean   |       1.46 |   1.1  |   1.8  | True          |
| ours vs PetFace             | P2         | clean   |       1.96 |   1.6  |   2.34 | True          |
| ours vs PetFace             | P3         | clean   |      -2.74 |  -3.37 |  -2.07 | True          |
| ours+door vs PetFace        | P1         | clean   |       5.1  |   4.77 |   5.34 | True          |
| ours+door vs PetFace        | P1         | door    |      26.24 |  25.71 |  26.78 | True          |
| ours+door vs PetFace        | P1         | gray    |      26.42 |  25.78 |  27.08 | True          |
| ours+door vs PetFace        | P2         | clean   |       9.44 |   9.12 |   9.83 | True          |
| ours+door vs PetFace        | P2         | door    |      33    |  32.35 |  33.67 | True          |
| ours+door vs PetFace        | P2         | gray    |      35.4  |  34.79 |  36.02 | True          |
| ours+door vs PetFace        | P3         | clean   |      14.94 |  14.17 |  15.47 | True          |
| ours+door vs PetFace        | P3         | door    |      33.25 |  32.58 |  34.06 | True          |
| ours+door vs PetFace        | P3         | gray    |      39.05 |  38.18 |  39.92 | True          |
| aug on random batches       | P1         | clean   |      -2.07 |  -2.28 |  -1.89 | True          |
| aug on random batches       | P1         | door    |      14.23 |  13.82 |  14.79 | True          |
| aug on random batches       | P1         | gray    |      13.65 |  13.14 |  14.11 | True          |
| aug on random batches       | P2         | clean   |      -3.5  |  -3.8  |  -3.24 | True          |
| aug on random batches       | P2         | door    |      16.09 |  15.56 |  16.58 | True          |
| aug on random batches       | P2         | gray    |      15.9  |  15.36 |  16.49 | True          |
| aug on random batches       | P3         | clean   |      -3.03 |  -3.52 |  -2.58 | True          |
| aug on random batches       | P3         | door    |      16.07 |  15.35 |  16.67 | True          |
| aug on random batches       | P3         | gray    |      20.41 |  19.62 |  21.11 | True          |
| H1 with aug: coat vs random | P1         | clean   |       1.21 |   1.03 |   1.42 | True          |
| H1 with aug: coat vs random | P1         | door    |       2.82 |   2.52 |   3.08 | True          |
| H1 with aug: coat vs random | P1         | gray    |       1.99 |   1.75 |   2.19 | True          |
| H1 with aug: coat vs random | P2         | clean   |       3.33 |   3.09 |   3.6  | True          |
| H1 with aug: coat vs random | P2         | door    |       6.05 |   5.67 |   6.37 | True          |
| H1 with aug: coat vs random | P2         | gray    |       5.04 |   4.73 |   5.3  | True          |
| H1 with aug: coat vs random | P3         | clean   |       7.13 |   6.52 |   7.57 | True          |
| H1 with aug: coat vs random | P3         | door    |       9.32 |   8.92 |   9.77 | True          |
| H1 with aug: coat vs random | P3         | gray    |       8.36 |   7.84 |   8.94 | True          |
| random+aug vs PetFace       | P1         | clean   |       3.89 |   3.57 |   4.17 | True          |
| random+aug vs PetFace       | P1         | door    |      23.42 |  22.91 |  23.96 | True          |
| random+aug vs PetFace       | P1         | gray    |      24.43 |  23.78 |  25.09 | True          |
| random+aug vs PetFace       | P2         | clean   |       6.11 |   5.74 |   6.48 | True          |
| random+aug vs PetFace       | P2         | door    |      26.94 |  26.32 |  27.63 | True          |
| random+aug vs PetFace       | P2         | gray    |      30.37 |  29.76 |  30.98 | True          |
| random+aug vs PetFace       | P3         | clean   |       7.81 |   7.22 |   8.39 | True          |
| random+aug vs PetFace       | P3         | door    |      23.93 |  23.19 |  24.66 | True          |
| random+aug vs PetFace       | P3         | gray    |      30.7  |  29.92 |  31.5  | True          |
| background swap (E12b)      | P1         | clean   |      -0.67 |  -0.86 |  -0.5  | True          |
| background swap (E12b)      | P1         | door    |      -1.54 |  -1.75 |  -1.24 | True          |
| background swap (E12b)      | P1         | gray    |      -1.02 |  -1.22 |  -0.8  | True          |
| background swap (E12b)      | P2         | clean   |      -0.88 |  -1.1  |  -0.65 | True          |
| background swap (E12b)      | P2         | door    |      -1.45 |  -1.76 |  -1.21 | True          |
| background swap (E12b)      | P2         | gray    |      -1.14 |  -1.36 |  -0.91 | True          |
| background swap (E12b)      | P3         | clean   |      -0.65 |  -1.09 |  -0.13 | True          |
| background swap (E12b)      | P3         | door    |      -1.26 |  -1.67 |  -0.9  | True          |
| background swap (E12b)      | P3         | gray    |      -0.73 |  -1.21 |  -0.35 | True          |
| small: coat vs random (aug) | P1         | clean   |       1.02 |   0.7  |   1.26 | True          |
| small: coat vs random (aug) | P1         | door    |       1.75 |   1.36 |   2.11 | True          |
| small: coat vs random (aug) | P1         | gray    |       0.17 |  -0.14 |   0.52 | False         |
| small: coat vs random (aug) | P3         | clean   |       5.95 |   5.39 |   6.54 | True          |
| small: coat vs random (aug) | P3         | door    |       6.88 |   6.46 |   7.44 | True          |
| small: coat vs random (aug) | P3         | gray    |       7.12 |   6.4  |   7.59 | True          |
