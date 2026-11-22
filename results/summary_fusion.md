# E9 score fusion

w = weight of the main model, tuned on val P3 (TAR@FAR=1%).

| a                  | b                      |   w | probe   | protocol   |   tar_a |   tar_b |   tar_fused |   gain |    lo |    hi | significant   |
|:-------------------|:-----------------------|----:|:--------|:-----------|--------:|--------:|------------:|-------:|------:|------:|:--------------|
| arc_coat_door_best | petface                | 1   | clean   | P1         |   91.82 |   86.72 |       91.82 |   0    |  0    |  0    | False         |
| arc_coat_door_best | petface                | 1   | clean   | P2         |   88.45 |   79.01 |       88.45 |   0    |  0    |  0    | False         |
| arc_coat_door_best | petface                | 1   | clean   | P3         |   74.08 |   59.13 |       74.08 |   0    |  0    |  0    | False         |
| arc_coat_door_best | petface                | 1   | door    | P1         |   86.43 |   60.18 |       86.43 |   0    |  0    |  0    | False         |
| arc_coat_door_best | petface                | 1   | door    | P2         |   81.27 |   48.27 |       81.27 |   0    |  0    |  0    | False         |
| arc_coat_door_best | petface                | 1   | door    | P3         |   65.11 |   31.86 |       65.11 |   0    |  0    |  0    | False         |
| arc_coat_door_best | petface                | 1   | gray    | P1         |   88.98 |   62.56 |       88.98 |   0    |  0    |  0    | False         |
| arc_coat_door_best | petface                | 1   | gray    | P2         |   84.96 |   49.56 |       84.96 |   0    |  0    |  0    | False         |
| arc_coat_door_best | petface                | 1   | gray    | P3         |   70.31 |   31.26 |       70.31 |   0    |  0    |  0    | False         |
| arc_coat_door_best | student_coat_door_best | 1   | clean   | P1         |   91.82 |   87.06 |       91.82 |   0    |  0    |  0    | False         |
| arc_coat_door_best | student_coat_door_best | 1   | clean   | P2         |   88.45 |   79.89 |       88.45 |   0    |  0    |  0    | False         |
| arc_coat_door_best | student_coat_door_best | 1   | clean   | P3         |   74.08 |   57.56 |       74.08 |   0    |  0    |  0    | False         |
| arc_coat_door_best | student_coat_door_best | 1   | door    | P1         |   86.43 |   76.79 |       86.43 |   0    |  0    |  0    | False         |
| arc_coat_door_best | student_coat_door_best | 1   | door    | P2         |   81.27 |   65.74 |       81.27 |   0    |  0    |  0    | False         |
| arc_coat_door_best | student_coat_door_best | 1   | door    | P3         |   65.11 |   42.87 |       65.11 |   0    |  0    |  0    | False         |
| arc_coat_door_best | student_coat_door_best | 1   | gray    | P1         |   88.98 |   82.57 |       88.98 |   0    |  0    |  0    | False         |
| arc_coat_door_best | student_coat_door_best | 1   | gray    | P2         |   84.96 |   73.53 |       84.96 |   0    |  0    |  0    | False         |
| arc_coat_door_best | student_coat_door_best | 1   | gray    | P3         |   70.31 |   51.01 |       70.31 |   0    |  0    |  0    | False         |
| arc_coat_door_best | arc_random_best        | 0.9 | clean   | P1         |   91.82 |   92.67 |       92.2  |   0.38 |  0.3  |  0.45 | True          |
| arc_coat_door_best | arc_random_best        | 0.9 | clean   | P2         |   88.45 |   88.61 |       88.78 |   0.33 |  0.24 |  0.41 | True          |
| arc_coat_door_best | arc_random_best        | 0.9 | clean   | P3         |   74.08 |   69.98 |       74    |  -0.08 | -0.16 |  0.17 | False         |
| arc_coat_door_best | arc_random_best        | 0.9 | door    | P1         |   86.43 |   69.37 |       86.53 |   0.1  | -0.02 |  0.18 | False         |
| arc_coat_door_best | arc_random_best        | 0.9 | door    | P2         |   81.27 |   59.12 |       80.98 |  -0.28 | -0.38 | -0.16 | True          |
| arc_coat_door_best | arc_random_best        | 0.9 | door    | P3         |   65.11 |   39.72 |       64.31 |  -0.8  | -1.02 | -0.62 | True          |
| arc_coat_door_best | arc_random_best        | 0.9 | gray    | P1         |   88.98 |   73.34 |       89.09 |   0.11 |  0.03 |  0.19 | True          |
| arc_coat_door_best | arc_random_best        | 0.9 | gray    | P2         |   84.96 |   64.03 |       84.9  |  -0.06 | -0.15 |  0.02 | False         |
| arc_coat_door_best | arc_random_best        | 0.9 | gray    | P3         |   70.31 |   41.54 |       69.94 |  -0.37 | -0.59 | -0.26 | True          |