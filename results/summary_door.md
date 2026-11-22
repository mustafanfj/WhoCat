# E10 the whole door (valB report homes; tuned on valA)

3593 report homes, 3 look-alike neighbours each. 95% CI from a home-level bootstrap.

| step   | system                                          | model              |   photos | probe   |   owner_accept |   owner_lo |   owner_hi |   lookalike_admitted |   look_lo |   look_hi |   homes_breached |
|:-------|:------------------------------------------------|:-------------------|---------:|:--------|---------------:|-----------:|-----------:|---------------------:|----------:|----------:|-----------------:|
| A      | Baseline model, 1 photo, random-tuned threshold | petface            |        1 | clean   |          85.64 |      84.27 |      86.56 |                20.3  |     19.57 |     20.99 |            58.75 |
| B      | Our model, 1 photo, random-tuned threshold      | arc_coat_door_best |        1 | clean   |          91.98 |      91.15 |      92.74 |                20.11 |     19.46 |     20.86 |            61.23 |
| C      | + threshold tuned on look-alikes                | arc_coat_door_best |        1 | clean   |          73.45 |      72    |      74.73 |                 1.04 |      0.85 |      1.22 |             4.09 |
| D      | + neighbour memory                              | arc_coat_door_best |        1 | clean   |          72.25 |      70.78 |      73.51 |                 0.27 |      0.19 |      0.34 |             1.61 |
| E      | + 3 enrolment photos                            | arc_coat_door_best |        3 | clean   |          82.3  |      81.24 |      83.53 |                 0.39 |      0.29 |      0.48 |             1.95 |
| F      | Baseline model with the full system             | petface            |        3 | clean   |          67.69 |      66.35 |      69.47 |                 0.46 |      0.36 |      0.57 |             2.45 |
| A      | Baseline model, 1 photo, random-tuned threshold | petface            |        1 | door    |          60.37 |      58.7  |      62.15 |                11.01 |     10.51 |     11.49 |            44.42 |
| B      | Our model, 1 photo, random-tuned threshold      | arc_coat_door_best |        1 | door    |          86.22 |      85.22 |      87.59 |                15.39 |     14.84 |     15.99 |            53.74 |
| C      | + threshold tuned on look-alikes                | arc_coat_door_best |        1 | door    |          58.84 |      57.36 |      60.59 |                 0.62 |      0.48 |      0.74 |             2.67 |
| D      | + neighbour memory                              | arc_coat_door_best |        1 | door    |          58    |      56.33 |      59.39 |                 0.19 |      0.13 |      0.26 |             1.22 |
| E      | + 3 enrolment photos                            | arc_coat_door_best |        3 | door    |          64.93 |      63.48 |      66.47 |                 0.28 |      0.21 |      0.36 |             1.45 |
| F      | Baseline model with the full system             | petface            |        3 | door    |          23.38 |      22.07 |      24.83 |                 0.14 |      0.09 |      0.2  |             0.92 |