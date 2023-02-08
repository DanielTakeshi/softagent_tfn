## General notes

Trying to figure out what I should focus on next in ToolFlow:
1. Get the sync issue where the timestamps are aligned manually
2. Try to get ```ApproximateTimeSynchronizer``` to work again

Assuming the sync issues work, what do we work on next:
1. Scaling up the target flow
2. Plot statistics of the flow -> Make a ```visualization.py``` type of interface as well
3. Composing of v04 demonstrations

# Trying to figure out an issue with the sync

this is the dist_idx list:
[  1   2   3   4   5   6   7   8   9  10  11  13  14  15  16  17  17  19
  19  20  21  22  22  26  27  28  29  30  30  32  33  34  35  37  37  38
  39  40  41  42  43  43  45  46  46  46  46  52  52  54  54  56  56  58
  59  59  61  61  63  63  65  67  67  69  69  71  71  73  73  73  73  73
  73  73  81  81  84  84  84  86  86  86  89  89  91  91  94  94  94  96
  96  98  98  98  98  98  98  98  98  98  98 110 110 110 114 114 114 114
 117 117 117 120 120 120 124 124 124 127 127 127 127 127 127 127 127 127
 127 127 127 127]

threre's a lot of sus in the 22 - 26 items

basically, for the idxs occupied by these values, the system maps them to the depths at these values

So IMG idxs: 23, 24, 25, 26
dpts: 23, 24, 25 not mapped to anything



