SUBSET_SELECTION_POOL = list(
    range(1000)
)  # The current optimal subset selection pool (the 1000-point data pool from which 100-point optimal subset is selected from) is created via `subset_selection_pool = sst2["train"].shuffle(seed=0).select(range(1000))`
