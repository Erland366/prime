from typing import Protocol, Optional

class InnerStepScheduler(Protocol):
    def __init__(self): ...

    def get_inner_steps(self): ...

    def step(self): ...

class ContinuousInnerStepScheduler:
    def __init__(
        self, 
        start_inner_steps: int, 
        end_inner_steps: int, 
        total_steps: int
    ):
        self.start_inner_steps = start_inner_steps
        self.end_inner_steps = end_inner_steps
        self.total_steps = total_steps
        self.curr_step = 0
        self.curr_inner_steps = self.start_inner_steps
        self.increment = (end_inner_steps - start_inner_steps) / total_steps

    def get_inner_steps(self):
        return int(min(self.start_inner_steps + (self.increment * self.curr_step), self.end_inner_steps))
        
    def step(self):
        self.curr_step += 1
        self.current_inner_steps = self.get_inner_steps()

class BinnedInnerStepScheduler:
    def __init__(
        self,
        start_inner_steps: int,
        end_inner_steps: int,
        total_steps: int,
        bin_size: int | None = None,
        num_bins: int | None = None,
    ):
        self.start_inner_steps = start_inner_steps
        self.end_inner_steps = end_inner_steps
        self.total_steps = total_steps
        self.curr_step = 0
        self.curr_inner_steps = self.start_inner_steps

        if bin_size is not None and num_bins is None:
            self.bin_size = bin_size
            self.num_bins = (total_steps + bin_size - 1) // bin_size
        elif num_bins is not None and bin_size is None:
            self.num_bins = num_bins
            self.bin_size = (total_steps + num_bins - 1) // num_bins
        elif bin_size is not None and num_bins is not None:
            # Decide which to prioritize or raise an error.
            # For now, let's prioritize bin_size and recalculate num_bins
            self.bin_size = bin_size
            self.num_bins = (total_steps + bin_size - 1) // bin_size
            print("Warning: Both bin_size and num_bins provided. Prioritizing bin_size and recalculating num_bins.")
        else:
            raise ValueError("Either `bin_size` or `num_bins` must be provided.")


        self.increment = (end_inner_steps - start_inner_steps) / (self.num_bins - 1) if self.num_bins > 1 else 0

    def get_inner_steps(self):
        bin_index = self.curr_step // self.bin_size
        return int(min(self.start_inner_steps + (self.increment * bin_index), self.end_inner_steps))

    def step(self):
        self.curr_step += 1
        self.curr_inner_steps = self.get_inner_steps()