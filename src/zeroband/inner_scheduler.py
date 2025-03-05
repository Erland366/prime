class BaseScheduler:
    def __init__(
        self, 
        warmup: int = 0, 
        inner_warmup_steps: int = 1,
        reverse: bool = False
    ):
        self.warmup = warmup
        self.inner_warmup_steps = inner_warmup_steps
        self.reverse = reverse

    def get_inner_steps(self):
        if self.warmup and self.curr_step < self.inner_warmup_steps:
            return 1

        return self._inner_method_steps()

    def _inner_method_steps(self) -> int:
        raise NotImplementedError

    def step(self):
        self.curr_step += 1
        self.curr_inner_steps = self.get_inner_steps()

class ContinuousInnerStepScheduler(BaseScheduler):
    def __init__(
        self,
        lower_steps: int,
        upper_steps: int,
        total_steps: int,
        reverse: bool = False,
        warmup: int = 0,
        inner_warmup_steps: int = 1
    ):
        super().__init__(warmup, inner_warmup_steps, reverse)
        self.lower_steps = lower_steps
        self.upper_steps = upper_steps
        self.total_steps = total_steps
        self.curr_step = 0
        self.curr_inner_steps = self.lower_steps
        self.increment = (upper_steps - lower_steps) / total_steps

    def _inner_method_steps(self) -> int:
        if not self.reverse:
            return int(min(max(self.lower_steps + (self.increment * self.curr_step), self.lower_steps), self.upper_steps))
        else:
            return int(max(min(self.lower_steps + (self.increment * self.curr_step), self.lower_steps), self.upper_steps))

class BinnedInnerStepScheduler(BaseScheduler):
    def __init__(
        self,
        lower_steps: int,
        upper_steps: int,
        total_steps: int,
        bin_size: int | None = None,
        num_bins: int | None = None,
        reverse: bool = False,
        warmup: int = 0,
        inner_warmup_steps: int = 1
    ):
        super().__init__(warmup, inner_warmup_steps, reverse)
        self.lower_steps = lower_steps
        self.upper_steps = upper_steps
        self.total_steps = total_steps
        self.curr_step = 0
        self.curr_inner_steps = self.lower_steps

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


        self.increment = (upper_steps - lower_steps) / (self.num_bins - 1) if self.num_bins > 1 else 0

    def _inner_method_steps(self):
        bin_index = self.curr_step // self.bin_size
        if not self.reverse:
            return int(min(max(self.lower_steps + (self.increment * bin_index), self.lower_steps), self.upper_steps))
        else:
            return int(max(min(self.lower_steps + (self.increment * bin_index), self.lower_steps) , self.upper_steps))
