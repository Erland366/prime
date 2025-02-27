import torch
import concurrent.futures as futures

class ParameterServer:
    def __init__(
        self, 
        server_model, 
        optimizer_state, 
        update_queue, 
        model_queue, 
        outer_opt_name, 
        outer_opt_kwargs, 
        diloco_config, 
        elastic_device_mesh_config,
        num_workers,
    ):
        self.server_model = server_model
        self.optimizer_state = optimizer_state
        self.update_queue = update_queue
        self.model_queue = model_queue
        self.outer_opt_name = outer_opt_name
        self.outer_opt_kwargs = outer_opt_kwargs
        self.diloco_config = diloco_config
        self.num_workers = num_workers
        self.elastic_device_mesh_config = elastic_device_mesh_config
        self.server_optimizer = None # Initialize later in `run`
        self.diloco_ps = None # Initialize later in `run`
        self.pseudo_gradient_accumulator = []

    def run(self):
        DEVICE_SERVER = "cpu" # For now it'll reside on CPU
        self.DEVICE_SERVER = DEVICE_SERVER
        """Main loop for the Parameter Server process."""
        print(f"[PS Process] Starting DiLoCo Parameter Server on {DEVICE_SERVER}")
        self.server_model.to(DEVICE_SERVER)
        self.server_model.share_memory()

        # PS lives in different thread, therefore we can't just send the live object
        # Instead, we need to send the serializable state dict from the main process
        # And load it from here
        self.server_optimizer = get_optimizer(self.server_model.parameters(), self.outer_opt_name, self.outer_opt_kwargs)
        self.server_optimizer.load_state_dict(self.optimizer_state)

        # --- Initialize DiLoCo on the PS ---
        elastic_device_mesh = ElasticDeviceMesh(**self.elastic_device_mesh_config)
        self.diloco_ps = Diloco(self.diloco_config, self.server_model, elastic_device_mesh, self.diloco_config.experiment)
        self.diloco_ps.param_list_cpu = [p.to(DEVICE_SERVER) for p in self.diloco_ps.param_list_cpu]
        self.diloco_ps.outer_optimizer.to(DEVICE_SERVER)


        while True:
            update_data = self.update_queue.get()
            if update_data == "STOP":
                print("[PS Process] Received STOP signal. Exiting.")
                break
            self.process_update(update_data)

    def process_update(self, update_data):
        outer_param_update = update_data["outer_param_update"]
        model_id = update_data["model_id"]
        num_inner_steps = update_data["num_inner_steps"]
        sync_weight = update_data["sync_weight"]

        self.pseudo_gradient_accumulator.append(outer_param_updatea)

        if len(self.pseudo_gradient_accumulator) > self.num_workers:
            self.aggregate_and_apply_outer_update()
            self.pseudo_gradient_accumulator = []

            for _ in range(self.num_workers):
                self.model_queue.put({
                    "model_id" : model_id + 1,
                    "server_model_state": self.server_model.state_dict(),
                    "outer_params" : [p.data.clone().cpu() for p in self.diloco_ps.param_list_cpu]
                })
        else:
            self.model_queue.put({"model_id" : model_id})

    def aggregate_and_apply_outer_update(self):
        if not self.pseudo_gradient_accumulator:
            return

        averaged_pseudo_gradient = [
            sum(pgs[i] for pgs in self.pseudo_gradient_accumulator) / len(self.pseudo_gradient_accumulator)
            for i in range(len(self.pseudo_gradient_accumulator[0]))
        ]

        self.diloco_ps.outer_optimizer.zero_grad()
        sync_gradient = [g.to(DEVICE_SERVER) for g in averaged_pseudo_gradient]
        for p_outer, g_outer in zip(self.diloco_ps.param_list_cpu, sync_gradient):
            p_outer.grad = g_outer
        self.diloco_ps.outer_optimizer.step()