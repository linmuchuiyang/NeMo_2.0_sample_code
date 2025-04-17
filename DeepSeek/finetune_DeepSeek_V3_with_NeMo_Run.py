#nvcr.io/nvidia/nemo:25.02.01
import nemo_run as run
from nemo.collections import llm
from typing import Optional

def configure_checkpoint_conversion():
    return run.Partial(
        llm.import_ckpt,
        model = llm.deepseek_v3.model(),
        source="hf:///DeepSeek-V3-Base-BF16",
        overwrite=False,
    )

def configure_finetuning_recipe(nodes: int = 1, gpus_per_node: int = 1):
    recipe = llm.recipes.deepseek_v3.finetune_recipe(
        dir="/nemo_practice/DeepSeek_V3",
        name = "DeepSeek_FT",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
    )
    # 暂时添加下面的内容，因为原始配置里面PP的设定导致bug
    #recipe.trainer.strategy.pipeline_model_parallel_size = 6
    recipe.resume.restore_config.path="nemo://DeepSeek-V3-Base-BF16"
    return recipe

def slurm_executor(
    user: str,
    host: str,
    remote_job_dir: str,
    account: str,
    partition: str,
    nodes: int,
    devices: int,
    time: str = "02:00:00",
    custom_mounts: Optional[list[str]] = None,
    custom_env_vars: Optional[dict[str, str]] = None,
    container_image: str = "nvcr.io#nvidia/nemo:25.02.01",   #nvcr.io#nvidia/nemo:25.02.rc6 nvcr.io/nvidia/nemo:25.02.01# 注意：原始参考文档写的nvcr.io 后面是 / 而实际在EOS集群测试发现，必须是# 才可以提交，不同集群也许不一样，这里要注意。
    retries: int = 0,
    )   -> run.SlurmExecutor:
    if not (user and host and remote_job_dir and account and partition and nodes and devices):
        raise RuntimeError(
            "Please set user, host, remote_job_dir, account, partition, nodes and devices args for using this function."
        )

    mounts = []
    # Custom mounts are defined here.
    if custom_mounts:
        mounts.extend(custom_mounts)
    
    # Env vars for jobs are configured here
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
    }
    if custom_env_vars:
        env_vars |= custom_env_vars

    # This defines the slurm executor.
    # We connect to the executor via the tunnel defined by user, host and remote_job_dir.
    executor = run.SlurmExecutor(
        account=account,
        partition=partition,
        tunnel=run.SSHTunnel(
            user=user,
            host=host,
            job_dir=remote_job_dir, # This is where the results of the run will be stored by default.
            # identity="/path/to/identity/file" OPTIONAL: Provide path to the private key that can be used to establish the SSH connection without entering your password.
        ),
        nodes=nodes,
        ntasks_per_node=devices,
        #gpus_per_node=devices, # EOS 集群上没有这个参数，需要去掉
        mem="0",
        exclusive=True,
        #gres="gpu:1",          # EOS 集群上没有这个参数，需要去掉
        packager=run.Packager(),
    )
    executor.container_image = container_image
    executor.container_mounts = mounts
    executor.env_vars = env_vars
    executor.retries = retries
    executor.time = time
    
    return executor   

def run_finetuning_on_slurm():
    import_ckpt = configure_checkpoint_conversion()

    NEMO_HOME = "/nemo_practice/NEMO_HOME" 
    EOS_DIST_FILE_PATH = "/absolute/path/of/your/cluster/checkpoints/NEMO_HOME"
    MOUNT_CKPT_TO_CONTAINER= EOS_DIST_FILE_PATH + ":" + NEMO_HOME

    Processed_HF_CKPT = "/absolute/path/of/converted/ckpt/DeepSeek-V3-Base-BF16:/DeepSeek-V3-Base-BF16"

    recipe = configure_finetuning_recipe(gpus_per_node=8, nodes=5)
    executor = slurm_executor(
        user="your_account_name_of_slurm_cluster",
        host="slurm_cluster_host",
        remote_job_dir = "/absolute/path/of/your/job_dir",
        account="your_slurm_cluster_account",
        partition="your_slurm_cluster_partition", 
        custom_mounts=["/absolute/path/of/DeepSeekV3/ckpt_path:/nemo_practice/DeepSeek_V3", MOUNT_CKPT_TO_CONTAINER, Processed_HF_CKPT], # 前提要确保EOS上的文件夹路径存在，否则会报错，因为container没有权限去创建文件夹。
        # 这里map的是上面configure_recipe里面的checkpoint dir的文件夹。
        container_image="nvcr.io#nvidia/nemo:25.02.01",
        #nodes=recipe.trainer.num_nodes,
        nodes=5, #4, 
        #devices=recipe.trainer.devices,
        devices=8,
    )
    executor.env_vars["NEMO_HOME"] = NEMO_HOME

    import_executor = executor.clone()
    import_executor.nodes = 1
    import_executor.ntasks_per_node = 1

    with run.Experiment("DS-finetune-slurm") as exp:
        exp.add(import_ckpt, executor=import_executor, name = "import_DS_from_hf")
        exp.add(recipe, executor=executor, name="DS_peft_finetune")
        exp.run(sequential=True, tail_logs=True)

if __name__ == "__main__":
    run_finetuning_on_slurm()

