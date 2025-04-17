# 利用NeMo 2.0 + Nemo-Run 从HuggingFace导入DeepSeek-V3 模型并进行Finetune

## 1. 从HuggingFace 导入DeepSeek-V3 模型
> **说明1**：DeepSeek-V3的HuggingFace模型处理和其他常规LLM模型略有不同，由于HuggingFace没有正式支持FP8的量化，所以我们需要将其转换成BF16的checkpoint。

>**说明2**：我们下面的操作将会分为client端和cluster端，client端是利用NeMo-Run去实际发射任务提交的，cluster端则是我们真正执行模型权重转换和finetune的集群。checkpoint的下载和转换则是在cluster上进行的。
### HuggingFace下载checkpoint并进行转换
从HuggingFace clone DeepSeek-V3 weights，由于权重存储量大，所以该步耗时较长
```bash
git lfs install
git clone https://huggingface.co/deepseek-ai/DeepSeek-V3-Base
```

从DeepSeek-V3的官方github上clone代码，用于转换weights
```bash
git clone https://github.com/deepseek-ai/DeepSeek-V3.git
```
我们要用到的文件是 `DeepSeek-V3/inference/fp8_cast_bf16.py`, 需要确认的是这个文件的第[88行](https://github.com/deepseek-ai/DeepSeek-V3/blob/4cc6253d5c225e2c5fea32c54573449c1c46470a/inference/fp8_cast_bf16.py#L88)是否是如下：
```python
        save_file(new_state_dict, new_safetensor_file)
```
我们需要通过如下命令，
```bash
cd DeepSeek-V3/inference
sed -i '88{s/new_safetensor_file/new_safetensor_file, metadata={"format": "pt"}/}' fp8_cast_bf16.py
```
将这行代码改为：
```python
        save_file(new_state_dict, new_safetensor_file, metadata={"format": "pt"})
```
接下来，我们就可以用`fp8_cast_bf16.py`来转换DeepSeek-V3的权重文件了
```bash
python fp8_cast_bf16.py --input-fp8-hf-path ../../DeepSeek-V3-Base --output-bf16-hf-path ../../DeepSeek-V3-Base-BF16
```
>**说明**：这里`../../DeepSeek-V3-Base`是我们从HuggingFace下载的原始DeepSeek-V3的权重文件，我们导出的权重文件则放到`../../DeepSeek-V3-Base-BF16`

由于这个过程时间过长，我们依然可以利用slurm脚本让其在集群上后台进行执行，这里以EOS集群为例，可以参考`convert_DS_hf_to_bf16.sh`, 用`sbatch convert_DS_hf_to_bf16.sh`。具体提交作业前，请根据集群的情况修改`convert_DS_hf_to_bf16.sh`中的参数(尤其是`run_cmd`和`--container-mounts`中的路径对应关系)。

拷贝剩余文件：
```bash
# 进入HuggingFace原始checkpoint目录，将里面的多个文件拷贝到转换后的目录中
cd DeepSeek-V3-Base
cp tokenizer_config.json tokenizer.json modeling_deepseek.py configuration_deepseek.py ../DeepSeek-V3-Base-BF16/
```
>**注意**：当前，个别NGC container中的NeMo 版本会有哦一些bug，所以我们还需要将DeepSeek-R1中的[`generation_config.json`](https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/generation_config.json)文件拷贝到我们转换后的DeepSeek checkpoint中，即，../DeepSeek-V3-Base-BF16/,请手动自行下载并防止到../DeepSeek-V3-Base-BF16/下。

拷贝config.json 文件，并移除`quantization_config`字段：
```bash
jq 'del(.quantization_config)' DeepSeek-V3-Base/config.json > DeepSeek-V3-Base-BF16/config.json
```
## 2. 转换HuggingFace checkpoint到`.nemo`格式
NVIDIA NeMo 2.0的官方文档中给出了[使用NeMo 2.0 API的方式对DeepSeek-V3的HuggingFace checkpoint进行转换](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/deepseek_v3.html#nemo-2-0-finetuning-recipes)，但是由于DeepSeek-V3模型较大，后续finetune往往使用多节点进行finetune，需要结合调度系统进行任务提交，所以下面我们重点介绍使用NeMo-Run借助Slurm的方式进行任务提交。

**下面我们来对该环节进行解释，如果想直接执行，请手动根据实际情况修改`finetune_DeepSeek_V3_with_NeMo_Run.py`文件，在任意带有 NGC nvcr.io/nvidia/nemo:25.02.01的环境中执行脚本，请直接参考[基于NeMo-Run的任务提交配置](#5-基于nemo-run的任务提交配置)

在python文件`finetune_DeepSeek_V3_with_NeMo_Run.py`中，我们采用函数`configure_checkpoint_conversion()`对DeepSeek-V3的HuggingFace checkpoint进行序列化，以便利用NeMo-Run进行任务提交，即：
```python
def configure_checkpoint_conversion():
    return run.Partial(
        llm.import_ckpt,
        model = llm.deepseek_v3.model(),
        source="hf:///DeepSeek-V3-Base-BF16",
        overwrite=False,
    )
```
这里需要注意的是：
- `source="hf:///DeepSeek-V3-Base-BF16"` 指明HuggingFace的文件位置。前缀`hf:`是一定要加的，表明文件的形式是HuggingFace的。
- 前缀中的`///`是有讲究的，
    - 这里的三条`/`表明来自本地目录，需要和后面的函数`slurm_executor`相互配合，需要通过`custom_mounts`将本地目录与container中的目录进行挂载映射，即代码中的`Processed_HF_CKPT`变量对应的映射关系。
    - 如果是两条`/`,即`hf://`，则不用考虑本地目录的映射关系，因为会直接通过网络拉取HuggingFace的权重文件。

转换后的`.nemo`格式的权重文件会存储在环境变量`NEMO_HOME`下面的文件夹中，关于环境变量`NEMO_HOME`的设置与映射关系，参见[`NEMO_HOME`环境变量与容器挂载映射目录](#52-nemo_home环境变量与容器挂载映射目录)
```bash
tree
.
|-- datasets
|   `-- squad
|       |-- test.jsonl
|       |-- training.jsonl
|       |-- training.jsonl.idx.info
|       |-- training.jsonl.idx.npy
|       |-- validation.jsonl
|       |-- validation.jsonl.idx.info
|       `-- validation.jsonl.idx.npy
`-- models
    |-- DeepSeek-V3-Base-BF16
    |   |-- context
    |   |   |-- io.json
    |   |   |-- model.yaml
    |   |   `-- nemo_tokenizer
    |   |       |-- special_tokens_map.json
    |   |       |-- tokenizer.json
    |   |       `-- tokenizer_config.json
    |   `-- weights
    |       |-- __0_0.distcp
    |       |-- __0_1.distcp
    |       |-- common.pt
    |       `-- metadata.json
```
可以看到经过转换后，DeepSeek-V3的`.nemo`格式的权重已经被放置到了`NEMO_HOME`所对应的文件夹下了，在该文件下的`models`的目录下，有对应的`DeepSeek-V3-Base-BF16`文件夹（包括`context`和`weights`两个）。

## 3. 为DeepSeek设置Finetune recipe
函数`configure_finetuning_recipe` 中定义了DeepSeek的finetune策略，即：
```python
def configure_finetuning_recipe(nodes: int = 1, gpus_per_node: int = 1):
    recipe = llm.recipes.deepseek_v3.finetune_recipe(
        dir="/nemo_practice/DeepSeek_V3",
        name = "DeepSeek_FT",
        num_nodes=nodes,
        num_gpus_per_node=gpus_per_node,
    )
    recipe.resume.restore_config.path="nemo://DeepSeek-V3-Base-BF16"
    return recipe
```
- 这里需要重点说明的是，我们需要通过`recipe.resume.restore_config.path`来指定我们转换后的`.nemo`格式的权重文件的位置。
- 用`nemo://`表示我们的权重文件存储在`NEMO_HOME`环境变量所对应的文件夹下，如果存储在其它位置，需要用`nemo:///`来指定`.nemo`的checkpoint的文件位置。
- 如果有其它超参活并行配置的修改，也需要在这个函数中指定，如更改pipeline并行的设置`recipe.trainer.strategy.pipeline_model_parallel_size=6`等。

## 4. 执行器的配置
在python文件`finetune_DeepSeek_V3_with_NeMo_Run.py`中，我们定义了执行器[`slurm_executor`](https://github.com/linmuchuiyang/NeMo_2.0_sample_code/blob/590ebbe43a5e8519b59343918fb9dd41b437ba15/DeepSeek/finetune_DeepSeek_V3_with_NeMo_Run.py#L26),可以在其中指定slurm的环境配置。

## 5. 基于NeMo-Run的任务提交配置
在函数[`run_finetuning_on_slurm`](https://github.com/linmuchuiyang/NeMo_2.0_sample_code/blob/590ebbe43a5e8519b59343918fb9dd41b437ba15/DeepSeek//finetune_DeepSeek_V3_with_NeMo_Run.py#L87)中, 我们重点配置了执行器（executor）、`NEMO_HOME`等环境变量、容器挂载映射目录等。

### 5.1 执行器的配置
两个executor，分别是用于finetune的`executor`和用于转换checkpoint的`import_executor`。`import_executor`比较容易，可以直接`clone()`finetune的`executor`并将节点数`nodes`和每个节点的任务数`ntasks_per_node`设置为1，即可，即：
```python
    import_executor = executor.clone()
    import_executor.nodes = 1
    import_executor.ntasks_per_node = 1
```

用于finetune的`executor`需要提供用户名，节点数，集群主机地址，分区，容器挂载映射关系等：
```python
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
```
### 5.2 `NEMO_HOME`环境变量与容器挂载映射目录
`NEMO_HOME`环境变量是NeMo 2.0 中的重点变量，是保存`.nemo`格式的checkpoint的默认文件夹，在函数[`run_finetuning_on_slurm`](https://github.com/linmuchuiyang/NeMo_2.0_sample_code/blob/590ebbe43a5e8519b59343918fb9dd41b437ba15/DeepSeek/finetune_DeepSeek_V3_with_NeMo_Run.py#L87)中，
我们通过如下代码定义了`NEMO_HOME`中的本地路径`EOS_DIST_FILE_PATH`，在容器中的路径`/nemo_practice/NEMO_HOME`, 并通过`.env_vars`进行了环境变量的声明：
```python
    NEMO_HOME = "/nemo_practice/NEMO_HOME" 
    EOS_DIST_FILE_PATH = "/absolute/path/of/your/cluster/checkpoints/NEMO_HOME"
    MOUNT_CKPT_TO_CONTAINER= EOS_DIST_FILE_PATH + ":" + NEMO_HOME
    ...
    executor = slurm_executor(
        ...
        custom_mounts=[..., MOUNT_CKPT_TO_CONTAINER, ...]
        ...
    ...
    executor.env_vars["NEMO_HOME"] = NEMO_HOME
```

### 5.3 NeMo-Run的序列化运行配置
通过如下代码：
```python
    with run.Experiment("DS-finetune-slurm") as exp:
        exp.add(import_ckpt, executor=import_executor, name = "import_DS_from_hf")
        exp.add(recipe, executor=executor, name="DS_peft_finetune")
        exp.run(sequential=True, tail_logs=True)
```
我们将从HuggingFace转换checkpoint(`import_ckpt`)，执行finetune(`recipe`)以序列化的方式放入了任务队列，这样，如果执行代码，代码就会将任务提交给集群（比如EOS集群），集群会依次生成两个任务，finetune的slurm任务会等待转换任务完成之后再申请节点资源进行运行。
```bash
(base) jzhai@login-eos01:/lustre/fsw/general_sa/jzhai/checkpoints/NEMO_HOME$ squeue -u jzhai
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           2505513  backfill general_    jzhai PD       0:00      1 (Priority)
           2505514  backfill general_    jzhai PD       0:00      5 (Dependency)
```
如上，finetune任务处于`(Dependency)`状态。
## 6. 任务提交
我们的任务提交脚本是在client端执行的，即任意一个可以执行nvcr.io/nvidia/nemo:25.02.01镜像的机器上，作业真正执行则在远端的集群上（如EOS集群），所以我们需要在client端根据[5. 基于NeMo-Run的任务提交配置](#5-基于nemo-run的任务提交配置)修改完脚本之后，执行如下：
```bash
python finetune_DeepSeek_V3_with_NeMo_Run.py
```
这样，在client端我们则会看到如下输出：

<details>
<summary>点击展开client端日志</summary>

```bash
[NeMo W 2025-04-16 03:37:13 nemo_logging:405] /opt/megatron-lm/megatron/core/transformer/cuda_graphs.py:741: SyntaxWarning: assertion is always true, perhaps remove parentheses?
      assert (
    
[WARNING  | py.warnings        ]: /usr/local/lib/python3.12/dist-packages/mamba_ssm/ops/selective_scan_interface.py:163: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  @custom_fwd

[WARNING  | py.warnings        ]: /usr/local/lib/python3.12/dist-packages/mamba_ssm/ops/selective_scan_interface.py:239: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
  @custom_bwd

[WARNING  | py.warnings        ]: /usr/local/lib/python3.12/dist-packages/mamba_ssm/ops/triton/layer_norm.py:985: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  @custom_fwd

[WARNING  | py.warnings        ]: /usr/local/lib/python3.12/dist-packages/mamba_ssm/ops/triton/layer_norm.py:1044: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
  @custom_bwd

[WARNING  | py.warnings        ]: /usr/local/lib/python3.12/dist-packages/mamba_ssm/distributed/tensor_parallel.py:25: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  @custom_fwd

[WARNING  | py.warnings        ]: /usr/local/lib/python3.12/dist-packages/mamba_ssm/distributed/tensor_parallel.py:61: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
  @custom_bwd

[WARNING  | py.warnings        ]: /usr/local/lib/python3.12/dist-packages/mamba_ssm/ops/triton/ssd_combined.py:757: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  @custom_fwd

[WARNING  | py.warnings        ]: /usr/local/lib/python3.12/dist-packages/mamba_ssm/ops/triton/ssd_combined.py:835: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
  @custom_bwd

──────────────────────────────────────────────────────── Entering Experiment DS-finetune-slurm with id: DS-finetune-slurm_1744774639 ────────────────────────────────────────────────────────
[03:37:19] Connecting to jzhai@login-eos                                                                                                                                        client.py:257
[03:37:19] INFO     Connected (version 2.0, client OpenSSH_8.9p1)                                                                                                           transport.py:1944
Password: 
[03:37:27] INFO     Authentication (keyboard-interactive) successful!                                                                                                       transport.py:1944
           INFO     rsyncing /root/.nemo_run/experiments/DS-finetune-slurm/DS-finetune-slurm_1744774639 to /lustre/fsw/general_sa/jzhai/checkpoints/DeepSeekV3/DS-finetune-slurm  rsync.py:37
                    ...                                                                                                                                                                      
(jzhai@login-eos) Password: 
[03:37:44] INFO     Successfully ran `rsync  -pthrvz  --rsh='ssh  -p 22 ' /root/.nemo_run/experiments/DS-finetune-slurm/DS-finetune-slurm_1744774639                              rsync.py:93
                    jzhai@login-eos:/lustre/fsw/general_sa/jzhai/checkpoints/DeepSeekV3/DS-finetune-slurm`                                                                                   
[03:37:44] Launching job import_DS_from_hf for experiment DS-finetune-slurm                                                                                                 experiment.py:744
           INFO     Packaging for job import_DS_from_hf in tunnel                                                                                                                slurm.py:546
                    SSHTunnel(job_dir='/lustre/fsw/general_sa/jzhai/checkpoints/DeepSeekV3/DS-finetune-slurm/DS-finetune-slurm_1744774639', host='login-eos', user='jzhai',                  
                    packaging_jobs={'import_DS_from_hf': PackagingJob(symlink=False, src_path=None, dst_path=None), 'DS_peft_finetune': PackagingJob(symlink=False,                          
                    src_path=None, dst_path=None)}, identity=None, shell=None, pre_command=None) already done. Skipping subsequent packagings.                                               
                    This may cause issues if you have multiple tasks with the same name but different packagers, as only the first packager will be used.                                    
[03:37:45] INFO     Launched app: slurm_tunnel://nemo_run/2505513                                                                                                             launcher.py:111
[03:37:45] Launching job DS_peft_finetune for experiment DS-finetune-slurm                                                                                                  experiment.py:744
           INFO     Packaging for job DS_peft_finetune in tunnel                                                                                                                 slurm.py:546
                    SSHTunnel(job_dir='/lustre/fsw/general_sa/jzhai/checkpoints/DeepSeekV3/DS-finetune-slurm/DS-finetune-slurm_1744774639', host='login-eos', user='jzhai',                  
                    packaging_jobs={'import_DS_from_hf': PackagingJob(symlink=False, src_path=None, dst_path=None), 'DS_peft_finetune': PackagingJob(symlink=False,                          
                    src_path=None, dst_path=None)}, identity=None, shell=None, pre_command=None) already done. Skipping subsequent packagings.                                               
                    This may cause issues if you have multiple tasks with the same name but different packagers, as only the first packager will be used.                                    
           INFO     Launched app: slurm_tunnel://nemo_run/2505514                                                                                                             launcher.py:111
─────────────────────────────────────────────────────────────── Waiting for Experiment DS-finetune-slurm_1744774639 to finish ───────────────────────────────────────────────────────────────

Experiment Status for DS-finetune-slurm_1744774639

Task 0: import_DS_from_hf
- Status: SUBMITTED
- Executor: SlurmExecutor on jzhai@login-eos
- Job id: 2505513
- Local Directory: /root/.nemo_run/experiments/DS-finetune-slurm/DS-finetune-slurm_1744774639/import_DS_from_hf
- Remote Directory: /lustre/fsw/general_sa/jzhai/checkpoints/DeepSeekV3/DS-finetune-slurm/DS-finetune-slurm_1744774639/import_DS_from_hf

Task 1: DS_peft_finetune
- Status: SUBMITTED
- Executor: SlurmExecutor on jzhai@login-eos
- Job id: 2505514
- Local Directory: /root/.nemo_run/experiments/DS-finetune-slurm/DS-finetune-slurm_1744774639/DS_peft_finetune
- Remote Directory: /lustre/fsw/general_sa/jzhai/checkpoints/DeepSeekV3/DS-finetune-slurm/DS-finetune-slurm_1744774639/DS_peft_finetune

           INFO     Waiting for job 2505513 to finish [log=True]...                                                                                                           launcher.py:130
           INFO     Waiting for job 2505514 to finish [log=True]...                                                                                                           launcher.py:130
           Waiting for app state response before fetching logs...                                                                                                                 logs.py:105
           Waiting for app state response before fetching logs...                                                                                                                 logs.py:105
[03:37:56] Connecting to jzhai@login-eos                                                                                                                                        client.py:257
           Connecting to jzhai@login-eos                                                                                                                                        client.py:257
[03:37:56] INFO     Connected (version 2.0, client OpenSSH_8.9p1)                                                                                                           transport.py:1944
           INFO     Connected (version 2.0, client OpenSSH_8.9p1)                                                                                                           transport.py:1944
Password: Password:            Connecting to jzhai@login-eos                                                                                                                                        client.py:257
           INFO     Connected (version 2.0, client OpenSSH_8.9p1)                                                                                                           transport.py:1944
           Connecting to jzhai@login-eos                                                                                                                                        client.py:257
           INFO     Connected (version 2.0, client OpenSSH_8.9p1)                                                                                                           transport.py:1944
Password: Password: 
[03:38:32] INFO     Authentication (keyboard-interactive) successful!                                                                                                       transport.py:1944
                                                                                                                                                                                             
# The experiment was run with the following tasks: ['import_DS_from_hf', 'DS_peft_finetune']                                                                                                 
# You can inspect and reconstruct this experiment at a later point in time using:                                                                                                            
experiment = run.Experiment.from_id("DS-finetune-slurm_1744774639")                                                                                                                          
experiment.status() # Gets the overall status                                                                                                                                                
experiment.logs("import_DS_from_hf") # Gets the log for the provided task                                                                                                                    
experiment.cancel("import_DS_from_hf") # Cancels the provided task if still running                                                                                                          
                                                                                                                                                                                             
                                                                                                                                                                                             
# You can inspect this experiment at a later point in time using the CLI as well:                                                                                                            
nemo experiment status DS-finetune-slurm_1744774639                                                                                                                                          
nemo experiment logs DS-finetune-slurm_1744774639 0                                                                                                                                          
nemo experiment cancel DS-finetune-slurm_1744774639 0
```

</details>


说明：
- 代码执行过程中，需要输入远端集群的ssh账户密码，手动输入即可
- 日志中，两个作业提交成功会给出`Status: SUBMITTED`的提示
- 日志中，提供了在client端查作业在集群上运行情况的命令，可以直接拷贝命令在client端执行。

我们也可以通过slurm命令`squeue -u 用户名` 在远端集群中查看作业的运行状态。

在远端集群中，可以看到两个slurm任务分别产生了`sbatch`的作业提交脚本，并生成了对应的日志，同时，每个任务重有对应的NeMo的`yaml`配置文件生成。
```bash
tree
.
|-- DS_peft_finetune
|   |-- code
|   |-- configs
|   |   |-- DS_peft_finetune_config.yaml
|   |   |-- DS_peft_finetune_executor.yaml
|   |   |-- DS_peft_finetune_fn_or_script
|   |   `-- DS_peft_finetune_packager
|   |-- log-general_sa-sa.DS_peft_finetune_2464681_0.out
|   `-- sbatch_general_sa-sa.DS_peft_finetune_2464681.out
|-- DS_peft_finetune_sbatch.sh
|-- _CONFIG
|-- _TASKS
|-- _VERSION
|-- __main__.py
|-- import_DS_from_hf
|   |-- code
|   |-- configs
|   |   |-- import_DS_from_hf_config.yaml
|   |   |-- import_DS_from_hf_executor.yaml
|   |   |-- import_DS_from_hf_fn_or_script
|   |   `-- import_DS_from_hf_packager
|   |-- log-general_sa-sa.import_DS_from_hf_2464680_0.out
|   `-- sbatch_general_sa-sa.import_DS_from_hf_2464680.out
`-- import_DS_from_hf_sbatch.sh

6 directories, 18 files
```

两个任务的日志说明：
- checkpoint格式转换的任务要确保如下日志，
```bash
....
[WARNING  | py.warnings        ]: /usr/local/lib/python3.12/dist-packages/mamba_ssm/ops/triton/ssd_combined.py:835: FutureWarning: `torch.cuda.amp.custom_bwd(args...)` is deprecated. Please use `torch.amp.custom_bwd(args..., device_type='cuda')` instead.
  @custom_bwd

 $NEMO_MODELS_CACHE=/nemo_practice/NEMO_HOME/models 
✓ Checkpoint imported to /nemo_practice/NEMO_HOME/models/DeepSeek-V3-Base-BF16
```
- finetune的任务则需要确保训练的过程真正启动
```bash
...
[NeMo W 2025-04-14 00:11:22 rerun_state_machine:1088] Implicit initialization of Rerun State Machine!
[NeMo W 2025-04-14 00:11:22 rerun_state_machine:211] RerunStateMachine initialized in mode RerunMode.DISABLED
Training epoch 0, iteration 0/999 | lr: 1.961e-06 | global_batch_size: 128 | global_step: 0 | reduced_train_loss: 2.515
Training epoch 0, iteration 1/999 | lr: 3.922e-06 | global_batch_size: 128 | global_step: 1 | reduced_train_loss: 2.65 | consumed_samples: 256
Training epoch 0, iteration 2/999 | lr: 5.882e-06 | global_batch_size: 128 | global_step: 2 | reduced_train_loss: 2.464 | consumed_samples: 384
Training epoch 0, iteration 3/999 | lr: 7.843e-06 | global_batch_size: 128 | global_step: 3 | reduced_train_loss: 2.629 | consumed_samples: 512
Training epoch 0, iteration 4/999 | lr: 9.804e-06 | global_batch_size: 128 | global_step: 4 | reduced_train_loss: 2.362 | consumed_samples: 640
Training epoch 0, iteration 5/999 | lr: 1.176e-05 | global_batch_size: 128 | global_step: 5 | reduced_train_loss: 2.282 | consumed_samples: 768
Training epoch 0, iteration 6/999 | lr: 1.373e-05 | global_batch_size: 128 | global_step: 6 | reduced_train_loss: 2.191 | consumed_samples: 896
Training epoch 0, iteration 7/999 | lr: 1.569e-05 | global_batch_size: 128 | global_step: 7 | reduced_train_loss: 2.464 | consumed_samples: 1024
...
```
